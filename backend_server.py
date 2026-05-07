#!/usr/bin/env python3
"""
backend_server.py — FastAPI backend for Agentic RAG Room Layout Recommendation.
Run: uvicorn backend_server:app --host 0.0.0.0 --port 8000

Env vars:
  LAYOUTS_PATH      path to layouts.json        (default: layouts.json)
  LLM_MODEL_ID      Ollama model tag             (default: qwen3.6:35b)
  EMBEDDER_MODEL    sentence-transformers model  (default: BAAI/bge-m3)
  OLLAMA_HOST       Ollama API base URL          (default: http://localhost:11434)
  DIRECT_THRESHOLD  layouts left before Phase 2 (default: 5)
  N_SAMPLE          cluster sample size          (default: 10)
  MAX_K             max cluster count            (default: 10)
"""

import os, re, json, random, threading
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from ollama import Client

KEEP_ALL_SENTINEL = "__KEEP_ALL__"
UNSURE_SENTINEL   = "__UNSURE__"

DIRECT_THRESHOLD = int(os.environ.get("DIRECT_THRESHOLD", 5))
N_SAMPLE         = int(os.environ.get("N_SAMPLE",         10))
MAX_K            = int(os.environ.get("MAX_K",            10))
EMBEDDER_MODEL   = os.environ.get("EMBEDDER_MODEL", "BAAI/bge-m3")
LLM_MODEL_ID     = os.environ.get("LLM_MODEL_ID",   "qwen3.6:35b")
OLLAMA_HOST      = os.environ.get("OLLAMA_HOST",     "http://localhost:11434")
LAYOUTS_PATH     = os.environ.get("LAYOUTS_PATH",    "layouts1.json")


def load_layouts_from_json(filepath: str) -> tuple[list[dict], list[str]]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Layout JSON not found: {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("JSON root must be a list of layout entries.")
    layouts:       list[dict] = []
    all_furniture: set[str]   = set()
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        query_id    = str(entry.get("query_id", f"layout_{idx:05d}"))
        description = str(entry.get("prompt",   "")).strip()
        room_size   = entry.get("room_size",   [1, 1])
        obj_list    = entry.get("object_list", [])
        if not description:
            description = f"Room layout {query_id}."
        W = float(room_size[0]) if room_size[0] > 0 else 1.0
        H = float(room_size[1]) if room_size[1] > 0 else 1.0
        layout: dict = {"id": query_id, "description": description}
        for item in obj_list:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                continue
            name   = str(item[0]).lower().strip()
            coords = item[1]
            if not (isinstance(coords, (list, tuple)) and len(coords) == 4):
                continue
            x1, y1, x2, y2 = [float(c) for c in coords]
            if max(x1, y1, x2, y2) > 1.0:
                x1, y1, x2, y2 = x1 / W, y1 / H, x2 / W, y2 / H
            x1, y1, x2, y2 = (
                max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1)),
                max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2)),
            )
            layout[name] = [x1, y1, x2, y2]
            all_furniture.add(name)
        layouts.append(layout)
    if not layouts:
        raise ValueError("No valid layouts found in the JSON file.")
    return layouts, sorted(all_furniture)


def filter_layouts_by_furniture(
    layouts: list[dict],
    required_furniture: list[str],
) -> tuple[list[dict], list[dict]]:
    if not required_furniture:
        return list(layouts), []
    matching, excluded = [], []
    for layout in layouts:
        missing = [f for f in required_furniture if f not in layout]
        if missing:
            excluded.append(layout)
        else:
            matching.append(layout)
    return matching, excluded


class UserContext:
    def __init__(self):
        self.likes:    list[str] = []
        self.dislikes: list[str] = []
        self.unsure:   list[str] = []

    def add_like(self, fact: str):
        if fact and fact not in self.likes:
            self.likes.append(fact)

    def add_dislike(self, fact: str):
        if fact and fact not in self.dislikes:
            self.dislikes.append(fact)

    def add_unsure(self, topic: str):
        if topic and topic not in self.unsure:
            self.unsure.append(topic)

    def as_prompt_block(self) -> str:
        if not self.likes and not self.dislikes and not self.unsure:
            return ""
        lines = ["\nWhat we know about the user so far:"]
        if self.likes:
            lines.append("  Definitely wants: " + "; ".join(self.likes))
        if self.dislikes:
            lines.append("  Definitely does NOT want: " + "; ".join(self.dislikes))
        if self.unsure:
            lines.append("  Still undecided about: " + "; ".join(self.unsure))
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict:
        return {"likes": self.likes, "dislikes": self.dislikes, "unsure": self.unsure}


class LayoutVectorStore:
    def __init__(self, model_name: str = EMBEDDER_MODEL):
        print(f"[VectorStore] Loading embedder: {model_name}")
        self.embedder        = SentenceTransformer(model_name)
        self.active_layouts: list[dict] = []
        self.embeddings:     np.ndarray  = np.array([])

    def build(self, layouts: list[dict]):
        self.active_layouts = list(layouts)
        self._recompute_embeddings()
        print(f"[VectorStore] Built — {len(self.active_layouts)} layouts, dim={self.embeddings.shape[1]}")

    def _recompute_embeddings(self):
        descriptions    = [l["description"] for l in self.active_layouts]
        self.embeddings = self.embedder.encode(
            descriptions, normalize_embeddings=True, show_progress_bar=False
        )

    def compute_dynamic_k(self) -> int:
        n   = len(self.active_layouts)
        cap = min(n - 1, MAX_K)
        if cap < 2:
            return 2
        if n <= 4:
            return 2
        best_k, best_score = 2, -1.0
        for k in range(2, cap + 1):
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.embeddings)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(self.embeddings, labels, metric="cosine")
            if score > best_score:
                best_score = score
                best_k     = k
        return best_k

    def cluster(self, k: int) -> dict[int, list[dict]]:
        k      = min(k, len(self.active_layouts))
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(self.embeddings)
        clusters: dict[int, list[dict]] = {}
        for layout, label in zip(self.active_layouts, labels):
            clusters.setdefault(int(label), []).append(layout)
        return clusters

    def eliminate(self, layout_ids: list[str]):
        remove    = set(layout_ids)
        survivors = [(l, e) for l, e in zip(self.active_layouts, self.embeddings)
                     if l["id"] not in remove]
        if not survivors:
            return
        self.active_layouts, emb_list = zip(*survivors)
        self.active_layouts = list(self.active_layouts)
        self.embeddings     = np.array(emb_list, dtype=np.float32)

    @property
    def count(self) -> int:
        return len(self.active_layouts)


# ══════════════════════════════════════════════════════════════════════════════
# 4. LLM AGENT  (Qwen3.6 35B via Ollama — think=False passed natively)
# ══════════════════════════════════════════════════════════════════════════════
class QwenAgent:

    def __init__(self, model_id: str = LLM_MODEL_ID, host: str = OLLAMA_HOST):
        print(f"[LLM] Connecting to Ollama at {host} — model: {model_id}")
        self.model_id = model_id
        self.client   = Client(host=host)
        print(f"[LLM] Ready — thinking disabled globally via think=False\n")

    # ── FIX: use think=False as top-level client.chat() param ─────────────
    # Prepending /no_think to the message content causes empty output.
    # think=False is Ollama's native way to disable the reasoning chain.
    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float  = 0.3,
    ) -> str:
        response = self.client.chat(
            model=self.model_id,
            think=False,                      # ✅ disables thinking natively
            messages=[
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature":    temperature,
                "num_predict":    max_new_tokens,
                "top_p":          0.9,
                "top_k":          30,
                "repeat_penalty": 1.1,
            },
        )

        raw = response["message"]["content"]

        # Strip any residual <think>...</think> blocks just in case
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"^[\n\r\s]+", "", raw).strip()

        return raw

    @staticmethod
    def _first_line(text: str) -> str:
        for line in text.strip().split("\n"):
            line = line.strip()
            if line:
                return line
        return ""

    @staticmethod
    def _history_block(asked_questions: list[str]) -> str:
        if not asked_questions:
            return ""
        lines = "\n".join(f"  - {q}" for q in asked_questions)
        return f"\nQuestions already asked (DO NOT repeat these topics):\n{lines}\n"

    def refine_user_answer(self, question: str, raw_answer: str) -> str:
        prompt = (
            "Rewrite the user's raw answer into a single, clear, unambiguous English sentence under 30 words.\n\n"
            "<question>\n"
            f"{question}\n"
            "</question>\n\n"
            "<raw_answer>\n"
            f"{raw_answer}\n"
            "</raw_answer>\n\n"
            "Rules:\n"
            "1. Preserve the exact original intent. Do not add any new opinions or context.\n"
            "2. If the raw answer is already a clear, complete sentence, output it exactly as is.\n"
            "3. Output ONLY the final sentence. No prefixes, no explanations, no quotes."
        )
        refined = self._first_line(
            self._generate(prompt, max_new_tokens=1024, temperature=0.1)
        )
        return refined if refined else raw_answer

    def summarise_cluster(self, descriptions: list[str]) -> str:
        sample = random.sample(descriptions, min(N_SAMPLE, len(descriptions)))
        layouts_text = "\n".join(f"<layout>\n{desc}\n</layout>" for desc in sample)
        prompt = (
            "Summarize the shared spatial arrangement from the provided room layout descriptions into a single sentence.\n\n"
            "<layouts>\n"
            f"{layouts_text}\n"
            "</layouts>\n\n"
            "Rules:\n"
            "1. Write exactly ONE sentence (maximum 40 words).\n"
            "2. Capture ALL key spatial details: positions of ALL furniture and their relative placement.\n"
            "3. Describe how open or compact the room feels.\n"
            "4. Use plain English only without jargon.\n"
            "5. Output ONLY the final summary sentence. Do not include prefixes, explanations, or quotes."
        )
        return self._first_line(
            self._generate(prompt, max_new_tokens=1024, temperature=0.2)
        )

    def generate_cluster_question(
        self,
        cluster_summaries: dict[int, str],
        asked_questions:   list[str],
        user_ctx:          UserContext,
    ) -> tuple[str, dict[int, int]]:

        option_map   = {i + 1: cid for i, cid in enumerate(cluster_summaries)}
        styles_text  = "\n".join(f"Style {i+1}: {cluster_summaries[cid]}" for i, cid in enumerate(cluster_summaries))
        history_text = self._history_block(asked_questions)
        ctx_text     = user_ctx.as_prompt_block()

        prompt = (
            "Analyze the provided room styles, user context, and conversation history to formulate a single question that determines the user's preference.\n\n"
            "<styles>\n"
            f"{styles_text}\n"
            "</styles>\n\n"
            "<user_context>\n"
            f"{ctx_text}\n"
            "</user_context>\n\n"
            "<history>\n"
            f"{history_text}\n"
            "</history>\n\n"
            "Rules for the question:\n"
            "1. Identify a concrete spatial feature (e.g., wardrobe placement, density) that clearly DIFFERS between at least two styles.\n"
            "2. Write exactly ONE question (maximum 30 words) that presents a clear 'this or that' choice based on these differences.\n"
            "3. Example format: 'Would you prefer the wardrobe close to the bed, or placed across the room to create distinct zones?'\n"
            "4. Do NOT ask open-ended questions like 'How is the wardrobe positioned?'. You MUST provide the specific options within the question itself.\n"
            "5. Do NOT ask about features shared by all styles.\n"
            "6. Do NOT introduce furniture or topics not explicitly mentioned in the <styles> block.\n"
            "7. Do NOT ask about topics already covered in the <history> block or decided in the <user_context> block.\n"
            "8. Do NOT list numbered options or reference the styles directly (e.g., never say 'Style 1' or 'Option A').\n"
            "9. Use plain, friendly English and end with a '?'.\n"
            "10. Output ONLY the final question. No prefixes, explanations, or formatting."
        )
        question = self._first_line(
            self._generate(prompt, max_new_tokens=1024, temperature=0.4)
        )
        return question, option_map

    def detect_unsure(self, question: str, user_input: str) -> bool:
        prompt = (
            "Analyze the user's response to the given question to determine if they are uncertain or if they have a clear preference.\n\n"
            "<question>\n"
            f"{question}\n"
            "</question>\n\n"
            "<response>\n"
            f"{user_input}\n"
            "</response>\n\n"
            "Rules:\n"
            "1. Output exactly the word 'UNSURE' if the user is uncertain, says 'I don't know', 'not sure', 'maybe', 'either', 'both', 'no preference', or 'doesn't matter'.\n"
            "2. Output exactly the word 'CLEAR' if the user expresses any definite preference or choice.\n"
            "3. Output ONLY a single word: either UNSURE or CLEAR. Do not include any explanations, punctuation, or quotes."
        )
        raw = self._first_line(
            self._generate(prompt, max_new_tokens=1024, temperature=0.0)
        ).strip().upper()
        return "UNSURE" in raw

    def interpret_cluster_choice(
        self,
        question:          str,
        cluster_summaries: dict[int, str],
        user_input:        str,
        option_map:        dict[int, int],
    ) -> list[int]:

        styles_text = "\n".join(
            f"Style {num}: {cluster_summaries[cid]}"
            for num, cid in option_map.items()
        )
        prompt = (
            "Analyze the user's response to the question against the provided style summaries. "
            "Determine which style numbers CONTRADICT or do NOT match the user's preferences.\n\n"
            "<question>\n"
            f"{question}\n"
            "</question>\n\n"
            "<styles>\n"
            f"{styles_text}\n"
            "</styles>\n\n"
            "<response>\n"
            f"{user_input}\n"
            "</response>\n\n"
            "Rules:\n"
            "1. Output a comma-separated list of style numbers (e.g., 1, 3) that should be eliminated.\n"
            f"2. If and ONLY if the user's response is completely unrelated to the spatial differences between the styles, output exactly '{KEEP_ALL_SENTINEL}'.\n"
            f"3. If the user expresses ANY clear preference, even partial, you MUST eliminate at least one style. Do NOT use '{KEEP_ALL_SENTINEL}'.\n"
            "4. NEVER eliminate ALL styles. Always leave at least one style uneliminated.\n"
            "5. You MUST wrap your final output inside <answer> tags. Example: <answer>1, 3</answer> or <answer>KEEP_ALL</answer>.\n"
            "6. Output nothing else outside the <answer> tags."
        )
        raw_output = self._generate(prompt, max_new_tokens=2048, temperature=0.1)

        match = re.search(r"<answer>(.*?)</answer>", raw_output, flags=re.DOTALL | re.IGNORECASE)
        raw   = match.group(1).strip() if match else self._first_line(raw_output).strip()

        print(f"[LLM] Raw interpretation of user answer: '{raw}'")

        if KEEP_ALL_SENTINEL not in raw and raw.strip():
            eliminate_ids: set[int] = set()
            for token in raw.replace(",", " ").split():
                if token.isdigit():
                    num = int(token)
                    if num in option_map:
                        eliminate_ids.add(option_map[num])
            kept = [cid for cid in cluster_summaries.keys() if cid not in eliminate_ids]
            if kept and len(kept) < len(cluster_summaries):
                return kept

        return list(cluster_summaries.keys())

    def update_user_context(
        self,
        question:          str,
        user_input:        str,
        kept_summaries:    list[str],
        dropped_summaries: list[str],
        user_ctx:          UserContext,
    ):
        kept_txt    = "; ".join(kept_summaries)    or "none"
        dropped_txt = "; ".join(dropped_summaries) or "none"

        prompt = (
            "Extract concise preference facts based on the user's response to the question and the resulting kept/dropped layout styles.\n\n"
            "<question>\n"
            f"{question}\n"
            "</question>\n\n"
            "<response>\n"
            f"{user_input}\n"
            "</response>\n\n"
            "<kept_styles>\n"
            f"{kept_txt}\n"
            "</kept_styles>\n\n"
            "<dropped_styles>\n"
            f"{dropped_txt}\n"
            "</dropped_styles>\n\n"
            "Rules:\n"
            "1. Output EXACTLY two lines.\n"
            "2. Line 1 MUST start with 'LIKE: ' followed by a short phrase (maximum 10 words) describing what the user definitely wants.\n"
            "3. Line 2 MUST start with 'DISLIKE: ' followed by a short phrase (maximum 10 words) describing what the user definitely does NOT want.\n"
            "4. If nothing is certain for a category, output the word 'nothing' for that line (e.g., 'LIKE: nothing').\n"
            "5. Output ONLY these two lines. Do not add conversational filler, markdown formatting, or explanations."
        )
        raw = self._generate(prompt, max_new_tokens=1024, temperature=0.1)

        like_fact = dislike_fact = ""
        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("LIKE:"):
                like_fact    = line.split(":", 1)[1].strip()
            elif line.upper().startswith("DISLIKE:"):
                dislike_fact = line.split(":", 1)[1].strip()

        if like_fact    and like_fact.lower()    != "nothing": user_ctx.add_like(like_fact)
        if dislike_fact and dislike_fact.lower() != "nothing": user_ctx.add_dislike(dislike_fact)

    def generate_direct_question(
        self,
        layouts:         list[dict],
        asked_questions: list[str],
        user_ctx:        UserContext,
    ) -> str:

        layouts_text = "\n\n".join(f"Layout {l['id']}: {l['description']}" for l in layouts)
        history_text = self._history_block(asked_questions)
        ctx_text     = user_ctx.as_prompt_block()

        prompt = (
            "Analyze the provided room layouts, user context, and conversation history to formulate a single question that differentiates the options based on a concrete spatial feature.\n\n"
            "<layouts>\n"
            f"{layouts_text}\n"
            "</layouts>\n\n"
            "<user_context>\n"
            f"{ctx_text}\n"
            "</user_context>\n\n"
            "<history>\n"
            f"{history_text}\n"
            "</history>\n\n"
            "Rules:\n"
            "1. Focus on the biggest concrete spatial difference between the remaining layouts to help eliminate non-matching options.\n"
            "2. Write exactly ONE question (maximum 20 words) asking the user about this differing feature.\n"
            "3. Do NOT list numbered options, reference layout IDs, or ask about features shared by all layouts.\n"
            "4. Do NOT ask about topics already covered in the <history> block or already decided in the <user_context> block.\n"
            "5. Use plain, friendly English and end with a '?'.\n"
            "6. Output ONLY the final question. Do not include prefixes, explanations, or conversational filler."
        )
        return self._first_line(
            self._generate(prompt, max_new_tokens=1024, temperature=0.4)
        )

    def interpret_direct_elim(
        self,
        question:   str,
        layouts:    list[dict],
        user_input: str,
        user_ctx:   UserContext,
    ) -> list[str]:
        valid_ids  = {l["id"] for l in layouts}
        desc_block = "\n\n".join(f"Layout ID {l['id']}: {l['description']}" for l in layouts)
        ctx_block  = user_ctx.as_prompt_block()

        prompt = (
            "Analyze the user's response to determine which room layouts should be eliminated because they DO NOT match the user's preference.\n\n"
            "<question>\n"
            f"{question}\n"
            "</question>\n\n"
            "<response>\n"
            f"{user_input}\n"
            "</response>\n\n"
            "<user_context>\n"
            f"{ctx_block}\n"
            "</user_context>\n\n"
            "<layouts>\n"
            f"{desc_block}\n"
            "</layouts>\n\n"
            "Rules:\n"
            "1. Output a comma-separated list of Layout IDs (e.g., layout_01, layout_03) that contradict the user's response.\n"
            "2. If the user's response is unclear or does not contradict any layout, output exactly the word 'none'.\n"
            "3. NEVER eliminate all layouts. Ensure at least one Layout ID remains unlisted.\n"
            "4. Output ONLY the comma-separated IDs or the word 'none'. Do not include explanations, prefixes, or quotes."
        )
        raw = self._first_line(
            self._generate(prompt, max_new_tokens=1024, temperature=0.1)
        ).strip().lower()

        if raw == "none" or not raw:
            return []

        eliminate = [tok.strip() for tok in raw.split(",") if tok.strip() in valid_ids]

        if len(eliminate) >= len(layouts):
            eliminate = eliminate[: len(layouts) - 1]

        return eliminate

# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(title="Room Layout Recommendation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_session: dict = {
    "status":             "idle",
    "question":           None,
    "phase":              None,
    "turn":               0,
    "remaining_count":    0,
    "user_context":       {},
    "result":             None,
    "furniture_excluded": [],
    "process_eliminated": [],
    "all_furniture":      [],
    "required_furniture": [],
    "error":              None,
    "_user_answer":       None,
}
_lock         = threading.Lock()
_answer_event = threading.Event()
_worker: Optional[threading.Thread] = None


class StartRequest(BaseModel):
    furniture: str = ""


class AnswerRequest(BaseModel):
    answer: str


def _set(**kwargs):
    with _lock:
        _session.update(kwargs)


def _snap() -> dict:
    with _lock:
        return {k: v for k, v in _session.items() if not k.startswith("_")}


@app.get("/health")
def health():
    snap = _snap()
    return {"server": "ok", "session_status": snap["status"],
            "turn": snap["turn"], "remaining": snap["remaining_count"]}


@app.get("/question")
def get_question():
    return _snap()


@app.post("/answer")
def post_answer(req: AnswerRequest):
    with _lock:
        if _session["status"] != "waiting":
            raise HTTPException(
                status_code=400,
                detail=f"Not waiting for answer (status='{_session['status']}').",
            )
        _session["_user_answer"] = req.answer.strip()
        _session["status"]       = "processing"
    _answer_event.set()
    return {"ok": True, "message": "Answer received — CRS is processing."}


@app.post("/start")
def start_session(req: StartRequest):
    global _worker
    with _lock:
        if _session["status"] in ("initializing", "waiting", "processing"):
            raise HTTPException(
                status_code=409,
                detail=f"Session already running (status='{_session['status']}').",
            )
        _session.update({
            "status": "initializing", "question": None, "phase": None,
            "turn": 0, "remaining_count": 0, "user_context": {},
            "result": None, "furniture_excluded": [], "process_eliminated": [],
            "all_furniture": [], "required_furniture": [], "error": None,
            "_user_answer": None,
        })
        _answer_event.clear()
    _worker = threading.Thread(target=_crs_worker, args=(req,), daemon=True)
    _worker.start()
    return {"ok": True, "message": "Session started — poll GET /question."}


def _ask_and_wait(question: str, phase: str, remaining_count: int, user_ctx: UserContext) -> str:
    _set(status="waiting", question=question, phase=phase,
         remaining_count=remaining_count, user_context=user_ctx.to_dict())
    _answer_event.wait()
    _answer_event.clear()
    with _lock:
        answer = _session["_user_answer"]
    return answer or ""


def _crs_worker(req: StartRequest):
    try:
        print(f"\n[CRS] Loading: {LAYOUTS_PATH}")
        all_layouts, furniture_items = load_layouts_from_json(LAYOUTS_PATH)
        _set(all_furniture=furniture_items)

        required_furniture: list[str] = []
        if req.furniture:
            required_furniture = [
                f.strip().lower() for f in req.furniture.split(",")
                if f.strip().lower() in furniture_items
            ]
        _set(required_furniture=required_furniture)

        filtered_layouts, furniture_excluded = filter_layouts_by_furniture(
            all_layouts, required_furniture
        )
        _set(furniture_excluded=[l["id"] for l in furniture_excluded])

        if len(filtered_layouts) < 2:
            _set(status="error", error="Fewer than 2 layouts remain after furniture filtering.")
            return

        llm      = QwenAgent(model_id=LLM_MODEL_ID, host=OLLAMA_HOST)
        store    = LayoutVectorStore(model_name=EMBEDDER_MODEL)
        user_ctx = UserContext()
        store.build(filtered_layouts)

        asked_questions:    list[str] = []
        process_eliminated: list[str] = []
        turn = 0

        # Phase 1 — Cluster-Based Elimination
        print("[CRS] Phase 1 — cluster elimination")
        while store.count > DIRECT_THRESHOLD:
            turn += 1
            print(f"\n[CRS] Turn {turn} | {store.count} layouts remaining")
            k        = store.compute_dynamic_k()
            clusters = store.cluster(k)
            cluster_summaries: dict[int, str] = {}
            for cid, cl_layouts in clusters.items():
                descs   = [l["description"] for l in cl_layouts]
                cluster_summaries[cid] = llm.summarise_cluster(descs)
            question, option_map = llm.generate_cluster_question(
                cluster_summaries, asked_questions, user_ctx
            )
            asked_questions.append(question)
            _set(turn=turn)
            raw_input = _ask_and_wait(question, "1", store.count, user_ctx)
            if not raw_input:
                asked_questions.pop()
                continue
            user_input = llm.refine_user_answer(question, raw_input)
            is_unsure  = llm.detect_unsure(question, user_input)
            if is_unsure:
                topic = question[:60] + ("..." if len(question) > 60 else "")
                user_ctx.add_unsure(topic)
                print(f"[CRS] Unsure — skipping topic: {topic}")
                continue
            keep_ids        = llm.interpret_cluster_choice(question, cluster_summaries, user_input, option_map)
            keep_layout_ids = {l["id"] for cid in keep_ids for l in clusters[cid]}
            eliminate_ids   = [l["id"] for l in store.active_layouts if l["id"] not in keep_layout_ids]
            kept_summaries    = [cluster_summaries[cid] for cid in keep_ids if cid in cluster_summaries]
            dropped_summaries = [cluster_summaries[cid] for cid in cluster_summaries if cid not in keep_ids]
            llm.update_user_context(question, user_input, kept_summaries, dropped_summaries, user_ctx)
            process_eliminated.extend(eliminate_ids)
            store.eliminate(eliminate_ids)
            _set(process_eliminated=list(process_eliminated))
            if store.count == 0:
                _set(status="error", error="All layouts were eliminated during Phase 1.")
                return

        # Phase 2 — Direct Elimination
        print(f"[CRS] Phase 2 — direct elimination ({store.count} remaining)")
        while store.count > 1:
            turn += 1
            remaining = store.active_layouts
            print(f"[CRS] Turn {turn} | {len(remaining)} layouts remaining")
            question = llm.generate_direct_question(remaining, asked_questions, user_ctx)
            asked_questions.append(question)
            _set(turn=turn)
            raw_input = _ask_and_wait(question, "2", store.count, user_ctx)
            if not raw_input:
                asked_questions.pop()
                continue
            user_input = llm.refine_user_answer(question, raw_input)
            is_unsure  = llm.detect_unsure(question, user_input)
            if is_unsure:
                topic = question[:60] + ("..." if len(question) > 60 else "")
                user_ctx.add_unsure(topic)
                print(f"[CRS] Unsure — skipping topic: {topic}")
                continue
            eliminate_ids = llm.interpret_direct_elim(question, remaining, user_input, user_ctx)
            kept_layouts    = [l for l in remaining if l["id"] not in set(eliminate_ids)]
            dropped_layouts = [l for l in remaining if l["id"] in set(eliminate_ids)]
            llm.update_user_context(
                question, user_input,
                [l["description"][:120] for l in kept_layouts],
                [l["description"][:120] for l in dropped_layouts],
                user_ctx,
            )
            process_eliminated.extend(eliminate_ids)
            store.eliminate(eliminate_ids)
            _set(process_eliminated=list(process_eliminated))

        # Final Result
        winner = store.active_layouts[0] if store.active_layouts else None
        print(f"[CRS] Done! Winner: {winner['id'] if winner else 'none'}")
        _set(
            status="done", question=None,
            remaining_count=1 if winner else 0,
            result=winner,
            user_context=user_ctx.to_dict(),
            process_eliminated=list(process_eliminated),
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set(status="error", error=str(exc))


if __name__ == "__main__":
    uvicorn.run("backend_server:app", host="0.0.0.0", port=8000, reload=False, workers=1)
