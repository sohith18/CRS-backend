#!/usr/bin/env python3
"""
backend_vlm.py — FastAPI backend for Agentic RAG Room Layout Recommendation (VLM + DINOv3).
Run: uvicorn backend_vlm:app --host 0.0.0.0 --port 8000

Env vars:
  LAYOUTS_PATH      path to layouts.json              (default: layouts.json)
  VLM_MODEL_ID      Ollama VLM model tag               (default: qwen3-vl:30b-a3b-instruct)
  DINOV3_MODEL      HuggingFace DINOv3 model name      (default: facebook/dinov3-vith16plus-pretrain-lvd1689m)
  OLLAMA_HOST       Ollama API base URL                 (default: http://localhost:11434)
  DIRECT_THRESHOLD  layouts to surface as final picks   (default: 5)
  MAX_K             max cluster count                   (default: 10)
"""

import os, re, json, base64, threading
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import torch
from transformers import AutoImageProcessor, AutoModel
import ollama


# ── Sentinels ──────────────────────────────────────────────────────────────────
KEEP_ALL_SENTINEL = "__KEEP_ALL__"
UNSURE_SENTINEL   = "__UNSURE__"

# ── Knobs (overridable via env) ────────────────────────────────────────────────
DIRECT_THRESHOLD = int(os.environ.get("DIRECT_THRESHOLD", 5))
MAX_K            = int(os.environ.get("MAX_K",            10))
DINOV3_MODEL     = os.environ.get("DINOV3_MODEL", "facebook/dinov3-vith16plus-pretrain-lvd1689m")
VLM_MODEL_ID     = os.environ.get("VLM_MODEL_ID", "qwen3-vl:30b-a3b-instruct")
OLLAMA_HOST      = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")
LAYOUTS_PATH     = os.environ.get("LAYOUTS_PATH", "vlm_layouts.json")


# ══════════════════════════════════════════════════════════════════════════════
# 1. JSON LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_layouts_from_json(filepath: str) -> list[dict]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Layout JSON not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("JSON root must be a list of layout entries.")

    layouts: list[dict] = []

    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            print(f"  [Loader] ⚠ Skipping entry {idx} — not a dict.")
            continue

        query_id   = str(entry.get("query_id", f"layout_{idx:05d}"))
        image_path = str(entry.get("image_path", "")).strip()

        if not image_path:
            print(f"  [Loader] ⚠ No image_path for {query_id} — skipping.")
            continue

        if not Path(image_path).exists():
            print(f"  [Loader] ⚠ Image not found for {query_id}: '{image_path}' — skipping.")
            continue

        layouts.append({"id": query_id, "image_path": image_path})

    if not layouts:
        raise ValueError("No valid layouts found in the JSON file.")

    print(f"[Loader] Loaded {len(layouts)} layouts")
    return layouts


# ══════════════════════════════════════════════════════════════════════════════
# 2. USER CONTEXT
# ══════════════════════════════════════════════════════════════════════════════
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
            return "No preferences recorded yet."
        lines = ["What we know about the user so far:"]
        if self.likes:
            lines.append("  Definitely wants: " + "; ".join(self.likes))
        if self.dislikes:
            lines.append("  Definitely does NOT want: " + "; ".join(self.dislikes))
        if self.unsure:
            lines.append("  Still undecided about: " + "; ".join(self.unsure))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"likes": self.likes, "dislikes": self.dislikes, "unsure": self.unsure}


# ══════════════════════════════════════════════════════════════════════════════
# 3. DINOV3 IMAGE EMBEDDER  (loaded once at startup)
# ══════════════════════════════════════════════════════════════════════════════
class DINOv3Embedder:

    def __init__(self, model_name: str = DINOV3_MODEL):
        print(f"[DINOv3] Loading model: {model_name}")
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype     = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype)
        self.model.eval().to(self.device)
        print(f"[DINOv3] Ready on {self.device} | dtype={self.dtype}")

    def embed(self, image_paths: list[str], batch_size: int = 8) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch  = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls.float().cpu().numpy())
        return np.vstack(all_embeddings)


# ══════════════════════════════════════════════════════════════════════════════
# 4. VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════
class LayoutVectorStore:

    def __init__(self, embedder: DINOv3Embedder):
        self.embedder        = embedder
        self.active_layouts: list[dict] = []
        self.embeddings:     np.ndarray  = np.array([])

    def build(self, layouts: list[dict]):
        self.active_layouts = list(layouts)
        self._recompute_embeddings()
        print(f"[VectorStore] Built — {len(self.active_layouts)} layouts, "
              f"dim={self.embeddings.shape[1]}")

    def _recompute_embeddings(self):
        paths = [l["image_path"] for l in self.active_layouts]
        raw   = self.embedder.embed(paths)
        self.embeddings = normalize(raw, norm='l2').astype(np.float32)

    def compute_dynamic_k(self) -> int:
        n   = len(self.active_layouts)
        cap = min(n - 1, MAX_K)
        if cap < 2 or n <= 4:
            return 2
        best_k, best_score = 2, -1.0
        for k in range(2, cap + 1):
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.embeddings)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(self.embeddings, labels, metric="cosine")
            print(f"  [K={k}] silhouette={score:.4f}")
            if score > best_score:
                best_score, best_k = score, k
        print(f"  → Best K={best_k} (silhouette={best_score:.4f})")
        return best_k

    def cluster(self, k: int) -> tuple[dict[int, list[dict]], KMeans]:
        k  = min(k, len(self.active_layouts))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(self.embeddings)
        clusters: dict[int, list[dict]] = {}
        for layout, label in zip(self.active_layouts, labels):
            clusters.setdefault(int(label), []).append(layout)
        return clusters, km

    def get_ranked_members(
        self, cluster_layouts: list[dict], cid: int, km: KMeans
    ) -> list[dict]:
        indices  = [i for i, l in enumerate(self.active_layouts)
                    if l["id"] in {cl["id"] for cl in cluster_layouts}]
        centroid = km.cluster_centers_[cid]
        paired   = sorted(
            zip([np.linalg.norm(self.embeddings[i] - centroid) for i in indices],
                cluster_layouts),
            key=lambda x: x[0],
        )
        return [layout for _, layout in paired]

    def eliminate(self, layout_ids: list[str]):
        remove    = set(layout_ids)
        survivors = [(l, e) for l, e in zip(self.active_layouts, self.embeddings)
                     if l["id"] not in remove]
        if not survivors:
            print("[VectorStore] ⚠ Tried to eliminate everything — skipping.")
            return
        self.active_layouts, emb_list = zip(*survivors)
        self.active_layouts = list(self.active_layouts)
        self.embeddings     = np.array(emb_list, dtype=np.float32)
        print(f"[VectorStore] Eliminated {len(remove)} | Remaining: {len(self.active_layouts)}")

    @property
    def count(self) -> int:
        return len(self.active_layouts)


# ══════════════════════════════════════════════════════════════════════════════
# 5. QWEN3-VL AGENT
# ══════════════════════════════════════════════════════════════════════════════
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _is_valid_question(q: str) -> bool:
    """Reject questions that are too long, missing '?', or reference Image 1/2 directly."""
    if not q or len(q) > 160:
        return False
    if re.search(r'\bimage\s*[12]\b', q, re.IGNORECASE):
        return False
    if "?" not in q:
        return False
    return True


class Qwen3VLAgent:

    def __init__(self, model_id: str = VLM_MODEL_ID, host: str = OLLAMA_HOST):
        print(f"[VLM] Connecting to Ollama at {host} — model: {model_id}")
        self.model_id = model_id
        self.host     = host
        print("[VLM] Ready")

    def _generate(self, text: str, img1_path: str, img2_path: str,
                  max_tokens: int = 300, temperature: float = 0.3) -> str:
        img1_b64 = encode_image(img1_path)
        img2_b64 = encode_image(img2_path)

        client   = ollama.Client(host=self.host)
        response = client.chat(
            model=self.model_id,
            messages=[{
                "role":    "user",
                "content": text,
                "images":  [img1_b64, img2_b64],
            }],
            options={
                "temperature":    temperature,
                "num_predict":    max_tokens,
                "top_p":          0.9,
                "repeat_penalty": 1.1,
            },
        )
        raw = response["message"]["content"]
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*",          "", raw, flags=re.DOTALL)
        return raw.strip()

    @staticmethod
    def _first_line(text: str) -> str:
        for line in text.strip().split("\n"):
            line = line.strip()
            if line:
                return line
        return ""

    @staticmethod
    def _history_block(asked_questions: list[str]) -> str:
        return (
            "\n".join(f"- {q}" for q in asked_questions)
            if asked_questions else "None yet."
        )

    # ── IMPROVED: generate_cluster_question ───────────────────────────────────
    def generate_cluster_question(
        self,
        img1_path: str, img2_path: str,
        asked_questions: list[str], user_ctx: UserContext,
    ) -> str:
        history_str = self._history_block(asked_questions)
        ctx_text    = user_ctx.as_prompt_block()

        prompt = (
            "You are analyzing two top-down (bird's-eye view) 2D room layout floor plans.\n"
            "The FIRST image you received is Layout A. The SECOND image is Layout B.\n\n"

            "STEP 1 — Identify the single most visually obvious difference between "
            "Layout A and Layout B. Focus on ONE of:\n"
            "  • Position of a major furniture piece relative to a fixed room feature "
            "(door, window)\n"
            "  • Furniture grouping or cluster arrangement\n"
            "  • Traffic flow or walkway direction through the room\n"
            "  • How the room would FEEL to live in — consider:\n"
            "      - Airy and open vs. cozy and filled\n"
            "      - Balanced/formal vs. relaxed/casual\n"
            "      - A clear focal point vs. spread-out arrangement\n"
            "      - Easy to move through vs. intimate and snug\n"
            "      - Social/inviting vs. private/retreat-like\n\n"

            "STEP 2 — Anchor all spatial language to a named reference object. "
            "NEVER use bare 'left' or 'right' without a reference "
            "(e.g., say 'left of the bed', 'beside the window', 'against the far wall', "
            "'near the door'). "
            "If you cannot name a reference object, describe by wall proximity instead "
            "(e.g., 'pushed against the top wall', 'centered in the room').\n"
            "For feel-based questions, use mood words directly "
            "(e.g., 'airy', 'cozy', 'open', 'snug', 'balanced', 'casual').\n\n"

            "STEP 3 — Write ONE preference question (max 25 words) that:\n"
            "  • Is a direct this-or-that choice (e.g., 'Do you prefer X or Y?')\n"
            "  • Uses either room-relative anchored spatial terms OR feel/mood language\n"
            "  • Is about something a real person would notice and care about\n"
            "  • Is answerable without seeing the images "
            "(e.g., 'I prefer the cozy one' or 'I prefer more open space')\n"
            "  • Does NOT mention 'Layout A', 'Layout B', 'Image 1', or 'Image 2'\n\n"

            f"Previously asked questions — DO NOT ask about the same spatial feature or feeling again:\n"
            f"{history_str}\n\n"
            f"Known user preferences (skip features already resolved):\n{ctx_text}\n\n"

            "Output ONLY the question text. No preamble, no numbering, no explanation."
        )

        return self._first_line(
            self._generate(prompt, img1_path, img2_path, max_tokens=100, temperature=0.3)
        )

    # ── IMPROVED: deduce_preferred_cluster ────────────────────────────────────
    def deduce_preferred_cluster(
        self,
        img1_path: str, img2_path: str,
        question: str, user_input: str, user_ctx: UserContext,
    ) -> str:
        prompt = (
            "You are looking at two top-down room layout floor plans.\n"
            "The FIRST image you received is Layout A.\n"
            "The SECOND image you received is Layout B.\n\n"

            f"The user was asked: \"{question}\"\n"
            f"The user replied: \"{user_input}\"\n\n"

            "Follow these steps:\n"
            "STEP 1 — Identify the specific spatial feature that DIFFERS between "
            "Layout A and Layout B as it relates to the question.\n"
            "STEP 2 — Determine which layout's spatial arrangement matches what "
            "the user described in their reply.\n\n"

            "Decision rules:\n"
            "• User reply clearly matches ONLY Layout A → output: A\n"
            "• User reply clearly matches ONLY Layout B → output: B\n"
            "• User reply is vague, too short, contradictory, or matches both equally → output: both\n"
            "• User reply addresses a feature that is THE SAME in both layouts → output: both\n"
            "• User reply is off-topic or unrelated to the question → output: both\n\n"

            f"Known user preferences for additional context:\n{user_ctx.as_prompt_block()}\n\n"

            "Output ONLY one token: A, B, or both. Absolutely nothing else."
        )

        raw = self._first_line(
            self._generate(prompt, img1_path, img2_path, max_tokens=15, temperature=0.0)
        ).strip().upper()

        if re.search(r'\bA\b', raw) and not re.search(r'\bB\b', raw):
            return "1"
        elif re.search(r'\bB\b', raw) and not re.search(r'\bA\b', raw):
            return "2"
        else:
            return "both"

    # ── IMPROVED: detect_unsure ────────────────────────────────────────────────
    def detect_unsure(self, img1_path: str, img2_path: str,
                      question: str, user_input: str) -> bool:
        prompt = (
            f"A user was asked this preference question about two room layouts:\n"
            f"Question: \"{question}\"\n"
            f"User reply: \"{user_input}\"\n\n"
            "Does the user's reply show a CLEAR spatial preference, "
            "or are they uncertain / non-committal?\n"
            "Signs of UNSURE: 'I don't know', 'maybe', 'either', 'both are fine', "
            "very short replies (<3 words), or replies that don't address the question.\n"
            "Signs of CLEAR: names a specific feature, direction, or furniture arrangement.\n\n"
            "Output exactly one word: UNSURE or CLEAR."
        )
        raw = self._first_line(
            self._generate(prompt, img1_path, img2_path, max_tokens=10, temperature=0.0)
        ).strip().upper()
        return "UNSURE" in raw

    # ── IMPROVED: update_user_context ─────────────────────────────────────────
    def update_user_context(
        self,
        img1_path: str, img2_path: str,
        question: str, user_input: str,
        kept_img: str, dropped_img: str,
        user_ctx: UserContext,
    ):
        prompt = (
            "Two top-down room layouts were compared and the user expressed a preference.\n\n"
            f"Question asked: \"{question}\"\n"
            f"User replied: \"{user_input}\"\n"
            f"Preferred layout: {kept_img}\n"
            f"Rejected layout: {dropped_img}\n\n"
            "Extract exactly one spatial LIKE and one spatial DISLIKE from this exchange.\n"
            "Use room-relative, anchored language "
            "(e.g., 'bed against the far wall', 'open space near entrance', "
            "'sofa facing the window', 'desk beside the door').\n"
            "Be specific — avoid generic phrases like 'good layout' or 'nice arrangement'.\n\n"
            "Output EXACTLY two lines in this format:\n"
            "LIKE: <phrase, max 10 words>\n"
            "DISLIKE: <phrase, max 10 words>\n"
            "Use the word 'nothing' if nothing specific is certain for a category."
        )
        raw = self._generate(prompt, img1_path, img2_path, max_tokens=80, temperature=0.1)
        like_fact = dislike_fact = ""
        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("LIKE:"):
                like_fact    = line.split(":", 1)[1].strip()
            elif line.upper().startswith("DISLIKE:"):
                dislike_fact = line.split(":", 1)[1].strip()
        if like_fact    and like_fact.lower()    != "nothing": user_ctx.add_like(like_fact)
        if dislike_fact and dislike_fact.lower() != "nothing": user_ctx.add_dislike(dislike_fact)


# ══════════════════════════════════════════════════════════════════════════════
# 6. PAIR SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
class PairSelector:
    def __init__(self):
        self.pair_index  = 0
        self.anchor_step = 0

    def reset(self):
        self.pair_index  = 0
        self.anchor_step = 0

    def get_pair(
        self, left_pool: list[dict], right_pool: list[dict]
    ) -> tuple[dict, dict]:
        r_idx = self.pair_index  % len(right_pool)
        l_idx = self.anchor_step % len(left_pool)
        return left_pool[l_idx], right_pool[r_idx]

    def advance_on_ambiguity(
        self, left_pool: list[dict], right_pool: list[dict]
    ) -> str:
        self.anchor_step += 1
        if self.anchor_step >= len(left_pool):
            self.anchor_step = 0
            self.pair_index += 1
            return "right_advanced"
        return "left_rotated"

    def on_clear_answer(self):
        self.anchor_step = 0


# ══════════════════════════════════════════════════════════════════════════════
# 7. FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="Room Layout Recommendation API (VLM+DINOv3)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared session state ───────────────────────────────────────────────────────
_session: dict = {
    "status":             "idle",
    "question":           None,
    "phase":              None,
    "turn":               0,
    "remaining_count":    0,
    "user_context":       {},
    "comparison_images":  None,
    "result":             None,
    "process_eliminated": [],
    "error":              None,
    "_user_answer":       None,
}
_lock         = threading.Lock()
_answer_event = threading.Event()
_worker: Optional[threading.Thread] = None
_embedder: Optional[DINOv3Embedder] = None


@app.on_event("startup")
def _startup():
    global _embedder
    print("[Startup] Pre-loading DINOv3 embedder ...")
    _embedder = DINOv3Embedder(model_name=DINOV3_MODEL)
    print("[Startup] DINOv3 ready.\n")


def _set(**kwargs):
    with _lock:
        _session.update(kwargs)


def _snap() -> dict:
    with _lock:
        return {k: v for k, v in _session.items() if not k.startswith("_")}


# ══════════════════════════════════════════════════════════════════════════════
# 8. PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════
class StartRequest(BaseModel):
    pass


class AnswerRequest(BaseModel):
    answer: str


# ══════════════════════════════════════════════════════════════════════════════
# 9. ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    snap = _snap()
    return {
        "server":         "ok",
        "session_status": snap["status"],
        "turn":           snap["turn"],
        "remaining":      snap["remaining_count"],
    }


@app.get("/question")
def get_question():
    return _snap()


@app.post("/answer")
def post_answer(req: AnswerRequest):
    with _lock:
        if _session["status"] != "waiting":
            raise HTTPException(
                status_code=400,
                detail=f"Not waiting for an answer (status='{_session['status']}').",
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
            "comparison_images": None,
            "result": None, "process_eliminated": [], "error": None,
            "_user_answer": None,
        })
        _answer_event.clear()
    _worker = threading.Thread(target=_crs_worker, daemon=True)
    _worker.start()
    return {"ok": True, "message": "Session started — poll GET /question."}


@app.post("/reset")
def reset_session():
    with _lock:
        _session.update({
            "status": "idle", "question": None, "phase": None,
            "turn": 0, "remaining_count": 0, "user_context": {},
            "comparison_images": None,
            "result": None, "process_eliminated": [], "error": None,
            "_user_answer": None,
        })
        _answer_event.set()
    return {"ok": True, "message": "Session reset to idle."}


@app.get("/image/{image_path:path}")
def serve_image(image_path: str):
    """Serve layout image files to the frontend."""
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
    return FileResponse(str(p))


# ══════════════════════════════════════════════════════════════════════════════
# 10. BLOCKING HELPER
# ══════════════════════════════════════════════════════════════════════════════
def _ask_and_wait(
    question:         str,
    phase:            str,
    remaining_count:  int,
    user_ctx:         UserContext,
    img_left_path:    str,
    img_right_path:   str,
) -> str:
    comparison_images = {
        "left":  f"/image/{img_left_path}",
        "right": f"/image/{img_right_path}",
    }
    _set(
        status            = "waiting",
        question          = question,
        phase             = phase,
        remaining_count   = remaining_count,
        user_context      = user_ctx.to_dict(),
        comparison_images = comparison_images,
    )
    _answer_event.wait()
    _answer_event.clear()
    with _lock:
        if _session["status"] == "idle":
            raise RuntimeError("Session was reset — aborting worker.")
        answer = _session["_user_answer"]
    return answer or ""


# ══════════════════════════════════════════════════════════════════════════════
# 11. CRS WORKER
# ══════════════════════════════════════════════════════════════════════════════
def _crs_worker():
    try:
        print(f"\n[CRS] Loading layouts from: {LAYOUTS_PATH}")
        all_layouts = load_layouts_from_json(LAYOUTS_PATH)

        if len(all_layouts) < 2:
            _set(status="error", error="Need at least 2 valid layouts.")
            return

        vlm      = Qwen3VLAgent(model_id=VLM_MODEL_ID, host=OLLAMA_HOST)
        store    = LayoutVectorStore(embedder=_embedder)
        user_ctx = UserContext()
        selector = PairSelector()

        store.build(all_layouts)

        asked_questions:    list[str] = []
        process_eliminated: list[str] = []
        turn = 0

        print("[CRS] Phase 1 — cluster elimination")
        _set(status="processing", phase="1", remaining_count=store.count)

        while store.count > DIRECT_THRESHOLD:
            print(f"\n[CRS] Turn {turn} | {store.count} layouts remaining")

            k = store.compute_dynamic_k()
            clusters, km = store.cluster(k)

            print(f"[CRS] K={k} | Clusters: { {cid: len(cl) for cid, cl in clusters.items()} }")

            cids         = list(clusters.keys())
            cid_a, cid_b = cids[0], cids[1]

            left_pool  = store.get_ranked_members(clusters[cid_a], cid_a, km)
            right_pool = store.get_ranked_members(clusters[cid_b], cid_b, km)
            selector.reset()

            cluster_eliminated = False
            while not cluster_eliminated:
                turn += 1
                _set(turn=turn)

                rep_a, rep_b = selector.get_pair(left_pool, right_pool)
                img_a = rep_a["image_path"]
                img_b = rep_b["image_path"]

                print(f"[CRS] Comparing '{rep_a['id']}' vs '{rep_b['id']}'")

                # ── Generate + validate question (retry up to 3 times) ──────
                question = ""
                for attempt in range(3):
                    candidate = vlm.generate_cluster_question(
                        img_a, img_b, asked_questions, user_ctx
                    )
                    if _is_valid_question(candidate):
                        question = candidate
                        break
                    print(f"[CRS] ⚠ Invalid question (attempt {attempt+1}): '{candidate}'")

                if not question:
                    print("[CRS] ⚠ Could not generate a valid question — rotating pair.")
                    selector.advance_on_ambiguity(left_pool, right_pool)
                    continue

                asked_questions.append(question)

                raw_input = _ask_and_wait(
                    question, "1", store.count, user_ctx, img_a, img_b
                )

                if not raw_input:
                    asked_questions.pop()
                    continue

                is_unsure = vlm.detect_unsure(img_a, img_b, question, raw_input)
                if is_unsure:
                    user_ctx.add_unsure(question[:60] + ("..." if len(question) > 60 else ""))
                    print(f"[CRS] Unsure – rotating pair")
                    move = selector.advance_on_ambiguity(left_pool, right_pool)
                    next_a, next_b = selector.get_pair(left_pool, right_pool)
                    print(f"[CRS] ({move}): next → {next_a['id']} vs {next_b['id']}")
                    _set(user_context=user_ctx.to_dict())
                    continue

                preferred = vlm.deduce_preferred_cluster(
                    img_a, img_b, question, raw_input, user_ctx
                )
                print(f"[CRS] VLM preferred: Image {preferred}")

                if preferred == "both":
                    move = selector.advance_on_ambiguity(left_pool, right_pool)
                    next_a, next_b = selector.get_pair(left_pool, right_pool)
                    print(f"[CRS] Ambiguous ({move}): next → {next_a['id']} vs {next_b['id']}")
                    continue

                selector.on_clear_answer()

                eliminate_cid = cid_b if preferred == "1" else cid_a
                eliminate_ids = [l["id"] for l in clusters[eliminate_cid]]
                print(f"[CRS] Eliminating cluster {eliminate_cid} ({len(eliminate_ids)} layouts)")

                vlm.update_user_context(
                    img_a, img_b, question, raw_input,
                    kept_img    = f"Layout {'A' if preferred == '1' else 'B'}",
                    dropped_img = f"Layout {'B' if preferred == '1' else 'A'}",
                    user_ctx    = user_ctx,
                )
                process_eliminated.extend(eliminate_ids)
                store.eliminate(eliminate_ids)
                cluster_eliminated = True

                _set(
                    process_eliminated = list(process_eliminated),
                    remaining_count    = store.count,
                    user_context       = user_ctx.to_dict(),
                )

            if store.count == 0:
                _set(status="error", error="All layouts were eliminated during Phase 1.")
                return

        # ── Final recommendation ───────────────────────────────────────────
        surviving = store.active_layouts
        print(f"\n[CRS] Done! Recommending {len(surviving)} layouts.")

        result_payload = [
            {"id": l["id"], "image_path": l["image_path"],
             "image_url": f"/image/{l['image_path']}"}
            for l in surviving
        ]

        _set(
            status             = "done",
            question           = None,
            phase              = None,
            comparison_images  = None,
            remaining_count    = len(surviving),
            result             = result_payload,
            user_context       = user_ctx.to_dict(),
            process_eliminated = list(process_eliminated),
        )

    except RuntimeError as exc:
        print(f"[CRS] Worker interrupted: {exc}")

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set(status="error", error=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# 12. ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run("backend_vlm:app", host="0.0.0.0", port=8000, reload=False, workers=1)