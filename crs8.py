"""
Agentic RAG — Room Layout Recommendation
=========================================
Phase 0 (Furniture Filter):
  • User provides a list of required furniture items
  • Only layouts containing ALL specified furniture are kept
  • Filtered-out layouts are recorded and shown at the end

Phase 1 (K-means Elimination):
  • Embed all active layout descriptions with BGE-M3
  • Run K-means with dynamic K (elbow/silhouette, capped ≤ 10)
  • LLM summarises each cluster → generates a conversational question
  • User context (likes/dislikes/unsure) is maintained and passed to every question
  • "Unsure" answers → cluster kept as-is, fresh question on a different topic
  • Previous questions passed so LLM never repeats topics
  • User answers naturally → LLM interprets which cluster(s) to KEEP
  • Rebuild embeddings on survivors, repeat until ≤ DIRECT_THRESHOLD remain

Phase 2 (Direct Elimination):
  • Too few layouts for clustering to be useful
  • LLM reads all remaining descriptions + user context → asks a pinpoint question
  • "Unsure" answers → no elimination, fresh question
  • Repeat until exactly 1 layout survives

Usage:
  python crs8.py --layouts path/to/layouts.json
  python crs8.py --layouts layouts.json --furniture "bed,desk,wardrobe"
  python crs8.py --layouts layouts.json --direct-threshold 4 --strict-validation
"""

import os, re, json, random, argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"


# ── Sentinels ──────────────────────────────────────────────────────────────────
KEEP_ALL_SENTINEL = "__KEEP_ALL__"
UNSURE_SENTINEL   = "__UNSURE__"


# ── Default Knobs (overridable via CLI) ────────────────────────────────────────
DIRECT_THRESHOLD  = 5
MAX_STALE_TURNS   = 2     # ✅ NEW: force Phase 2 after this many consecutive no-eliminations
N_SAMPLE          = 10
MAX_K             = 10
EMBEDDER_MODEL    = "BAAI/bge-m3"
LLM_MODEL_ID      = "Qwen/Qwen3.5-9B"


# ══════════════════════════════════════════════════════════════════════════════
# 1. JSON LOADER
# ══════════════════════════════════════════════════════════════════════════════

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
            print(f"  [Loader] ⚠ Skipping entry {idx} — not a dict.")
            continue

        query_id    = str(entry.get("query_id", f"layout_{idx:05d}"))
        description = str(entry.get("prompt",   "")).strip()
        room_size   = entry.get("room_size", [1, 1])
        obj_list    = entry.get("object_list", [])

        if not description:
            description = f"Room layout {query_id}."

        W = float(room_size[0]) if room_size[0] > 0 else 1.0
        H = float(room_size[1]) if room_size[1] > 0 else 1.0

        layout: dict = {"id": query_id, "description": description}

        for item in obj_list:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                print(f"  [Loader] ⚠ Malformed object in {query_id}: {item}")
                continue

            name   = str(item[0]).lower().strip()
            coords = item[1]

            if not (isinstance(coords, (list, tuple)) and len(coords) == 4):
                print(f"  [Loader] ⚠ Bad coords for '{name}' in {query_id}: {coords}")
                continue

            x1, y1, x2, y2 = [float(c) for c in coords]

            if max(x1, y1, x2, y2) > 1.0:
                x1, y1, x2, y2 = x1 / W, y1 / H, x2 / W, y2 / H

            x1, y1, x2, y2 = (
                max(0.0, min(1.0, x1)),
                max(0.0, min(1.0, y1)),
                max(0.0, min(1.0, x2)),
                max(0.0, min(1.0, y2)),
            )

            layout[name] = [x1, y1, x2, y2]
            all_furniture.add(name)

        layouts.append(layout)

    if not layouts:
        raise ValueError("No valid layouts found in the JSON file.")

    furniture_items = sorted(all_furniture)
    print(f"[Loader] Loaded {len(layouts)} layouts")
    print(f"[Loader] Furniture keys found: {furniture_items}")
    return layouts, furniture_items


# ══════════════════════════════════════════════════════════════════════════════
# 1b. FURNITURE FILTER
# ══════════════════════════════════════════════════════════════════════════════

def get_required_furniture(cli_value: str | None, all_furniture: list[str]) -> list[str]:
    if cli_value:
        raw_items = cli_value
    else:
        print("\n" + "=" * 55)
        print(" PHASE 0  —  Furniture Filter")
        print("=" * 55)
        print(f"\nAvailable furniture items in this dataset:")
        print("  " + ", ".join(all_furniture))
        print("\nEnter the furniture items you MUST have in your room layout.")
        print("Type them as a comma-separated list (leave blank to use ALL layouts):")
        raw_items = input("  Furniture: ").strip()

    if not raw_items:
        print("[Filter] No furniture filter applied — using all layouts.\n")
        return []

    requested = [f.strip().lower() for f in raw_items.split(",") if f.strip()]
    unknown   = [f for f in requested if f not in all_furniture]
    valid     = [f for f in requested if f in all_furniture]

    if unknown:
        print(f"[Filter] ⚠ Unknown furniture items (will be ignored): {unknown}")
    if not valid:
        print("[Filter] ⚠ No valid furniture items specified — using all layouts.\n")
        return []

    print(f"[Filter] Required furniture: {valid}\n")
    return valid


def filter_layouts_by_furniture(
    layouts:            list[dict],
    required_furniture: list[str],
) -> tuple[list[dict], list[dict]]:
    if not required_furniture:
        return list(layouts), []

    matching: list[dict] = []
    excluded: list[dict] = []

    for layout in layouts:
        missing = [f for f in required_furniture if f not in layout]
        if missing:
            excluded.append(layout)
        else:
            matching.append(layout)

    print(f"[Filter] {len(matching)} layouts contain all required furniture "
          f"({required_furniture}).")
    if excluded:
        print(f"[Filter] {len(excluded)} layouts excluded (missing at least one item):")
        for l in excluded:
            missing_in_l = [f for f in required_furniture if f not in l]
            print(f"  • {l['id']} — missing: {missing_in_l}")
    print()
    return matching, excluded


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
            return ""
        lines = ["\nWhat we know about the user so far:"]
        if self.likes:
            lines.append("  Definitely wants: " + "; ".join(self.likes))
        if self.dislikes:
            lines.append("  Definitely does NOT want: " + "; ".join(self.dislikes))
        if self.unsure:
            lines.append("  Still undecided about: " + "; ".join(self.unsure))
        return "\n".join(lines) + "\n"

    def __repr__(self):
        return (f"UserContext(likes={self.likes}, "
                f"dislikes={self.dislikes}, unsure={self.unsure})")


# ══════════════════════════════════════════════════════════════════════════════
# 3. VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════
class LayoutVectorStore:

    def __init__(self, model_name: str = EMBEDDER_MODEL):
        print(f"[VectorStore] Loading embedder: {model_name}")
        self.embedder        = SentenceTransformer(model_name)
        self.active_layouts: list[dict] = []
        self.embeddings:     np.ndarray  = np.array([])

    def build(self, layouts: list[dict]):
        self.active_layouts = list(layouts)
        self._recompute_embeddings()
        print(f"[VectorStore] Built. {len(self.active_layouts)} layouts, "
              f"dim={self.embeddings.shape[1]}\n")

    def _recompute_embeddings(self):
        descriptions = [l["description"] for l in self.active_layouts]
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
            print(f"  [K={k}] silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_k     = k
        print(f"  → Best K={best_k} (silhouette={best_score:.4f})")
        return best_k

    def cluster(self, k: int) -> dict[int, list[dict]]:
        k  = min(k, len(self.active_layouts))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
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
            print("[VectorStore] ⚠ Tried to eliminate everything — skipping.")
            return
        self.active_layouts, emb_list = zip(*survivors)
        self.active_layouts = list(self.active_layouts)
        self.embeddings     = np.array(emb_list, dtype=np.float32)
        print(f"[VectorStore] Eliminated {len(remove)} layouts. "
              f"Remaining: {len(self.active_layouts)}")

    @property
    def count(self) -> int:
        return len(self.active_layouts)


# ══════════════════════════════════════════════════════════════════════════════
# 4. LLM AGENT
# ══════════════════════════════════════════════════════════════════════════════
class QwenAgent:

    def __init__(self, model_id: str = LLM_MODEL_ID):
        print(f"[LLM] Loading {model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model     = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map        = "auto",
            dtype             = torch.bfloat16,
            trust_remote_code = True,
        )
        self.model.eval()
        print(f"[LLM] Ready on {next(self.model.parameters()).device}\n")

    def _generate(self, system: str, user: str,
                  max_new_tokens: int = 150,
                  temperature: float  = 0.3) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user",   "content": user}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens     = max_new_tokens,
                do_sample          = temperature > 0,
                temperature        = temperature or None,
                top_p              = 0.9,
                top_k              = 30,
                repetition_penalty = 1.1,
                pad_token_id       = self.tokenizer.eos_token_id,
            )
        generated = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        raw = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    @staticmethod
    def _first_line(text: str) -> str:
        return text.strip().split("\n")[0].strip()

    @staticmethod
    def _history_block(asked_questions: list[str]) -> str:
        if not asked_questions:
            return ""
        lines = "\n".join(f"  - {q}" for q in asked_questions)
        return f"\nQuestions already asked (DO NOT repeat these topics):\n{lines}\n"

    def refine_user_answer(self, question: str, raw_answer: str) -> str:
        system = (
            "You are a helpful assistant that clarifies user responses in a "
            "room-layout recommendation conversation.\n"
            "Given the chatbot's question and the user's raw answer, rewrite "
            "the answer so it is:\n"
            "  • Clear and unambiguous\n"
            "  • One complete English sentence (≤ 30 words)\n"
            "  • Faithful to the user's original intent — do NOT add opinions "
            "    they did not express\n"
            "  • If the answer is already clear and complete, return it unchanged\n"
            "Output ONLY the refined answer. No explanations, no labels."
        )
        user = (
            f"Question: {question}\n"
            f"Raw answer: \"{raw_answer}\"\n"
            "Refined answer:"
        )
        refined = self._first_line(
            self._generate(system, user, max_new_tokens=60, temperature=0.1)
        )
        return refined if refined else raw_answer

    # ✅ FIX Bug 1 — summarise ALL furniture positions, not just bed/openness
    def summarise_cluster(self, descriptions: list[str]) -> str:
        sample = random.sample(descriptions, min(N_SAMPLE, len(descriptions)))
        system = (
            "You summarise room layout styles.\n"
            "Write ONE sentence (max 40 words) capturing ALL key spatial details: "
            "where every major piece of furniture is (bed, wardrobe, desk, sofa, etc.), "
            "their positions relative to each other, and how open or compact the room feels.\n"
            "Plain English only. No jargon. Output the sentence only."
        )
        user = (
            "Layouts:\n" + "\n---\n".join(sample) +
            "\n\nSummarise the shared spatial arrangement in one sentence:"
        )
        return self._first_line(
            self._generate(system, user, max_new_tokens=80, temperature=0.2)
        )

    # ✅ FIX Bug 4 — constrain question to features present in summaries
    def generate_cluster_question(
        self,
        cluster_summaries: dict[int, str],
        asked_questions:   list[str],
        user_ctx:          UserContext,
    ) -> tuple[str, dict[int, int]]:
        option_map   = {i + 1: cid for i, cid in enumerate(cluster_summaries)}
        styles_block = "\n".join(
            f"Style {i+1}: {cluster_summaries[cid]}"
            for i, cid in enumerate(cluster_summaries)
        )
        history   = self._history_block(asked_questions)
        ctx_block = user_ctx.as_prompt_block()

        system = (
            "You help someone find their ideal room layout through friendly conversation.\n"
            "Given summaries of different room style groups, ask ONE natural question "
            "(max 20 words) that reveals which group fits the user best.\n"
            "Rules:\n"
            "• Do NOT list numbered options or say '1)', '2)', 'Style 1', etc.\n"
            "• Ask about a single concrete preference (e.g. bed placement, wardrobe "
            "  position, open space, workspace location, storage arrangement).\n"
            "• Ask ONLY about spatial features that are explicitly mentioned in the "
            "  style summaries below — do NOT introduce furniture items or topics "
            "  that do not appear in those summaries.\n"
            "• The question MUST be answerable by referring to a feature that clearly "
            "  differs between at least two styles — do not ask about features shared "
            "  by ALL styles.\n"
            "• End with '?'.\n"
            "• Plain, friendly English.\n"
            "• Use the user context to avoid asking about things they already decided.\n"
            + ("• Do NOT ask about topics already covered.\n" if asked_questions else "") +
            "Output the question only."
        )
        user = (
            f"Room style groups:\n{styles_block}\n"
            f"{ctx_block}"
            f"{history}"
            "\nWrite one natural question:"
        )
        question = self._first_line(
            self._generate(system, user, max_new_tokens=60, temperature=0.4)
        )
        return question, option_map

    def detect_unsure(self, question: str, user_input: str) -> bool:
        system = (
            "You detect whether a user is uncertain or cannot decide.\n"
            "Given a question and the user's response, output exactly one word:\n"
            "  UNSURE  — if the user is uncertain, says 'I don't know', 'not sure', "
            "'maybe', 'either', 'both', 'no preference', 'doesn't matter', etc.\n"
            "  CLEAR   — if the user expresses any real preference.\n"
            "Output only UNSURE or CLEAR."
        )
        user = (
            f"Question: {question}\n"
            f"User said: \"{user_input}\"\n"
            "Decision:"
        )
        raw = self._first_line(
            self._generate(system, user, max_new_tokens=5, temperature=0.0)
        ).upper()
        return "UNSURE" in raw

    # ✅ FIX Bug 3 + NEW two-pass fallback to break __KEEP_ALL__ loops
    def interpret_cluster_choice(
        self,
        question:          str,
        cluster_summaries: dict[int, str],
        user_input:        str,
        option_map:        dict[int, int],
    ) -> list[int]:
        styles_block = "\n".join(
            f"Style {num}: {cluster_summaries[cid]}"
            for num, cid in option_map.items()
        )

        # ── Pass 1: eliminate-framed prompt ───────────────────────────────
        system = (
            "You interpret a user's room layout preference.\n"
            "Given style summaries and the user's free-text answer, "
            "output ONLY a comma-separated list of style NUMBERS whose description "
            "CONTRADICTS or does NOT match what the user wants — these will be eliminated.\n"
            "Examples: '2'  or  '1,3'\n"
            "Important rules:\n"
            f"• Only output '{KEEP_ALL_SENTINEL}' when the user's answer is completely "
            f"  unrelated to any spatial difference between the styles listed.\n"
            f"• If the user expressed ANY clear preference — even partial — you MUST "
            f"  eliminate at least one style number. Do NOT default to "
            f"  '{KEEP_ALL_SENTINEL}' when a preference is discernible.\n"
            "• Never eliminate ALL styles — always keep at least one.\n"
            "No other text."
        )
        user_prompt = (
            f"Question asked: {question}\n"
            f"Style summaries:\n{styles_block}\n"
            f"User said: \"{user_input}\"\n"
            "Style numbers to ELIMINATE:"
        )
        raw = self._first_line(
            self._generate(system, user_prompt, max_new_tokens=50, temperature=0.1)
        )
        print(f"[LLM] Raw interpretation of user answer: '{raw}'")

        if KEEP_ALL_SENTINEL not in raw and raw.strip():
            eliminate_ids: set[int] = set()
            for token in raw.replace(",", " ").split():
                if token.isdigit():
                    num = int(token)
                    if num in option_map:
                        eliminate_ids.add(option_map[num])
            kept = [cid for cid in cluster_summaries.keys() if cid not in eliminate_ids]
            # Only accept if we actually eliminated something
            if kept and len(kept) < len(cluster_summaries):
                return kept

        # ── Pass 2: keep-framed forced best-match fallback ────────────────
        print("[LLM] Pass 1 ambiguous — running forced best-match (Pass 2).")
        forced_system = (
            "You must decide which room layout styles BEST match what the user wants.\n"
            "Given style summaries and the user's answer, output ONLY the style NUMBER(S) "
            "that BEST match the user's stated preference — these will be KEPT, all others "
            "will be eliminated.\n"
            "Rules:\n"
            "• You MUST output at least one number.\n"
            "• You CANNOT output all style numbers — at least one must be eliminated.\n"
            "• If one style clearly matches better, output just that number.\n"
            "• If two styles tie, output both separated by a comma.\n"
            "Output ONLY numbers separated by commas. No other text."
        )
        forced_user = (
            f"Question: {question}\n"
            f"Style summaries:\n{styles_block}\n"
            f"User said: \"{user_input}\"\n"
            "Style number(s) to KEEP (best match):"
        )
        forced_raw = self._first_line(
            self._generate(forced_system, forced_user, max_new_tokens=30, temperature=0.0)
        )
        print(f"[LLM] Forced best-match result: '{forced_raw}'")

        keep_nums: set[int] = set()
        for token in forced_raw.replace(",", " ").split():
            if token.isdigit():
                num = int(token)
                if num in option_map:
                    keep_nums.add(option_map[num])

        if keep_nums and len(keep_nums) < len(cluster_summaries):
            return list(keep_nums)

        # Final safety: keep all (should rarely reach here)
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
        system = (
            "You extract concise preference facts from a room layout conversation turn.\n"
            "Given the question, the user's answer, what was kept, and what was dropped, "
            "output EXACTLY two lines:\n"
            "LIKE: <one short phrase describing what the user definitely wants, max 10 words>\n"
            "DISLIKE: <one short phrase describing what the user definitely does not want, max 10 words>\n"
            "If nothing is certain for either, write 'nothing' for that line.\n"
            "No other text."
        )
        user = (
            f"Question: {question}\n"
            f"User said: \"{user_input}\"\n"
            f"Layouts/styles KEPT: {kept_txt}\n"
            f"Layouts/styles DROPPED: {dropped_txt}\n"
            "Extract facts:"
        )
        raw = self._generate(system, user, max_new_tokens=60, temperature=0.1)
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
        desc_block = "\n\n".join(f"[{l['id']}]: {l['description']}" for l in layouts)
        history    = self._history_block(asked_questions)
        ctx_block  = user_ctx.as_prompt_block()
        system = (
            "You help narrow down room layout choices through friendly conversation.\n"
            "Given a small set of layout descriptions, ask ONE natural question "
            "(max 20 words) that reveals the user's preference and eliminates as many "
            "non-matching layouts as possible.\n"
            "Rules:\n"
            "• Do NOT list numbered options or layout IDs.\n"
            "• Focus on the biggest concrete difference between the remaining layouts.\n"
            "• Use the user context to avoid redundant questions.\n"
            "• End with '?'. Plain, friendly English.\n"
            + ("• Do NOT ask about topics already covered.\n" if asked_questions else "") +
            "Output the question only."
        )
        user = (
            f"Remaining layouts:\n{desc_block}\n"
            f"{ctx_block}"
            f"{history}"
            "\nWrite one natural question:"
        )
        return self._first_line(
            self._generate(system, user, max_new_tokens=60, temperature=0.4)
        )

    def interpret_direct_elim(
        self,
        question:   str,
        layouts:    list[dict],
        user_input: str,
        user_ctx:   UserContext,
    ) -> list[str]:
        valid_ids  = {l["id"] for l in layouts}
        desc_block = "\n\n".join(f"{l['id']}: {l['description']}" for l in layouts)
        ctx_block  = user_ctx.as_prompt_block()
        system = (
            "You decide which room layouts to eliminate based on the user's answer.\n"
            "Output ONLY a comma-separated list of layout IDs to ELIMINATE "
            "(those that do NOT match what the user wants).\n"
            "Format: layout_01,layout_03\n"
            "Always keep at least one layout.\n"
            "If the answer is unclear, output: none\n"
            "No other text."
        )
        user = (
            f"Question asked: {question}\n"
            f"User said: \"{user_input}\"\n"
            f"{ctx_block}"
            f"\nLayouts:\n{desc_block}\n\n"
            "Layout IDs to eliminate:"
        )
        raw = self._first_line(
            self._generate(system, user, max_new_tokens=50, temperature=0.1)
        ).lower()
        if raw == "none" or not raw:
            return []
        eliminate = [tok.strip() for tok in raw.split(",") if tok.strip() in valid_ids]
        if len(eliminate) >= len(layouts):
            eliminate = eliminate[: len(layouts) - 1]
        return eliminate


# ══════════════════════════════════════════════════════════════════════════════
# 5. VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def validate_layouts(layouts: list[dict], furniture: list[str], strict: bool = False):
    warnings = []
    for l in layouts:
        missing = [f for f in furniture if f not in l]
        if missing:
            warnings.append(f"  {l['id']} missing keys: {missing}")

    if warnings:
        msg = "Layout validation issues:\n" + "\n".join(warnings)
        if strict:
            raise ValueError(msg)
        else:
            print(f"[Validation] ⚠ {msg}")
            print("[Validation] Continuing in non-strict mode.\n")
    else:
        print(f"[Validation] All {len(layouts)} layouts have required keys ✅\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. UNUSED LAYOUTS REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_unused_layouts_report(
    all_layouts:          list[dict],
    furniture_excluded:   list[dict],
    process_eliminated:   list[str],
    winner_id:            str | None,
    required_furniture:   list[str],
):
    print("\n" + "=" * 55)
    print(" LAYOUTS NOT USED IN FINAL RECOMMENDATION")
    print("=" * 55)

    if furniture_excluded:
        print(f"\n📦  A) Excluded by furniture filter "
              f"(missing required items from {required_furniture}):")
        for l in furniture_excluded:
            missing = [f for f in required_furniture if f not in l]
            print(f"   • {l['id']}  — missing: {missing}")
            print(f"     {l['description'][:100]}...")
    else:
        if required_furniture:
            print("\n📦  A) Furniture filter: all layouts passed (none excluded).")
        else:
            print("\n📦  A) No furniture filter was applied.")

    if process_eliminated:
        elim_set = set(process_eliminated)
        print(f"\n🔴  B) Eliminated during recommendation phases "
              f"({len(process_eliminated)} layouts):")
        for l in all_layouts:
            if l["id"] in elim_set:
                print(f"   • {l['id']}")
                print(f"     {l['description'][:100]}...")
    else:
        print("\n🔴  B) Process-eliminated: none.")

    if winner_id:
        print(f"\n🏆  C) Winner (recommended): {winner_id}")

    total_unused = len(furniture_excluded) + len(process_eliminated)
    print(f"\n[Summary] {total_unused} / {len(all_layouts)} layouts were NOT recommended.")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLI + ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Agentic RAG — Room Layout Recommendation CRS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--layouts", "-l", type=str, required=True, metavar="PATH",
                   help="Path to the JSON file containing room layouts.")
    p.add_argument("--furniture", "-f", type=str, default=None, metavar="ITEMS",
                   help="Comma-separated required furniture (e.g. 'bed,desk,wardrobe').")
    p.add_argument("--direct-threshold", type=int, default=DIRECT_THRESHOLD, metavar="N",
                   help="Switch to Phase 2 when ≤ N layouts remain.")
    p.add_argument("--max-k", type=int, default=MAX_K, metavar="K",
                   help="Hard cap on number of K-means clusters.")
    p.add_argument("--max-stale-turns", type=int, default=MAX_STALE_TURNS, metavar="S",
                   help="Force Phase 2 after this many consecutive no-elimination turns.")
    p.add_argument("--embedder", type=str, default=EMBEDDER_MODEL, metavar="MODEL",
                   help="HuggingFace model name for sentence embeddings.")
    p.add_argument("--llm", type=str, default=LLM_MODEL_ID, metavar="MODEL",
                   help="HuggingFace model name for the LLM agent.")
    p.add_argument("--strict-validation", action="store_true",
                   help="Raise an error if any layout is missing furniture keys.")
    return p


def main():
    parser = build_arg_parser()
    args   = parser.parse_args()

    global DIRECT_THRESHOLD, MAX_K, MAX_STALE_TURNS
    DIRECT_THRESHOLD = args.direct_threshold
    MAX_K            = args.max_k
    MAX_STALE_TURNS  = args.max_stale_turns

    print(f"\n[Loader] Reading: {args.layouts}")
    all_layouts, furniture_items = load_layouts_from_json(args.layouts)

    validate_layouts(all_layouts, furniture_items, strict=args.strict_validation)

    required_furniture = get_required_furniture(args.furniture, furniture_items)

    filtered_layouts, furniture_excluded = filter_layouts_by_furniture(
        all_layouts, required_furniture
    )

    if len(filtered_layouts) < 2:
        print("[Error] After furniture filtering, fewer than 2 layouts remain.")
        return

    llm      = QwenAgent(model_id=args.llm)
    store    = LayoutVectorStore(model_name=args.embedder)
    user_ctx = UserContext()
    store.build(filtered_layouts)

    initial_active_ids: set[str] = {l["id"] for l in filtered_layouts}
    turn             = 0
    asked_questions: list[str] = []
    stale_turns      = 0          # ✅ NEW: tracks consecutive no-elimination turns

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 1 — K-means cluster elimination
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 55)
    print(" PHASE 1  —  Cluster-Based Elimination")
    print("=" * 55)

    while store.count > DIRECT_THRESHOLD:
        turn += 1
        n = store.count
        print(f"\n--- Turn {turn}  |  {n} layouts remaining ---")
        print("[Phase 1] Computing optimal K via silhouette score ...")

        k        = store.compute_dynamic_k()
        clusters = store.cluster(k)
        print(f"[Phase 1] K={k} clusters selected | Clusters formed:")
        for cid, cl_layouts in clusters.items():
            print(f"  Cluster {cid}: {[l['id'] for l in cl_layouts]}")

        print("[Phase 1] Summarising clusters ...")
        cluster_summaries: dict[int, str] = {}
        for cid, cl_layouts in clusters.items():
            descs   = [l["description"] for l in cl_layouts]
            summary = llm.summarise_cluster(descs)
            cluster_summaries[cid] = summary
            print(f"  Cluster {cid} ({len(cl_layouts)} layouts): {summary}")

        if user_ctx.likes or user_ctx.dislikes or user_ctx.unsure:
            print(f"\n[Context] {user_ctx}")

        question, option_map = llm.generate_cluster_question(
            cluster_summaries, asked_questions, user_ctx
        )
        asked_questions.append(question)

        print(f"\nChatbot: {question}")
        raw_input = input("    You: ").strip()

        if not raw_input:
            print("[System] Empty input — please type an answer.")
            asked_questions.pop()
            continue

        user_input = llm.refine_user_answer(question, raw_input)
        if user_input != raw_input:
            print(f"[Refine] Interpreted as: \"{user_input}\"")

        is_unsure = llm.detect_unsure(question, user_input)
        if is_unsure:
            topic = question[:60] + ("..." if len(question) > 60 else "")
            user_ctx.add_unsure(topic)
            print("[System] Got it — skipping that topic and trying something else.")
            print(f"[Context] Marked as unsure: '{topic}'")
            continue

        keep_ids = llm.interpret_cluster_choice(
            question, cluster_summaries, user_input, option_map
        )
        print(f"[System] Keeping clusters: {keep_ids}")

        keep_layout_ids = {l["id"] for cid in keep_ids for l in clusters[cid]}
        eliminate_ids   = [l["id"] for l in store.active_layouts
                           if l["id"] not in keep_layout_ids]

        if not eliminate_ids:
            stale_turns += 1
            print(f"[System] No layouts eliminated this turn. "
                  f"[stale {stale_turns}/{MAX_STALE_TURNS}]")
            # ✅ NEW: bail out to Phase 2 after too many stale turns
            if stale_turns >= MAX_STALE_TURNS:
                print(f"[System] {MAX_STALE_TURNS} consecutive stale turns — "
                      f"dropping to Phase 2 early.")
                break
            continue

        # Successful elimination → reset stale counter
        stale_turns = 0

        kept_summaries    = [cluster_summaries[cid] for cid in keep_ids
                             if cid in cluster_summaries]
        dropped_summaries = [cluster_summaries[cid]
                             for cid in cluster_summaries if cid not in keep_ids]
        llm.update_user_context(question, user_input,
                                kept_summaries, dropped_summaries, user_ctx)

        store.eliminate(eliminate_ids)

        if store.count == 0:
            print("[System] ⚠ All layouts eliminated — stopping.")
            return

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 2 — Direct elimination
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f" PHASE 2  —  Direct Elimination  ({store.count} layouts left)")
    print("=" * 55)

    stale_turns = 0   # reset for Phase 2

    while store.count > 1:
        turn += 1
        remaining = store.active_layouts
        print(f"\n--- Turn {turn}  |  {len(remaining)} layouts remaining ---")
        for i, l in enumerate(remaining, 1):
            print(f"  {i}. {l['id']}: {l['description'][:90]}...")

        if user_ctx.likes or user_ctx.dislikes or user_ctx.unsure:
            print(f"\n[Context] {user_ctx}")

        question = llm.generate_direct_question(remaining, asked_questions, user_ctx)
        asked_questions.append(question)

        print(f"\nChatbot: {question}")
        raw_input = input("    You: ").strip()

        if not raw_input:
            print("[System] Empty input — please type an answer.")
            asked_questions.pop()
            continue

        user_input = llm.refine_user_answer(question, raw_input)
        if user_input != raw_input:
            print(f"[Refine] Interpreted as: \"{user_input}\"")

        is_unsure = llm.detect_unsure(question, user_input)
        if is_unsure:
            topic = question[:60] + ("..." if len(question) > 60 else "")
            user_ctx.add_unsure(topic)
            print("[System] Got it — skipping that topic and trying something else.")
            print(f"[Context] Marked as unsure: '{topic}'")
            continue

        eliminate_ids = llm.interpret_direct_elim(
            question, remaining, user_input, user_ctx
        )

        if not eliminate_ids:
            stale_turns += 1
            print(f"[System] No layouts eliminated — please give a clearer preference. "
                  f"[stale {stale_turns}/{MAX_STALE_TURNS}]")
            if stale_turns >= MAX_STALE_TURNS:
                print("[System] Too many stale turns in Phase 2 — picking best match.")
                break
            continue

        stale_turns = 0

        kept_layouts    = [l for l in remaining if l["id"] not in set(eliminate_ids)]
        dropped_layouts = [l for l in remaining if l["id"] in set(eliminate_ids)]
        llm.update_user_context(
            question, user_input,
            [l["description"][:120] for l in kept_layouts],
            [l["description"][:120] for l in dropped_layouts],
            user_ctx,
        )

        store.eliminate(eliminate_ids)

    # ──────────────────────────────────────────────────────────────────────
    # FINAL RESULT
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" FINAL RECOMMENDATION")
    print("=" * 55)

    if user_ctx.likes or user_ctx.dislikes:
        print(f"\n[User Profile]\n{user_ctx.as_prompt_block()}")

    winner_id: str | None = None

    if store.active_layouts:
        winner    = store.active_layouts[0]
        winner_id = winner["id"]
        print(f"\n🏆  Best Layout:  {winner['id']}")
        print(f"\nDescription:\n{winner['description']}")
        print(f"\nFurniture positions (required items highlighted with ★):")
        for item in furniture_items:
            coords = winner.get(item, "N/A")
            star   = " ★" if item in required_furniture else ""
            print(f"  {item:12s}{star}: {coords}")
    else:
        print("⚠  No layouts remaining.")

    surviving_ids      = {l["id"] for l in store.active_layouts}
    process_eliminated = [
        lid for lid in initial_active_ids
        if lid not in surviving_ids
    ]

    print_unused_layouts_report(
        all_layouts        = all_layouts,
        furniture_excluded = furniture_excluded,
        process_eliminated = process_eliminated,
        winner_id          = winner_id,
        required_furniture = required_furniture,
    )


if __name__ == "__main__":
    main()