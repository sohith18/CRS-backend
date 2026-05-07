"""
Agentic RAG — Room Layout Recommendation
=========================================
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
"""

import os, re, random
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

# ── Sentinels ─────────────────────────────────────────────────────────────────
KEEP_ALL_SENTINEL = "__KEEP_ALL__"
UNSURE_SENTINEL   = "__UNSURE__"

# ── Knobs ─────────────────────────────────────────────────────────────────────
DIRECT_THRESHOLD = 5
N_SAMPLE         = 10
MAX_K            = 10
EMBEDDER_MODEL   = "BAAI/bge-m3"
LLM_MODEL_ID     = "Qwen/Qwen3.5-9B"

# ── Furniture ─────────────────────────────────────────────────────────────────
FURNITURE_ITEMS = ["bed", "cupboard", "table", "chair", "sofa"]
FURNITURE_LABEL = ", ".join(FURNITURE_ITEMS)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LAYOUTS
# ══════════════════════════════════════════════════════════════════════════════
LAYOUTS = [
    {
        "id": "layout_01",
        "bed":      [0.02, 0.02, 0.40, 0.45],
        "cupboard": [0.02, 0.50, 0.18, 0.85],
        "table":    [0.45, 0.55, 0.70, 0.80],
        "chair":    [0.45, 0.05, 0.58, 0.22],
        "sofa":     [0.42, 0.83, 0.78, 0.98],
        "description": (
            "The bed occupies the top-left corner, spanning a large portion of the left wall. "
            "The cupboard sits directly below it on the same left wall, keeping sleeping and "
            "storage tightly together on one side. A chair is tucked into the top-right area "
            "as a compact reading spot. A table sits center-right with a sofa running along "
            "the bottom edge beneath it. The right half and bottom of the room hold the active "
            "zones while the left side is dominated by sleeping and storage."
        ),
    },
    {
        "id": "layout_02",
        "bed":      [0.30, 0.35, 0.70, 0.75],
        "cupboard": [0.82, 0.05, 0.98, 0.55],
        "table":    [0.05, 0.70, 0.25, 0.95],
        "chair":    [0.78, 0.65, 0.92, 0.90],
        "sofa":     [0.28, 0.02, 0.72, 0.20],
        "description": (
            "The bed is positioned dead center in the room, acting as the focal point with "
            "clearance on all sides. A sofa lines the top wall above the bed, creating a "
            "relaxed zone at the room's entrance. The cupboard stands against the right wall "
            "for storage, and a chair occupies the bottom-right corner. A table anchors the "
            "bottom-left corner. Furniture is distributed to all four walls around the central "
            "bed, creating a balanced arrangement. The style feels airy and symmetrical."
        ),
    },
    {
        "id": "layout_03",
        "bed":      [0.55, 0.05, 0.98, 0.50],
        "cupboard": [0.60, 0.55, 0.78, 0.95],
        "table":    [0.80, 0.55, 0.98, 0.80],
        "chair":    [0.55, 0.55, 0.62, 0.75],
        "sofa":     [0.02, 0.60, 0.45, 0.95],
        "description": (
            "The bed dominates the right wall from top to mid-room. The cupboard, table, and "
            "chair are clustered in the bottom-right, forming a dense right-side activity zone. "
            "A large sofa occupies the bottom-left corner, facing inward across the open left "
            "half. The entire left half above the sofa is completely free of furniture, creating "
            "a generous open movement zone. All functional items are packed on the right, open "
            "breathing space on the left."
        ),
    },
    {
        "id": "layout_04",
        "bed":      [0.05, 0.60, 0.55, 0.98],
        "cupboard": [0.60, 0.60, 0.78, 0.98],
        "table":    [0.60, 0.05, 0.90, 0.30],
        "chair":    [0.62, 0.32, 0.75, 0.48],
        "sofa":     [0.02, 0.05, 0.45, 0.22],
        "description": (
            "The bed spans the entire bottom-left, anchored to the bottom and left walls. "
            "A sofa sits in the top-left corner above the bed, defining a relaxation zone. "
            "The table occupies the top-right as a dedicated workspace, with a chair just below. "
            "The cupboard stands in the bottom-right corner. Strong diagonal zone separation — "
            "sleeping and lounging on the left, working top-right, storage bottom-right — "
            "with open space in the center."
        ),
    },
    {
        "id": "layout_05",
        "bed":      [0.03, 0.03, 0.35, 0.38],
        "cupboard": [0.03, 0.62, 0.15, 0.97],
        "table":    [0.60, 0.05, 0.85, 0.28],
        "chair":    [0.88, 0.05, 0.98, 0.28],
        "sofa":     [0.40, 0.70, 0.80, 0.95],
        "description": (
            "The bed and cupboard both hug the left wall — bed top-left, cupboard bottom-left — "
            "keeping sleeping and storage stacked vertically on one side. The table and chair "
            "are paired against the top-right wall forming a compact workspace. A sofa sits "
            "in the bottom-right area. The center of the room is completely open. All items "
            "are wall-hugging with a large open center, giving a studio-like feel with "
            "perimeter activity and maximum floor visibility."
        ),
    },
    {
        "id": "layout_06",
        "bed":      [0.30, 0.05, 0.70, 0.42],
        "cupboard": [0.02, 0.10, 0.22, 0.55],
        "table":    [0.35, 0.60, 0.65, 0.85],
        "chair":    [0.42, 0.87, 0.58, 0.98],
        "sofa":     [0.78, 0.10, 0.98, 0.55],
        "description": (
            "The bed is centered in the top half of the room, flanked by a cupboard on the "
            "left wall and a sofa on the right wall — a symmetrical sleeping zone. A table "
            "sits centered in the lower half directly below the bed, with a chair at its base. "
            "The layout has a strong vertical axis of symmetry. Sleeping zone top-center, "
            "dining zone bottom-center, storage left, seating right. Formal and "
            "architecturally balanced."
        ),
    },
    {
        "id": "layout_07",
        "bed":      [0.02, 0.02, 0.55, 0.42],
        "cupboard": [0.58, 0.02, 0.75, 0.42],
        "table":    [0.58, 0.48, 0.80, 0.72],
        "chair":    [0.82, 0.48, 0.98, 0.72],
        "sofa":     [0.05, 0.70, 0.45, 0.98],
        "description": (
            "The bed occupies the wide top-left area spanning over half the room width. "
            "A cupboard sits top-right adjacent to the bed. A table and chair are paired "
            "mid-right, forming a workspace. A large sofa anchors the bottom-left as a "
            "dedicated living zone. Four clearly delineated zones — sleeping top-left, "
            "storage top-right, workspace mid-right, seating bottom-left — with strong "
            "zone separation and a multi-purpose feel."
        ),
    },
    {
        "id": "layout_08",
        "bed":      [0.02, 0.02, 0.38, 0.40],
        "cupboard": [0.68, 0.65, 0.85, 0.98],
        "table":    [0.35, 0.30, 0.60, 0.58],
        "chair":    [0.55, 0.55, 0.70, 0.75],
        "sofa":     [0.02, 0.68, 0.35, 0.98],
        "description": (
            "Furniture follows a clear diagonal flow from top-left to bottom-right. The bed "
            "anchors the top-left corner, a table sits center bridging sleep and work, and a "
            "chair is placed center-right beside the table. The cupboard anchors the bottom-"
            "right corner completing the diagonal. A sofa sits in the bottom-left. Triangular "
            "open zones appear in the top-right and the center. Modern and asymmetric with "
            "strong directional movement through the room."
        ),
    },
    {
        "id": "layout_09",
        "bed":      [0.00, 0.02, 0.30, 0.45],
        "cupboard": [0.00, 0.55, 0.18, 0.90],
        "table":    [0.82, 0.02, 0.98, 0.35],
        "chair":    [0.82, 0.40, 0.98, 0.60],
        "sofa":     [0.30, 0.82, 0.75, 0.98],
        "description": (
            "All furniture is pushed to the perimeter walls, leaving the entire center "
            "completely open. The bed and cupboard occupy the left wall. The table and chair "
            "are aligned against the right wall. The sofa runs along the bottom wall. "
            "The massive open center creates a studio-like movement space. Zones are separated "
            "strictly by wall: left for sleep and storage, right for work, bottom for seating. "
            "Maximally open and studio-functional."
        ),
    },
    {
        "id": "layout_10",
        "bed":      [0.02, 0.02, 0.48, 0.48],
        "cupboard": [0.52, 0.02, 0.70, 0.48],
        "table":    [0.72, 0.02, 0.98, 0.35],
        "chair":    [0.72, 0.38, 0.88, 0.58],
        "sofa":     [0.02, 0.55, 0.45, 0.80],
        "description": (
            "All five items pack the room with high density. The bed fills the top-left "
            "with the cupboard immediately beside it and the table in the far top-right. "
            "A chair sits mid-right adjacent to the table. A large sofa fills the bottom-left. "
            "All zones are tightly arranged with minimal gaps. Density is very high with every "
            "wall section occupied. Cozy, maximally furnished, and apartment-efficient "
            "with no wasted space."
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# 2. USER CONTEXT
# ══════════════════════════════════════════════════════════════════════════════
class UserContext:
    """
    Maintains a running summary of what the user has definitely liked,
    definitely disliked, and is currently unsure about.
    Updated after every answered (non-unsure) turn.
    """

    def __init__(self):
        self.likes:    list[str] = []   # confirmed preferences
        self.dislikes: list[str] = []   # confirmed rejections
        self.unsure:   list[str] = []   # topics they couldn't decide on

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
        """Formatted block injected into every question-generation prompt."""
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
        self.embedder       = SentenceTransformer(model_name)
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
    """
    Tasks:
      1. summarise_cluster          — one-sentence summary of a cluster's spatial feel
      2. generate_cluster_question  — conversational question (no numbered options)
      3. detect_unsure              — is the user's answer uncertain/unclear?
      4. interpret_cluster_choice   — which cluster IDs to KEEP
      5. update_user_context        — extract like/dislike facts from a confirmed answer
      6. generate_direct_question   — pinpoint question for Phase 2
      7. interpret_direct_elim      — which layout IDs to ELIMINATE
    """

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

    # ── core generation ────────────────────────────────────────────────────────
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

    # ── 1. Cluster summarisation ───────────────────────────────────────────────
    def summarise_cluster(self, descriptions: list[str]) -> str:
        sample = random.sample(descriptions, min(N_SAMPLE, len(descriptions)))
        system = (
            "You summarise room layout styles.\n"
            "Write ONE sentence (max 30 words) capturing the shared spatial feel: "
            "where the bed is, how open the room feels, and the general arrangement.\n"
            "Plain English only. No jargon. Output the sentence only."
        )
        user = (
            "Layouts:\n" + "\n---\n".join(sample) +
            "\n\nSummarise the shared style in one sentence:"
        )
        return self._first_line(self._generate(system, user, max_new_tokens=70, temperature=0.2))

    # ── 2. Cluster question generation ─────────────────────────────────────────
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
        history  = self._history_block(asked_questions)
        ctx_block = user_ctx.as_prompt_block()

        system = (
            "You help someone find their ideal room layout through friendly conversation.\n"
            "Given summaries of different room style groups, ask ONE natural question "
            "(max 20 words) that reveals which group fits the user best.\n"
            "Rules:\n"
            "• Do NOT list numbered options or say '1)', '2)', 'Style 1', etc.\n"
            "• Ask about a single concrete preference (e.g. bed placement, open space, "
            "  workspace, storage position).\n"
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

    # ── 3. Detect unsure ──────────────────────────────────────────────────────
    def detect_unsure(self, question: str, user_input: str) -> bool:
        """
        Returns True if the user's answer signals uncertainty or inability to decide.
        Uses a lightweight LLM call (very short output).
        """
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

    # ── 4. Interpret cluster choice ────────────────────────────────────────────
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
        system = (
            "You interpret a user's room layout preference.\n"
            "Given style summaries and the user's free-text answer, "
            "output ONLY a comma-separated list of style NUMBERS whose description "
            "best matches what the user said.\n"
            "Examples: '2'  or  '1,3'\n"
            f"If the answer is unclear or the user wants all, output: {KEEP_ALL_SENTINEL}\n"
            "No other text."
        )
        user = (
            f"Question asked: {question}\n"
            f"Style summaries:\n{styles_block}\n"
            f"User said: \"{user_input}\"\n"
            "Style numbers to keep:"
        )
        raw = self._first_line(
            self._generate(system, user, max_new_tokens=20, temperature=0.1)
        )

        if KEEP_ALL_SENTINEL in raw:
            return list(cluster_summaries.keys())

        kept_ids = []
        for token in raw.replace(",", " ").split():
            if token.isdigit():
                num = int(token)
                if num in option_map:
                    kept_ids.append(option_map[num])

        return kept_ids if kept_ids else list(cluster_summaries.keys())

    # ── 5. Update user context from a confirmed answer ─────────────────────────
    def update_user_context(
        self,
        question:          str,
        user_input:        str,
        kept_summaries:    list[str],   # summaries of clusters/layouts being KEPT
        dropped_summaries: list[str],   # summaries of clusters/layouts being DROPPED
        user_ctx:          UserContext,
    ):
        """
        Extract one short like-fact and one short dislike-fact from this turn
        and append them to the UserContext.
        """
        kept_txt    = "; ".join(kept_summaries)   or "none"
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

        like_fact    = ""
        dislike_fact = ""
        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("LIKE:"):
                like_fact = line.split(":", 1)[1].strip()
            elif line.upper().startswith("DISLIKE:"):
                dislike_fact = line.split(":", 1)[1].strip()

        if like_fact and like_fact.lower() != "nothing":
            user_ctx.add_like(like_fact)
        if dislike_fact and dislike_fact.lower() != "nothing":
            user_ctx.add_dislike(dislike_fact)

    # ── 6. Direct question generation ─────────────────────────────────────────
    def generate_direct_question(
        self,
        layouts:         list[dict],
        asked_questions: list[str],
        user_ctx:        UserContext,
    ) -> str:
        desc_block = "\n\n".join(
            f"[{l['id']}]: {l['description']}" for l in layouts
        )
        history   = self._history_block(asked_questions)
        ctx_block = user_ctx.as_prompt_block()

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

    # ── 7. Interpret direct elimination ───────────────────────────────────────
    def interpret_direct_elim(
        self,
        question:   str,
        layouts:    list[dict],
        user_input: str,
        user_ctx:   UserContext,
    ) -> list[str]:
        valid_ids  = {l["id"] for l in layouts}
        desc_block = "\n\n".join(
            f"{l['id']}: {l['description'][:220]}" for l in layouts
        )
        ctx_block = user_ctx.as_prompt_block()

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

        eliminate = [
            tok.strip()
            for tok in raw.split(",")
            if tok.strip() in valid_ids
        ]
        if len(eliminate) >= len(layouts):
            eliminate = eliminate[: len(layouts) - 1]
        return eliminate


# ══════════════════════════════════════════════════════════════════════════════
# 5. VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def validate_layouts(layouts: list[dict], furniture: list[str]):
    errors = []
    for l in layouts:
        missing = [f for f in furniture if f not in l]
        if missing:
            errors.append(f"  {l['id']} missing keys: {missing}")
    if errors:
        raise ValueError("Layout validation failed:\n" + "\n".join(errors))
    print(f"[Validation] All {len(layouts)} layouts have required keys ✅\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
def main():
    validate_layouts(LAYOUTS, FURNITURE_ITEMS)

    llm      = QwenAgent()
    store    = LayoutVectorStore()
    user_ctx = UserContext()
    store.build(LAYOUTS)

    turn             = 0
    asked_questions: list[str] = []

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1  —  K-means cluster elimination
    # ──────────────────────────────────────────────────────────────────────────
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

        # Show running user context
        if user_ctx.likes or user_ctx.dislikes or user_ctx.unsure:
            print(f"\n[Context] {user_ctx}")

        question, option_map = llm.generate_cluster_question(
            cluster_summaries, asked_questions, user_ctx
        )
        asked_questions.append(question)

        print(f"\nChatbot: {question}")
        user_input = input("    You: ").strip()

        if not user_input:
            print("[System] Empty input — please type an answer.")
            asked_questions.pop()
            continue

        # ── Unsure detection ──────────────────────────────────────────────────
        is_unsure = llm.detect_unsure(question, user_input)
        if is_unsure:
            topic = question[:60] + ("..." if len(question) > 60 else "")
            user_ctx.add_unsure(topic)
            print(f"[System] Got it — skipping that topic and trying something else.")
            print(f"[Context] Marked as unsure: '{topic}'")
            # Keep all clusters, ask a different question next turn (question stays in history)
            continue

        # ── Normal interpretation ─────────────────────────────────────────────
        keep_ids = llm.interpret_cluster_choice(
            question, cluster_summaries, user_input, option_map
        )
        print(f"[System] Keeping clusters: {keep_ids}")

        keep_layout_ids = {
            l["id"]
            for cid in keep_ids
            for l in clusters[cid]
        }
        eliminate_ids = [l["id"] for l in store.active_layouts
                         if l["id"] not in keep_layout_ids]

        if not eliminate_ids:
            print("[System] No layouts eliminated this turn (ambiguous answer).")
            continue

        # Update user context from this confirmed preference
        kept_summaries    = [cluster_summaries[cid] for cid in keep_ids
                             if cid in cluster_summaries]
        dropped_summaries = [cluster_summaries[cid]
                             for cid in cluster_summaries if cid not in keep_ids]
        llm.update_user_context(
            question, user_input,
            kept_summaries, dropped_summaries,
            user_ctx
        )

        store.eliminate(eliminate_ids)

        if store.count == 0:
            print("[System] ⚠ All layouts eliminated — something went wrong. Stopping.")
            return

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2  —  Direct elimination (few layouts)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f" PHASE 2  —  Direct Elimination  ({store.count} layouts left)")
    print("=" * 55)

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
        user_input = input("    You: ").strip()

        if not user_input:
            print("[System] Empty input — please type an answer.")
            asked_questions.pop()
            continue

        # ── Unsure detection ──────────────────────────────────────────────────
        is_unsure = llm.detect_unsure(question, user_input)
        if is_unsure:
            topic = question[:60] + ("..." if len(question) > 60 else "")
            user_ctx.add_unsure(topic)
            print(f"[System] Got it — skipping that topic and trying something else.")
            print(f"[Context] Marked as unsure: '{topic}'")
            continue

        # ── Normal elimination ────────────────────────────────────────────────
        eliminate_ids = llm.interpret_direct_elim(
            question, remaining, user_input, user_ctx
        )

        if not eliminate_ids:
            print("[System] No layouts eliminated — please give a clearer preference.")
            continue

        # Update user context
        kept_layouts    = [l for l in remaining if l["id"] not in set(eliminate_ids)]
        dropped_layouts = [l for l in remaining if l["id"] in set(eliminate_ids)]
        llm.update_user_context(
            question, user_input,
            [l["description"][:120] for l in kept_layouts],
            [l["description"][:120] for l in dropped_layouts],
            user_ctx
        )

        store.eliminate(eliminate_ids)

    # ──────────────────────────────────────────────────────────────────────────
    # FINAL RESULT
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" FINAL RECOMMENDATION")
    print("=" * 55)

    if user_ctx.likes or user_ctx.dislikes:
        print(f"\n[User Profile]\n{user_ctx.as_prompt_block()}")

    if store.active_layouts:
        winner = store.active_layouts[0]
        print(f"\n🏆  Best Layout:  {winner['id']}")
        print(f"\nDescription:\n{winner['description']}")
        print(f"\nFurniture positions:")
        for item in FURNITURE_ITEMS:
            coords = winner.get(item, "N/A")
            print(f"  {item:10s}: {coords}")
    else:
        print("⚠  No layouts remaining.")


if __name__ == "__main__":
    main()