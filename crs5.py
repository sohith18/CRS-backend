import os
import math
import json
import hashlib
import re
import numpy as np
import faiss
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

GIBBERISH_SENTINEL = "__GIBBERISH__"
NEUTRAL_SENTINEL   = "__NEUTRAL__"

# ── Furniture list ────────────────────────────────────────────────────────────
FURNITURE_ITEMS = ["bed", "cupboard", "table", "chair", "sofa"]
FURNITURE_LABEL = ", ".join(FURNITURE_ITEMS)
# ─────────────────────────────────────────────────────────────────────────────

# ── Two strict question types ─────────────────────────────────────────────────
# Type A: Positional — asks about where a specific furniture item is located
# Type B: Open-space — asks about where the open floor area is
QUESTION_TYPE_SYSTEM = (
    "You write simple, friendly room layout questions for everyday people.\n"
    "Read two layout descriptions and write ONE short question a homeowner can "
    "easily answer — no jargon, no technical terms.\n\n"
    "The question MUST be one of these two types — pick whichever fits the contrast best:\n\n"
    "TYPE A — WHERE IS THE FURNITURE:\n"
    "  Ask where ONE specific item goes. Use plain directions only.\n"
    "  Allowed directions: top-left corner, top-right corner, bottom-left corner, "
    "  bottom-right corner, left side, right side, middle of the room.\n"
    "  Good examples:\n"
    "    'Would you rather have the bed in the top-left corner or on the right side?'\n"
    "    'Do you want the sofa on the left side or in the bottom-right corner?'\n\n"
    "TYPE B — HOW DOES THE ROOM FEEL:\n"
    "  Ask about open space, coziness, or how the room feels to live in.\n"
    "  Good examples:\n"
    "    'Do you prefer lots of open floor space in the middle, or a fully furnished cozy room?'\n"
    "    'Would you rather have one open empty side of the room, or furniture spread all around?'\n"
    "    'Do you like a room that feels spacious and open, or one that feels full and cozy?'\n\n"
    "Hard rules:\n"
    "1. Plain English only — no words like 'perimeter', 'quadrant', 'density', 'zone', 'anchor', 'asymmetric'.\n"
    "2. Maximum 20 words.\n"
    "3. Must end with '?'.\n"
    "4. Offer exactly two clear options joined by 'or'.\n"
    "5. No preamble. Output the question only."
)
# ─────────────────────────────────────────────────────────────────────────────


# ==========================================
# 1. LAYOUTS — all contain FURNITURE_ITEMS
# ==========================================
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
            "zones while the left side is dominated by sleeping and storage. Density is low-to-"
            "moderate with a clear left-wall anchor and open center-right movement space."
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
            "bed, creating a balanced perimeter arrangement. The style feels airy and "
            "symmetrical with the sleeping area as the undisputed room center."
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
            "a generous open movement zone. The layout is strongly asymmetric — all functional "
            "items packed on the right, open breathing space on the left. Density is high "
            "right, sparse left. The style is zone-concentrated and directional."
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
            "A sofa sits in the top-left corner above the bed, defining a relaxation zone "
            "separated from sleeping. The table occupies the top-right as a dedicated workspace, "
            "with a chair positioned just below it. The cupboard stands in the bottom-right "
            "corner for storage beside the bed. The layout achieves strong diagonal zone "
            "separation — sleeping and lounging bottom-left and top-left, working top-right, "
            "storage bottom-right — with open space in the center. Highly purposeful and "
            "organized with each item in its own clearly bounded quadrant."
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
            "keeping the sleeping and storage zones stacked vertically on one side. The table "
            "and chair are paired against the top-right wall forming a compact workspace. A sofa "
            "sits in the bottom-right area, creating a seating zone in the far corner. The center "
            "of the room is completely open. All items are wall-hugging with a large open center, "
            "giving a studio-like feel with perimeter activity and maximum floor visibility. "
            "Density is low overall with a clear left-sleep, right-work, bottom-right-lounge pattern."
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
            "left wall and a sofa on the right wall — creating a symmetrical sleeping zone with "
            "storage and seating on either side. A table sits centered in the lower half directly "
            "below the bed, with a chair at its base. The layout has a strong vertical axis of "
            "symmetry with the bed and table stacked centrally. Sleeping zone top-center, "
            "dining zone bottom-center, storage left, seating right. The style feels composed, "
            "formal, and architecturally balanced with moderate density."
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
            "The bed occupies the wide top-left area spanning over half the room width. A "
            "cupboard sits top-right adjacent to the bed, keeping storage within the sleeping "
            "zone. A table and chair are paired in the mid-right, forming a distinct workspace. "
            "A large sofa anchors the bottom-left as a dedicated living zone. The layout has "
            "four clearly delineated zones — sleeping top-left, storage top-right, workspace "
            "mid-right, seating bottom-left — with moderate-to-high density. The style is "
            "highly functional with strong zone separation and a multi-purpose feel."
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
            "right corner completing the diagonal. A sofa sits in the bottom-left, mirroring "
            "the cupboard across the diagonal and creating a seating anchor. Triangular open "
            "zones appear in the top-right and the center. Density is low-to-moderate with a "
            "dynamic flowing spatial feel. The style is modern and asymmetric with strong "
            "directional movement through the room."
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
            "completely open. The bed and cupboard occupy the left wall — sleeping top-left, "
            "storage bottom-left. The table and chair are aligned against the right wall — "
            "workspace top-right and mid-right. The sofa runs along the bottom wall between "
            "the two sides. The massive open center creates a studio-like movement space. "
            "Zones are separated strictly by wall: left for sleep and storage, right for work, "
            "bottom for seating. Density is low with high perimeter concentration. "
            "The style is maximally open and studio-functional."
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
            "quadrant with the cupboard immediately beside it and the table in the far top-"
            "right corner. A chair sits mid-right adjacent to the table. A large sofa fills "
            "the bottom-left. All zones are tightly arranged with minimal gaps — sleeping "
            "top-left, storage top-center, workspace top-right, seating bottom-left. "
            "Density is very high throughout with every wall section occupied. The style is "
            "cozy, maximally furnished, and apartment-efficient with no wasted space."
        ),
    },
]


# ==========================================
# 2. FAISS VECTOR STORE + EMBEDDING AXES
# ==========================================
class LayoutVectorStore:
    def __init__(self, model_name="BAAI/bge-m3"):
        print("[SYSTEM] Loading BGE-M3...")
        self.embedder         = SentenceTransformer(model_name)
        self.index            = None
        self.layout_ids       = []
        self.dim              = None
        self.embedding_matrix = None
        self.axes             = None
        self.axis_scores      = None
        self._index_path      = "layout_index.faiss"
        self._meta_path       = "layout_meta.json"

    def _compute_fingerprint(self, layouts):
        content = json.dumps(
            [{"id": l["id"], "description": l["description"]} for l in layouts],
            sort_keys=True
        )
        return hashlib.md5(content.encode()).hexdigest()

    def build_index(self, layouts):
        descriptions    = [l["description"] for l in layouts]
        self.layout_ids = [l["id"] for l in layouts]

        embeddings = self.embedder.encode(
            descriptions, normalize_embeddings=True, show_progress_bar=True
        )
        self.dim              = embeddings.shape[1]
        self.embedding_matrix = embeddings

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self._compute_axes()

        faiss.write_index(self.index, self._index_path)
        with open(self._meta_path, "w") as f:
            json.dump({
                "fingerprint":  self._compute_fingerprint(layouts),
                "layout_ids":   self.layout_ids,
                "dim":          self.dim,
                "descriptions": {l["id"]: l["description"] for l in layouts},
                "embeddings":   self.embedding_matrix.tolist(),
                "axes":         self.axes.tolist(),
                "axis_scores":  self.axis_scores.tolist(),
            }, f)

        print(f"[SYSTEM] Index built. {len(descriptions)} layouts | {self.axes.shape[0]} axes\n")

    def load_index(self, layouts):
        if not (os.path.exists(self._index_path) and os.path.exists(self._meta_path)):
            return False

        with open(self._meta_path) as f:
            meta = json.load(f)

        if meta.get("fingerprint") != self._compute_fingerprint(layouts):
            print("[SYSTEM] Descriptions changed — rebuilding index...")
            return False

        self.index            = faiss.read_index(self._index_path)
        self.layout_ids       = meta["layout_ids"]
        self.dim              = meta["dim"]
        self.embedding_matrix = np.array(meta["embeddings"],  dtype=np.float32)
        self.axes             = np.array(meta["axes"],         dtype=np.float32)
        self.axis_scores      = np.array(meta["axis_scores"],  dtype=np.float32)

        print(f"[SYSTEM] Index loaded. {len(self.layout_ids)} layouts | {self.axes.shape[0]} axes\n")
        return True

    def _compute_axes(self):
        E          = self.embedding_matrix
        E_centered = E - E.mean(axis=0, keepdims=True)
        _, S, Vt   = np.linalg.svd(E_centered, full_matrices=False)

        variances   = (S ** 2) / (S ** 2).sum()
        cumulative  = np.cumsum(variances)
        projections = E_centered @ Vt.T

        K = 0
        for i in range(len(S) - 1):
            spread = projections[:, i].max() - projections[:, i].min()
            if spread < 0.20:
                break
            K += 1
            if cumulative[i] >= 0.95:
                break
        K = max(2, K)

        self.axes        = Vt[:K]
        self.axis_scores = projections[:, :K]

        print(f"[SVD] Selected K={K} axes (spread≥0.20, cumvar={cumulative[K-1]*100:.1f}%)")
        for i in range(K):
            spread = self.axis_scores[:, i].max() - self.axis_scores[:, i].min()
            print(f"  Axis {i}: var={variances[i]*100:.1f}%  spread={spread:.4f}")

    def get_axis_contrast_pair(self, axis_idx):
        """Returns the single most extreme (high, low) description pair for this axis."""
        projections    = self.axis_scores[:, axis_idx]
        sorted_indices = np.argsort(projections)[::-1]
        high_desc = LAYOUTS[sorted_indices[0]]["description"]
        low_desc  = LAYOUTS[sorted_indices[-1]]["description"]
        return high_desc, low_desc

    def query(self, preference_text):
        query_vec = self.embedder.encode([preference_text], normalize_embeddings=True)
        sims, indices = self.index.search(
            np.array(query_vec, dtype=np.float32), len(self.layout_ids)
        )
        return {
            self.layout_ids[idx]: float(sim)
            for sim, idx in zip(sims[0], indices[0])
        }

    def compute_axis_reward(self, similarity_scores):
        return float(np.std(list(similarity_scores.values())))


# ==========================================
# 2B. ASKED-QUESTIONS VECTOR DB
# ==========================================
class AskedQuestionsDB:
    """
    Live FAISS vector DB of every question shown to the user this session.
    Prevents semantic duplicates across all turns.
    Threshold 0.72 catches paraphrases, not just near-identical wording.
    """
    def __init__(self, embedder, similarity_threshold=0.72):
        self.embedder             = embedder
        self.similarity_threshold = similarity_threshold
        self.index                = None
        self.questions            = []

    def _embed(self, text):
        return np.array(
            self.embedder.encode([text], normalize_embeddings=True), dtype=np.float32
        )

    def is_duplicate(self, question):
        """Returns (is_dup: bool, top_sim: float, nearest: str|None)."""
        if not self.questions:
            return False, 0.0, None
        emb = self._embed(question)
        if self.index is None:
            return False, 0.0, None
        sims, indices = self.index.search(emb, k=1)
        top_sim = float(sims[0][0])
        nearest = self.questions[int(indices[0][0])]
        return top_sim >= self.similarity_threshold, top_sim, nearest

    def register(self, question):
        emb = self._embed(question)
        if self.index is None:
            self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.questions.append(question)

    def recent_sample(self, n=5):
        """Return up to n most recently asked questions for prompt context."""
        return self.questions[-n:]


# ==========================================
# 3. EMBEDDING-SPACE UCB BANDIT
# ==========================================
class EmbeddingUCBBandit:
    def __init__(self, num_axes):
        self.num_arms    = num_axes
        self.counts      = [0]   * num_axes
        self.values      = [0.0] * num_axes
        self.total_pulls = 0

    def select_arm(self):
        unexplored = [i for i in range(self.num_arms) if self.counts[i] == 0]
        if unexplored:
            return unexplored[0]
        ucb = {
            i: self.values[i] + math.sqrt(2 * math.log(self.total_pulls) / self.counts[i])
            for i in range(self.num_arms)
        }
        return max(ucb, key=ucb.get)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.total_pulls += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) / n) * self.values[arm] + (1 / n) * reward

    def status(self):
        print("\n[Bandit] Arm status:")
        for i in range(self.num_arms):
            bar = "█" * int(self.values[i] * 200)
            print(f"  Axis {i}: value={self.values[i]:.4f}  pulls={self.counts[i]}  {bar}")


# ==========================================
# 4. THE LLM AGENT — Qwen3.5-9B, thinking OFF
# ==========================================
class QwenAgent:
    # Zones that must appear in positional questions
    _POSITIONAL_ZONES = {
        "top-left", "top-right", "bottom-left", "bottom-right",
        "left wall", "right wall", "top wall", "bottom wall", "center",
    }
    # Zones that must appear in open-space questions
    _OPENSPACE_ZONES = {
        "center", "left side", "right side", "top half", "bottom half",
    }


    def __init__(self, model_id="Qwen/Qwen3.5-9B"):
        print(f"[SYSTEM] Loading {model_id}...")
        self.model_id  = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model     = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[SYSTEM] Model on: {next(self.model.parameters()).device}\n")

    @staticmethod
    def _clean_output(text):
        text = text.strip()
        text = re.sub(r"^['\"""'']+|['\"""'']+$", "", text)
        text = re.sub(r"^(Question|Q)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\d+[\).\s-]+", "", text)
        # Keep only the first sentence that ends with '?'
        for line in text.split("\n"):
            line = line.strip()
            if line.endswith("?"):
                return line
        return text.split("\n")[0].strip()

    def _generate(self, system_prompt, user_prompt,
                  max_new_tokens=120, temperature=0.35,
                  top_p=0.95, top_k=20, repetition_penalty=1.1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        use_sampling = temperature > 0
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=use_sampling,
                temperature=temperature if use_sampling else None,
                top_p=top_p             if use_sampling else None,
                top_k=top_k             if use_sampling else None,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        raw = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw

    # ------------------------------------------------------------------
    def generate_question(self, axis_idx, vector_store, asked_db, max_retries=8):
        high_desc, low_desc = vector_store.get_axis_contrast_pair(axis_idx)

        recent = asked_db.recent_sample(n=6)
        avoid_block = ""
        if recent:
            avoid_block = (
                "\n\nDo NOT repeat or rephrase any of these already-asked questions:\n"
                + "\n".join(f"  - {q}" for q in recent)
                + "\n"
            )

        user_prompt = (
            f"{avoid_block}\n"
            f"Layout A (high extreme):\n{high_desc}\n\n"
            f"Layout B (low extreme):\n{low_desc}\n\n"
            "Simple question:"
        )


        best_candidate = None
        best_sim       = float("inf")

        print(f"  [Axis {axis_idx}] Generating question...")

        for attempt in range(max_retries):
            temp    = max(0.45 - 0.05 * attempt, 0.10)
            rep_pen = min(1.1  + 0.05 * attempt, 1.35)

            raw = self._generate(
                system_prompt=QUESTION_TYPE_SYSTEM,
                user_prompt=user_prompt,
                max_new_tokens=60,         # shorter → less room for jargon
                temperature=temp,
                top_p=0.95,
                top_k=40,
                repetition_penalty=rep_pen,
            )
            q = self._clean_output(raw)

            # 2. Semantic dedup check
            is_dup, sim_score, nearest = asked_db.is_duplicate(q)

            if sim_score < best_sim:
                best_candidate = q
                best_sim       = sim_score

            if not is_dup:
                asked_db.register(q)
                print(f"    → ACCEPTED  | sim={sim_score:.3f} | {q}")
                return q

            print(f"    → DUPLICATE | sim={sim_score:.3f} | attempt={attempt+1} | {q}")
            if nearest:
                print(f"      nearest: {nearest}")

        if best_candidate:
            asked_db.register(best_candidate)
            print(f"    → FORCED    | best_sim={best_sim:.3f} | {best_candidate}")
            return best_candidate

        fallback = "Do you prefer lots of open space in the middle, or a fully furnished room?"
        asked_db.register(fallback)
        print(f"    → FALLBACK  | {fallback}")
        return fallback
    # ------------------------------------------------------------------
    def extract_preference_statement(self, question, user_input):
        system = (
            "You are a spatial preference interpreter for a room layout vector database. "
            "Convert the user's answer into ONE positive, factual sentence describing "
            "what they want in their room layout.\n"
            "Rules:\n"
            f"1. The room contains: {FURNITURE_LABEL}. "
            "   Name the specific item if the user's answer refers to one of them.\n"
            "2. Include wall position or zone where relevant "
            "   (top-left, right wall, center, bottom corner, etc.).\n"
            "3. Remove ALL negations — state what they DO want.\n"
            "4. Translate regional terms: almara/almirah → wardrobe/cupboard, "
            "   takhat/palang → bed, kursi → chair, divan → sofa, meja → table.\n"
            f"5. If gibberish or completely off-topic → output exactly: {GIBBERISH_SENTINEL}\n"
            f"6. If no preference / uncertain / indifferent → output exactly: {NEUTRAL_SENTINEL}\n"
            "7. ONE sentence only. No extra text.\n"
            "Examples:\n"
            "   Q: 'Where do you want the bed?' / A: 'top left corner' "
            "   → 'The bed is positioned in the top-left corner of the room.'\n"
            "   Q: 'Sofa on same side as bed or opposite wall?' / A: 'opposite' "
            "   → 'The sofa is placed on the opposite wall from the bed.'\n"
            "   Q: 'Open space in center or on one side?' / A: 'middle open' "
            "   → 'The center of the room is kept open with furniture arranged around the perimeter.'"
        )
        user = (
            f'Question: "{question}"\n'
            f'User answered: "{user_input}"\n'
            "Spatial preference sentence:"
        )
        result = self._generate(
            system_prompt=system,
            user_prompt=user,
            max_new_tokens=100,
            temperature=0.1,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
        )
        result = result.strip().split("\n")[0].strip()
        if GIBBERISH_SENTINEL in result:
            return GIBBERISH_SENTINEL
        if NEUTRAL_SENTINEL in result:
            return NEUTRAL_SENTINEL
        return result


# ==========================================
# 5. LAYOUT VALIDATION
# ==========================================
def validate_layouts(layouts, furniture_items):
    errors = []
    for layout in layouts:
        missing = [item for item in furniture_items if item not in layout]
        if missing:
            errors.append(f"  {layout['id']} missing: {missing}")
    if errors:
        raise ValueError(
            f"Layout validation failed — missing items from "
            f"FURNITURE_ITEMS {furniture_items}:\n" + "\n".join(errors)
        )
    print(f"[Validation] All {len(layouts)} layouts contain: {furniture_items} ✅\n")


# ==========================================
# 6. DELTA-BASED REWARD
# ==========================================
def apply_balanced_reward(layouts, similarity_scores):
    scores   = [similarity_scores.get(l["id"], 0.0) for l in layouts]
    mean_sim = float(np.mean(scores))
    std_sim  = float(np.std(scores))
    print(f"   [Stats] mean={mean_sim:.4f}  std={std_sim:.4f}")
    for layout in layouts:
        sim   = similarity_scores.get(layout["id"], 0.0)
        delta = sim - mean_sim
        layout["score"] += delta
        tag = f"Δ={delta:+.4f} ✅" if delta > 0 else f"Δ={delta:+.4f} ❌"
        print(f"   [{layout['id']:10s}] score={layout['score']:+.4f}  sim={sim:.4f}  {tag}")


# ==========================================
# 7. ORCHESTRATOR
# ==========================================
def main():
    for layout in LAYOUTS:
        layout["score"] = 0.0

    validate_layouts(LAYOUTS, FURNITURE_ITEMS)

    llm          = QwenAgent(model_id="Qwen/Qwen3.5-9B")
    vector_store = LayoutVectorStore()

    print("=" * 45)
    print(" PHASE 1: SETUP")
    print("=" * 45)
    print(f"[Config] Furniture items: {FURNITURE_LABEL}")
    print(f"[Config] Layouts: {len(LAYOUTS)}\n")

    if not vector_store.load_index(LAYOUTS):
        print("[SYSTEM] Building FAISS index from scratch...")
        vector_store.build_index(LAYOUTS)

    num_axes = vector_store.axes.shape[0]

    # Live asked-questions DB — starts empty every session
    asked_db = AskedQuestionsDB(
        embedder=vector_store.embedder,
        similarity_threshold=0.72,
    )

    bandit = EmbeddingUCBBandit(num_axes)
    print(f"[Bandit] Initialized: {num_axes} axes\n")

    print("=" * 45)
    print(" PHASE 2: USER SESSION")
    print("=" * 45)

    MAX_TURNS      = 20
    turns_consumed = 0

    while turns_consumed < MAX_TURNS:
        print(f"\n--- Turn {turns_consumed + 1} of {MAX_TURNS} ---")

        axis_idx = bandit.select_arm()

        # Generate question live — checked against asked_db immediately
        question = llm.generate_question(axis_idx, vector_store, asked_db)

        pull_num = bandit.counts[axis_idx] + 1
        print(f"[System] Axis {axis_idx}  pull #{pull_num}")
        print(f"Chatbot: {question}")
        user_input = input("    You: ").strip()

        if not user_input:
            print("[System] Empty input — please type a response.")
            # Un-register the question so it can be regenerated next time
            asked_db.questions.pop()
            asked_db.index = None
            if asked_db.questions:
                for q in asked_db.questions:
                    emb = asked_db._embed(q)
                    if asked_db.index is None:
                        asked_db.index = faiss.IndexFlatIP(emb.shape[1])
                    asked_db.index.add(emb)
            continue

        preference_text = llm.extract_preference_statement(question, user_input)
        print(f'[System] Preference → "{preference_text}"')

        if preference_text == GIBBERISH_SENTINEL:
            print("[System] ⚠ Input not meaningful — please answer the question.")
            continue

        if preference_text == NEUTRAL_SENTINEL:
            print("[System] ↔ No preference expressed — turn consumed, reward=0.")
            bandit.update(axis_idx, reward=0.0)
            turns_consumed += 1
            continue

        similarity_scores = vector_store.query(preference_text)
        reward            = vector_store.compute_axis_reward(similarity_scores)
        bandit.update(axis_idx, reward)
        print(f"[Bandit] Axis {axis_idx} reward = {reward:.4f}")

        print("[System] Applying rewards:")
        apply_balanced_reward(LAYOUTS, similarity_scores)

        turns_consumed += 1

    bandit.status()

    print("\n" + "=" * 45)
    print(" FINAL RECOMMENDATIONS")
    print("=" * 45)
    LAYOUTS.sort(key=lambda x: x["score"], reverse=True)
    for i, layout in enumerate(LAYOUTS):
        print(f"\n  Rank {i+1}: '{layout['id']}' | Score: {layout['score']:+.4f}")
        print(f"  {layout['description'][:120]}...")


if __name__ == "__main__":
    main()