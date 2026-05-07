import os
import math
import json
import hashlib
import random
import re
import numpy as np
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


GIBBERISH_SENTINEL = "__GIBBERISH__"
NEUTRAL_SENTINEL   = "__NEUTRAL__"
QUESTION_BANK_PATH = "question_bank.json"


# ── Furniture list ────────────────────────────────────────────────────────────
FURNITURE_ITEMS = ["bed", "cupboard", "table", "chair", "sofa"]
FURNITURE_LABEL = ", ".join(FURNITURE_ITEMS)
# ─────────────────────────────────────────────────────────────────────────────


QUESTION_ANGLES = [
    "Identify the PRIMARY positional contrast — which side or zone of the room "
    "the largest or most dominant piece occupies in each layout. "
    "Ask the user which position they prefer for that dominant piece.",

    "Identify how the storage or secondary pieces relate spatially to the main piece "
    "in each layout — same side, adjacent, or on a completely different wall. "
    "Ask the user which proximity arrangement they prefer.",

    "Identify whether items are grouped tightly together on one side "
    "or distributed across multiple walls creating distinct activity zones. "
    "Ask the user which arrangement style they prefer.",

    "Identify where the largest open floor area falls in each layout — "
    "center of the room, one specific side, or distributed evenly. "
    "Ask the user where they want the open space to be.",

    "Identify the overall spatial feel contrast — one layout may feel "
    "structured and balanced while the other feels dynamic and asymmetric, "
    "or one sparse and the other dense. "
    "Ask the user which spatial feel they prefer. Do NOT name specific furniture items.",
]

N_ANGLES = len(QUESTION_ANGLES)


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
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.layout_ids = []
        self.dim = None
        self.embedding_matrix = None
        self.axes = None
        self.axis_scores = None
        self._index_path = "layout_index.faiss"
        self._meta_path  = "layout_meta.json"

    def _compute_fingerprint(self, layouts):
        content = json.dumps(
            [{"id": l["id"], "description": l["description"]} for l in layouts],
            sort_keys=True
        )
        return hashlib.md5(content.encode()).hexdigest()

    def build_index(self, layouts):
        descriptions    = [l['description'] for l in layouts]
        self.layout_ids = [l['id'] for l in layouts]

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
                "descriptions": {l['id']: l['description'] for l in layouts},
                "embeddings":   self.embedding_matrix.tolist(),
                "axes":         self.axes.tolist(),
                "axis_scores":  self.axis_scores.tolist()
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

    def get_axis_contrast_pairs(self, axis_idx, n_pairs):
        projections    = self.axis_scores[:, axis_idx]
        sorted_indices = list(np.argsort(projections)[::-1])
        N              = len(sorted_indices)

        pairs = []
        seen  = set()
        for hi in range(N):
            for lo in range(N - 1, hi, -1):
                pair = frozenset({sorted_indices[hi], sorted_indices[lo]})
                if pair not in seen:
                    seen.add(pair)
                    pairs.append((
                        LAYOUTS[sorted_indices[hi]]['description'],
                        LAYOUTS[sorted_indices[lo]]['description']
                    ))
                if len(pairs) == n_pairs:
                    return pairs

        while len(pairs) < n_pairs:
            pairs.append(pairs[0])
        return pairs

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
# 3. EMBEDDING-SPACE UCB BANDIT
# ==========================================
class EmbeddingUCBBandit:
    def __init__(self, num_axes):
        self.num_arms     = num_axes
        self.counts       = [0] * num_axes
        self.values       = [0.0] * num_axes
        self.total_pulls  = 0
        self.question_ptr = [0] * num_axes

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

    def get_question(self, arm, question_bank):
        bank     = question_bank[str(arm)]
        ptr      = self.question_ptr[arm] % len(bank)
        question = bank[ptr]
        self.question_ptr[arm] += 1
        return question

    def rewind_question_ptr(self, arm):
        self.question_ptr[arm] = max(0, self.question_ptr[arm] - 1)

    def status(self):
        print("\n[Bandit] Arm status:")
        for i in range(self.num_arms):
            bar = "█" * int(self.values[i] * 200)
            print(f"  Axis {i}: value={self.values[i]:.4f}  pulls={self.counts[i]}  {bar}")


# ==========================================
# 4. THE LLM AGENT
# ==========================================
class QwenAgent:
    def __init__(self):
        print("[SYSTEM] Loading Qwen 7B...")
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype="auto"
        )
        print("[SYSTEM] Model loaded!\n")

    def _generate(self, system_prompt, user_prompt, max_tokens=250, temperature=0.2):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}]
        text   = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=temperature,
            do_sample=(temperature > 0), pad_token_id=self.tokenizer.eos_token_id
        )
        out_ids = [o[len(i):] for i, o in zip(inputs.input_ids, output)]
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

    def generate_question_bank(self, axis_idx, pairs, already_asked):
        """
        Generates N_ANGLES questions for one axis.
        The LLM reads the actual layout descriptions and picks whichever
        furniture items are the most differentiating — the angles never
        prescribe specific item names.
        already_asked is mutated in-place so every subsequent call sees
        all previously generated questions.
        """
        questions = []
        for (high_desc, low_desc), angle in zip(pairs, QUESTION_ANGLES):

            is_feel_angle = angle.strip().startswith(
                "Identify the overall spatial feel contrast"
            )

            furniture_rule = (
                "Do NOT name specific furniture items — focus on spatial feel only."
                if is_feel_angle else
                f"You MAY name whichever furniture items from the room ({FURNITURE_LABEL}) "
                f"are the KEY differentiating factor in the descriptions. "
                f"Let the descriptions guide which items to mention — do not force-fit any item."
            )

            if already_asked:
                avoid_block = (
                    "ALREADY ASKED — do NOT ask about the same spatial dimension, "
                    "zone, or furniture placement as any of these questions:\n"
                    + "\n".join(f"  - {q}" for q in already_asked)
                    + "\n\n"
                )
            else:
                avoid_block = ""

            system = (
                "You are a room layout preference expert helping a user choose between layouts.\n"
                "Your job: read two layout descriptions that represent the EXTREMES of one "
                "spatial axis, find their KEY contrast, then write ONE natural question "
                "that captures that contrast.\n\n"
                f"{avoid_block}"
                "Hard rules:\n"
                f"1. {furniture_rule}\n"
                "2. Ground the question in ACTUAL positions from the descriptions "
                "   (e.g. 'left wall', 'top-right corner', 'center', 'bottom wall', "
                "   'one side of the room', 'spread across walls').\n"
                "3. The question must be answerable with a clear A-or-B preference.\n"
                "4. Maximum 40 words. No preamble, no explanation. Output the question only."
            )
            user = (
                f"Framing angle:\n{angle}\n\n"
                f"High-extreme layout:\n{high_desc}\n\n"
                f"Low-extreme layout:\n{low_desc}\n\n"
                "Question:"
            )
            q = self._generate(system, user, max_tokens=70, temperature=0.3)
            questions.append(q)
            already_asked.append(q)

            print(f"    [{'feel' if is_feel_angle else 'positional'} | {angle[:50]}...]")
            print(f"    → {q}")

        return questions

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
            "5. If gibberish or completely off-topic → output exactly: __GIBBERISH__\n"
            "6. If no preference / uncertain / indifferent → output exactly: __NEUTRAL__\n"
            "7. ONE sentence only. No extra text.\n"
            "Examples:\n"
            "  Q: 'Where do you want the bed?' / A: 'top left corner' "
            "  → 'The bed is positioned in the top-left corner of the room.'\n"
            "  Q: 'Sofa on same side as bed or opposite wall?' / A: 'opposite' "
            "  → 'The sofa is placed on the opposite wall from the bed.'\n"
            "  Q: 'Open space in center or on one side?' / A: 'middle open' "
            "  → 'The center of the room is kept open with furniture arranged around the perimeter.'"
        )
        user   = f'Question: "{question}"\nUser answered: "{user_input}"\nSpatial preference sentence:'
        result = self._generate(system, user, max_tokens=80, temperature=0.1)
        if GIBBERISH_SENTINEL in result:
            return GIBBERISH_SENTINEL
        if NEUTRAL_SENTINEL in result:
            return NEUTRAL_SENTINEL
        return result


# ==========================================
# 5. QUESTION BANK — save / load / build
# ==========================================
def _question_bank_fingerprint(layouts, num_axes):
    content = json.dumps({
        "layouts":         [{"id": l["id"], "description": l["description"]} for l in layouts],
        "num_axes":        num_axes,
        "angles":          QUESTION_ANGLES,
        "furniture_items": FURNITURE_ITEMS,
    }, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def load_question_bank(layouts, num_axes):
    if not os.path.exists(QUESTION_BANK_PATH):
        return None
    with open(QUESTION_BANK_PATH) as f:
        data = json.load(f)
    if data.get("fingerprint") != _question_bank_fingerprint(layouts, num_axes):
        print("[Setup] Question bank outdated — rebuilding...")
        return None
    print(f"[Setup] Question bank loaded from {QUESTION_BANK_PATH}")
    return data["bank"]


def save_question_bank(bank, layouts, num_axes):
    with open(QUESTION_BANK_PATH, "w") as f:
        json.dump({
            "fingerprint":     _question_bank_fingerprint(layouts, num_axes),
            "furniture_items": FURNITURE_ITEMS,
            "angles":          QUESTION_ANGLES,
            "bank":            bank
        }, f, indent=2)
    print(f"[Setup] Question bank saved to {QUESTION_BANK_PATH}")


def build_question_bank(llm, vector_store, num_axes):
    bank          = {}
    already_asked = []

    for axis_idx in range(num_axes):
        print(f"\n  [Axis {axis_idx}] Generating {N_ANGLES} questions "
              f"({len(already_asked)} already-asked questions in context)...")
        pairs     = vector_store.get_axis_contrast_pairs(axis_idx, n_pairs=N_ANGLES)
        questions = llm.generate_question_bank(axis_idx, pairs, already_asked)
        random.shuffle(questions)
        bank[str(axis_idx)] = questions

    return bank


# ==========================================
# 6. LAYOUT VALIDATION
# ==========================================
def validate_layouts(layouts, furniture_items):
    errors = []
    for layout in layouts:
        missing = [item for item in furniture_items if item not in layout]
        if missing:
            errors.append(f"  {layout['id']} missing: {missing}")
    if errors:
        raise ValueError(
            f"Layout validation failed — the following layouts are missing items "
            f"from FURNITURE_ITEMS {furniture_items}:\n" + "\n".join(errors)
        )
    print(f"[Validation] All {len(layouts)} layouts contain: {furniture_items} ✅\n")


# ==========================================
# 7. DELTA-BASED REWARD
# ==========================================
def apply_balanced_reward(layouts, similarity_scores):
    scores   = [similarity_scores.get(l['id'], 0.0) for l in layouts]
    mean_sim = float(np.mean(scores))
    std_sim  = float(np.std(scores))
    print(f"   [Stats] mean={mean_sim:.4f}  std={std_sim:.4f}")
    for layout in layouts:
        sim   = similarity_scores.get(layout['id'], 0.0)
        delta = sim - mean_sim
        layout['score'] += delta
        tag = f"Δ={delta:+.4f} ✅" if delta > 0 else f"Δ={delta:+.4f} ❌"
        print(f"   [{layout['id']:10s}] score={layout['score']:+.4f}  sim={sim:.4f}  {tag}")


# ==========================================
# 8. ORCHESTRATOR
# ==========================================
def main():
    for layout in LAYOUTS:
        layout['score'] = 0.0

    validate_layouts(LAYOUTS, FURNITURE_ITEMS)

    llm          = QwenAgent()
    vector_store = LayoutVectorStore()

    print("=" * 45)
    print(" PHASE 1: SETUP")
    print("=" * 45)
    print(f"[Config] Furniture items: {FURNITURE_LABEL}")
    print(f"[Config] Layouts: {len(LAYOUTS)}  |  Angles per axis: {N_ANGLES}\n")

    if not vector_store.load_index(LAYOUTS):
        print("[SYSTEM] Building FAISS index from scratch...")
        vector_store.build_index(LAYOUTS)

    num_axes = vector_store.axes.shape[0]

    question_bank = load_question_bank(LAYOUTS, num_axes)
    if question_bank is None:
        print("[Setup] Building question bank (one-time LLM cost)...")
        question_bank = build_question_bank(llm, vector_store, num_axes)
        save_question_bank(question_bank, LAYOUTS, num_axes)

    print("\n[Setup] Question bank contents:")
    for axis_idx in range(num_axes):
        print(f"  Axis {axis_idx}:")
        for i, q in enumerate(question_bank[str(axis_idx)]):
            print(f"    Q{i+1}: {q}")

    bandit = EmbeddingUCBBandit(num_axes)
    print(f"\n[Bandit] Initialized: {num_axes} axes × {N_ANGLES} questions each\n")

    print("=" * 45)
    print(" PHASE 2: USER SESSION")
    print("=" * 45)

    MAX_TURNS      = 20
    turns_consumed = 0

    while turns_consumed < MAX_TURNS:
        print(f"\n--- Turn {turns_consumed + 1} of {MAX_TURNS} ---")

        axis_idx = bandit.select_arm()
        question = bandit.get_question(axis_idx, question_bank)

        pull_num = bandit.counts[axis_idx] + 1
        q_num    = bandit.question_ptr[axis_idx]
        print(f"[System] Axis {axis_idx}  pull #{pull_num}  Q{q_num}/{N_ANGLES}")
        print(f"Chatbot: {question}")
        user_input = input("    You: ").strip()

        if not user_input:
            print("[System] Empty input — please type a response.")
            bandit.rewind_question_ptr(axis_idx)
            continue

        preference_text = llm.extract_preference_statement(question, user_input)
        print(f"[System] Preference → \"{preference_text}\"")

        if preference_text == GIBBERISH_SENTINEL:
            print("[System] ⚠ Input not meaningful — please answer the question.")
            bandit.rewind_question_ptr(axis_idx)
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