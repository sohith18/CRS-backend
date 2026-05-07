import os
import math
import json
import hashlib
import re
import numpy as np
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


GIBBERISH_SENTINEL = "__GIBBERISH__"
NEUTRAL_SENTINEL   = "__NEUTRAL__"
QUESTION_BANK_PATH = "question_bank.json"


# Each angle is a framing lens — furniture-agnostic but grounded in
# spatial position, zone logic, AND feel/aesthetic. The LLM reads the
# actual layout descriptions and surfaces the real differentiating factor.
QUESTION_ANGLES = [
    # Positional / structural
    "Identify the PRIMARY spatial difference between the two layouts — which side of the room "
    "the main sleeping/resting area occupies (left, right, center, corner, wall-hugging). "
    "Ask the user where they prefer that area to be positioned.",

    "Identify how storage items are positioned relative to the sleeping area in each layout. "
    "Ask whether the user wants storage close to and on the same side as their sleeping area, "
    "or separated and on a different wall entirely.",

    "Identify whether furniture in these layouts is clustered in one zone leaving the rest open, "
    "or spread across multiple walls creating distinct zones in different parts of the room. "
    "Ask the user which arrangement they prefer.",

    "Identify whether the open floor space in these layouts sits in the CENTER of the room "
    "or is concentrated on ONE SIDE (left, right, top, bottom). "
    "Ask the user where they want their open space to be.",

    # Feel / aesthetic (but anchored to spatial consequence)
    "Identify the aesthetic contrast between these two layouts — one may feel ordered and "
    "symmetrical while the other feels asymmetric and dynamic, OR one sparse and the other full. "
    "Ask the user which spatial feel they prefer for their room.",
]

N_ANGLES = len(QUESTION_ANGLES)


# ==========================================
# 1. LAYOUTS WITH HANDCRAFTED DESCRIPTIONS
# ==========================================
LAYOUTS = [
    {
        "id": "layout_01",
        "bed":      [0.02, 0.02, 0.40, 0.45],
        "cupboard": [0.02, 0.50, 0.18, 0.85],
        "chair":    [0.45, 0.05, 0.58, 0.22],
        "description": (
            "The bed occupies the top-left corner of the room, spanning a large portion of the left "
            "wall from floor to mid-height. A tall, narrow cupboard sits against the left wall in "
            "the lower half, directly below the sleeping zone, serving as dedicated storage. "
            "A small chair is placed in the top-right area, creating a compact reading nook. "
            "All three items are clustered along the left and top edges, leaving the center and "
            "bottom-right of the room completely open. The layout is minimalist with very low "
            "density, a single sleeping zone top-left, a storage zone mid-left, and an isolated "
            "seating spot top-right. The overall style feels sparse and intentionally uncluttered."
        ),
    },

    {
        "id": "layout_02",
        "bed":      [0.30, 0.35, 0.70, 0.75],
        "cupboard": [0.82, 0.05, 0.98, 0.55],
        "table":    [0.05, 0.70, 0.25, 0.95],
        "chair":    [0.78, 0.65, 0.92, 0.90],
        "description": (
            "The bed is positioned dead center in the room, acting as the focal point with equal "
            "clearance on all sides. The cupboard stands tall against the right wall in the upper "
            "half, providing storage well away from the bed. A small dining table sits in the "
            "bottom-left corner, and a chair occupies the bottom-right corner facing inward. "
            "Furniture is spread symmetrically to the walls while the bed anchors the middle, "
            "creating four distinct corner zones around a central sleeping area. Room density is "
            "moderate with generous open space in the top-left quadrant. The style feels balanced "
            "and airy with clear functional separation between sleeping, storage, and dining zones."
        ),
    },

    {
        "id": "layout_03",
        "bed":      [0.55, 0.05, 0.98, 0.50],
        "cupboard": [0.60, 0.55, 0.78, 0.95],
        "table":    [0.80, 0.55, 0.98, 0.80],
        "chair":    [0.55, 0.55, 0.62, 0.75],
        "description": (
            "The bed dominates the right wall from top to mid-room, occupying roughly the top-right "
            "quadrant. The cupboard, table, and chair are all clustered in the right-center and "
            "bottom-right area, forming a dense right-side arrangement. The entire left half of "
            "the room is completely open and free of furniture. The chair sits at the left edge of "
            "this right cluster, beside the cupboard, creating a compact seating nook. The layout "
            "creates a strong asymmetry — a dense activity zone on the right and a large open "
            "movement zone on the left. Density is high on the right side, sparse on the left. "
            "The style is functional and zone-concentrated rather than distributed."
        ),
    },

    {
        "id": "layout_04",
        "bed":      [0.05, 0.60, 0.55, 0.98],
        "cupboard": [0.60, 0.60, 0.78, 0.98],
        "table":    [0.60, 0.05, 0.90, 0.30],
        "chair":    [0.62, 0.32, 0.75, 0.48],
        "description": (
            "The bed spans the entire bottom-left of the room, creating a large, dedicated sleeping "
            "zone anchored to the bottom and left walls. The top-right corner holds a wide desk "
            "table, forming a clear workspace zone well separated from the bed. A chair is "
            "positioned in the mid-right area, facing the desk, completing the work area. "
            "A cupboard stands against the bottom-right wall for storage, adjacent to the bed "
            "but in its own corner. The layout achieves strong zone separation — sleeping "
            "bottom-left, working top-right, storage bottom-right — with open space in the "
            "top-left. Density is moderate with each zone cleanly delineated. The style is "
            "highly functional and purposefully organized."
        ),
    },

    {
        "id": "layout_05",
        "bed":      [0.03, 0.03, 0.35, 0.38],
        "cupboard": [0.03, 0.62, 0.15, 0.97],
        "description": (
            "The room contains only two items — a medium bed in the top-left corner and a narrow "
            "cupboard against the left wall in the lower portion. Both items hug the left wall, "
            "leaving the entire right side and center of the room completely empty. The sleeping "
            "zone and storage zone are both on the left wall, vertically separated by a gap. "
            "The room is ultra-sparse with maximum open floor space throughout the center and "
            "right half. There is no seating, dining, or workspace area. The style is extremely "
            "minimalist, prioritizing open space above all other considerations."
        ),
    },

    {
        "id": "layout_06",
        "bed":       [0.30, 0.05, 0.70, 0.42],
        "cupboard":  [0.02, 0.10, 0.22, 0.55],
        "cupboard2": [0.78, 0.10, 0.98, 0.55],
        "table":     [0.35, 0.60, 0.65, 0.85],
        "chair":     [0.42, 0.87, 0.58, 0.98],
        "description": (
            "The bed is centered in the top half of the room, flanked symmetrically by a tall "
            "cupboard on the left wall and a matching cupboard on the right wall — creating a "
            "perfectly mirrored storage arrangement. A dining table sits centered in the lower "
            "half of the room directly below the bed, with a chair at its bottom edge facing "
            "inward. The layout is highly symmetric with a clear visual axis running top to "
            "bottom through the center. The sleeping zone is top-center, the dining zone is "
            "bottom-center, and storage flanks both sides equally. Density is moderate with "
            "balanced distribution. The style feels formal, ordered, and architecturally composed."
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
            "The bed occupies the wide top-left area, spanning over half the room width in the "
            "upper section. A cupboard sits in the top-right, adjacent to the bed, providing "
            "storage within the sleeping zone. A dining table and chair are paired in the "
            "mid-right area, forming a distinct workspace and dining zone. A large sofa anchors "
            "the bottom-left, creating a dedicated seating and living zone well separated from "
            "the bed. The layout has four clearly delineated zones — sleeping top-left, storage "
            "top-right, dining mid-right, seating bottom-left — with moderate-to-high density. "
            "The style is highly functional with strong zone separation and a multi-purpose feel."
        ),
    },

    {
        "id": "layout_08",
        "bed":      [0.02, 0.02, 0.38, 0.40],
        "table":    [0.35, 0.30, 0.60, 0.58],
        "chair":    [0.55, 0.55, 0.70, 0.75],
        "cupboard": [0.68, 0.65, 0.85, 0.98],
        "description": (
            "Furniture is arranged in a clear diagonal flow from the top-left to the bottom-right. "
            "The bed occupies the top-left corner as the starting point of the diagonal. A table "
            "sits in the center of the room, bridging the sleeping and seating areas. A chair is "
            "placed center-right, immediately adjacent to the table, forming a compact work or "
            "dining pair. The cupboard anchors the bottom-right corner, completing the diagonal. "
            "The top-right and bottom-left areas are open, creating triangular open zones on "
            "either side of the furniture diagonal. Density is low-to-moderate with a dynamic, "
            "flowing spatial feel. The style is modern and asymmetric with strong directional movement."
        ),
    },

    {
        "id": "layout_09",
        "bed":      [0.00, 0.02, 0.30, 0.45],
        "cupboard": [0.00, 0.55, 0.18, 0.98],
        "table":    [0.82, 0.02, 0.98, 0.35],
        "chair":    [0.82, 0.40, 0.98, 0.60],
        "sofa":     [0.30, 0.80, 0.75, 0.98],
        "description": (
            "All furniture is pushed against the perimeter walls, leaving the entire center of "
            "the room completely open. The bed and cupboard occupy the left wall — sleeping zone "
            "top-left, storage bottom-left. A desk table and chair are aligned against the right "
            "wall — workspace top-right and mid-right. A long sofa runs along the bottom wall "
            "centered between the two sides. The massive open center creates a studio-like "
            "movement space. Zones are clearly separated by wall position: left for sleeping and "
            "storage, right for work, bottom for seating. Density is low overall with high "
            "perimeter concentration. The style is studio-functional and maximally open."
        ),
    },

    {
        "id": "layout_10",
        "bed":      [0.02, 0.02, 0.48, 0.48],
        "cupboard": [0.52, 0.02, 0.70, 0.48],
        "table":    [0.72, 0.02, 0.98, 0.35],
        "chair":    [0.72, 0.38, 0.88, 0.58],
        "sofa":     [0.02, 0.55, 0.45, 0.80],
        "shelf":    [0.50, 0.55, 0.98, 0.70],
        "description": (
            "Six furniture items pack the room with high density. The bed occupies the entire "
            "top-left quadrant, with a cupboard immediately beside it top-right and a table in "
            "the far top-right corner. A chair sits mid-right adjacent to the table. A large sofa "
            "fills the bottom-left area, and a wide shelf spans the entire bottom-right half. "
            "All six zones are tightly arranged with minimal gaps between items. The room has "
            "sleeping top-left, storage top-center, workspace top-right, seating bottom-left, "
            "and shelving bottom-right as distinct but closely packed zones. The density is very "
            "high throughout. The style is cozy, maximally furnished, and apartment-efficient."
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

    def generate_question_bank(self, axis_idx, pairs):
        """
        Generates N_ANGLES questions per axis.
        Each question is grounded in the actual spatial difference between the two
        layout descriptions — but framed through the angle lens so the bank covers
        both positional (where things are) and aesthetic (how it feels) dimensions.

        The LLM is instructed to:
          1. Read both descriptions and find the KEY spatial contrast.
          2. Frame the question through the provided angle.
          3. Name actual positions (wall, corner, center, side) when relevant.
          4. Keep it furniture-count-agnostic — valid whether the room has 2 or 8 items.
          5. Stay under 35 words and avoid abstract-only phrasing.
        """
        questions = []
        for (high_desc, low_desc), angle in zip(pairs, QUESTION_ANGLES):
            system = (
                "You are a room layout preference expert. "
                "Your job: read two layout descriptions, find their KEY spatial contrast, "
                "then write ONE question that captures that contrast from the given angle.\n\n"
                "Hard rules:\n"
                "1. Ground the question in ACTUAL spatial positions from the descriptions "
                "   (e.g. 'left wall', 'center', 'bottom corner', 'one side of the room', "
                "   'spread across walls', 'clustered together').\n"
                "2. Do NOT name specific furniture types — the question must be valid for any room.\n"
                "3. The question must be answerable with a clear spatial or aesthetic preference.\n"
                "4. Blend positional AND feel language naturally — not one or the other exclusively.\n"
                "5. Maximum 35 words. No preamble, no explanation. Output the question only."
            )
            user = (
                f"Angle to frame the question through:\n{angle}\n\n"
                f"Layout A description:\n{high_desc}\n\n"
                f"Layout B description:\n{low_desc}\n\n"
                "Question (≤35 words, spatially grounded, furniture-type-agnostic):"
            )
            q = self._generate(system, user, max_tokens=60, temperature=0.3)
            questions.append(q)
            print(f"    [Angle: {angle[:50]}...]")
            print(f"    → {q}")
        return questions

    def extract_preference_statement(self, question, user_input):
        system = (
            "You are a spatial preference interpreter for a room layout vector database. "
            "Convert the user's answer into ONE positive, factual sentence describing "
            "what they want in their room layout. "
            "Rules:\n"
            "1. Describe the spatial arrangement they prefer — include wall position, "
            "   zone location, or open space location where relevant "
            "   (e.g. 'top-left', 'right wall', 'center', 'one side', 'bottom corner').\n"
            "2. Remove ALL negations — state what they DO want, not what they don't.\n"
            "3. Translate regional furniture terms to standard English "
            "   (almara/almirah → wardrobe, takhat/palang → bed, kursi → chair, "
            "   divan → sofa, meja → table).\n"
            "4. If the answer is gibberish, random characters, or completely unrelated "
            "   to room layout, output exactly: __GIBBERISH__\n"
            "5. If the user expresses no preference, uncertainty, or indifference "
            "   (e.g. 'not sure', 'idk', 'doesn't matter', 'whatever'), "
            "   output exactly: __NEUTRAL__\n"
            "6. ONE sentence only. No extra text.\n"
            "Examples:\n"
            "  Q: 'Where do you want your sleeping area?' / A: 'top left' "
            "  → 'The sleeping area is positioned in the top-left corner of the room.'\n"
            "  Q: 'Clustered on one side or spread across walls?' / A: 'spread out' "
            "  → 'Furniture is distributed across multiple walls with clear zones on each side.'\n"
            "  Q: 'Open space in center or on one side?' / A: 'I want the middle open' "
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
        "layouts":  [{"id": l["id"], "description": l["description"]} for l in layouts],
        "num_axes": num_axes,
        "angles":   QUESTION_ANGLES,
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
            "fingerprint": _question_bank_fingerprint(layouts, num_axes),
            "angles":      QUESTION_ANGLES,
            "bank":        bank
        }, f, indent=2)
    print(f"[Setup] Question bank saved to {QUESTION_BANK_PATH}")


def build_question_bank(llm, vector_store, num_axes):
    bank = {}
    for axis_idx in range(num_axes):
        print(f"\n  [Axis {axis_idx}] Generating {N_ANGLES} questions...")
        pairs               = vector_store.get_axis_contrast_pairs(axis_idx, n_pairs=N_ANGLES)
        bank[str(axis_idx)] = llm.generate_question_bank(axis_idx, pairs)
    return bank


# ==========================================
# 6. DELTA-BASED REWARD
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
# 7. ORCHESTRATOR
# ==========================================
def main():
    for layout in LAYOUTS:
        layout['score'] = 0.0

    llm          = QwenAgent()
    vector_store = LayoutVectorStore()

    print("=" * 45)
    print(" PHASE 1: SETUP")
    print("=" * 45)

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