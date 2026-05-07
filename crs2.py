import math
import json
import re
import numpy as np
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


# LAYOUTS = [
#     {"id": "iter_0", "bed": [0.05, 0.0, 1.0, 0.06], "cupboard": [0.09, 0.03, 1.0, 0.10], "chair": [-0.15, 0.09, 0.46, 0.20]},
#     {"id": "iter_2", "bed": [0.0, 0.03, 1.0, 0.78], "cupboard": [0.06, 0.08, 1.0, 0.92], "chair": [0.18, 0.17, 0.65, 0.89], "table": [0.75, 0.25, 1.0, 0.65]},
#     {"id": "iter_3", "bed": [0.19, 0.03, 0.35, 0.11], "cupboard": [0.85, 0.23, 0.97, 0.32], "chair": [0.58, 0.31, 0.63, 0.46], "table": [0.42, 0.39, 0.59, 0.51]}
# ]

LAYOUTS = [
    # ── Minimalist top-left cluster ──────────────────────────────────────────
    {
        "id": "layout_01",
        "bed":      [0.02, 0.02, 0.40, 0.45],   # large, top-left corner
        "cupboard": [0.02, 0.50, 0.18, 0.85],   # tall, left wall bottom half
        "chair":    [0.45, 0.05, 0.58, 0.22],   # small, top-right area
    },

    # ── Bed centered, everything spread to walls ──────────────────────────────
    {
        "id": "layout_02",
        "bed":      [0.30, 0.35, 0.70, 0.75],   # large, dead center
        "cupboard": [0.82, 0.05, 0.98, 0.55],   # right wall, top half
        "table":    [0.05, 0.70, 0.25, 0.95],   # bottom-left corner
        "chair":    [0.78, 0.65, 0.92, 0.90],   # bottom-right corner
    },

    # ── Dense right-side cluster, open left half ──────────────────────────────
    {
        "id": "layout_03",
        "bed":      [0.55, 0.05, 0.98, 0.50],   # right wall, top half
        "cupboard": [0.60, 0.55, 0.78, 0.95],   # right-center, bottom half
        "table":    [0.80, 0.55, 0.98, 0.80],   # far right, bottom
        "chair":    [0.55, 0.55, 0.62, 0.75],   # clustered right side
    },

    # ── Bed along bottom wall, workspace top-right ────────────────────────────
    {
        "id": "layout_04",
        "bed":      [0.05, 0.60, 0.55, 0.98],   # large, bottom-left
        "cupboard": [0.60, 0.60, 0.78, 0.98],   # bottom-right storage
        "table":    [0.60, 0.05, 0.90, 0.30],   # top-right desk area
        "chair":    [0.62, 0.32, 0.75, 0.48],   # mid-right, facing desk
    },

    # ── Ultra sparse — only bed and cupboard, maximum open space ─────────────
    {
        "id": "layout_05",
        "bed":      [0.03, 0.03, 0.35, 0.38],   # small-ish, top-left
        "cupboard": [0.03, 0.62, 0.15, 0.97],   # narrow, left wall bottom
    },

    # ── Symmetric layout — mirrored left/right ────────────────────────────────
    {
        "id": "layout_06",
        "bed":      [0.30, 0.05, 0.70, 0.42],   # centered top half
        "cupboard": [0.02, 0.10, 0.22, 0.55],   # left wall
        "cupboard2":  [0.78, 0.10, 0.98, 0.55], # right wall (mirrored)
        "table":    [0.35, 0.60, 0.65, 0.85],   # centered bottom
        "chair":    [0.42, 0.87, 0.58, 0.98],   # center bottom edge
    },

    # ── L-shaped dense corner — bed + storage top, seating bottom-right ───────
    {
        "id": "layout_07",
        "bed":      [0.02, 0.02, 0.55, 0.42],   # wide, top-left
        "cupboard": [0.58, 0.02, 0.75, 0.42],   # top-right storage
        "table":    [0.58, 0.48, 0.80, 0.72],   # mid-right
        "chair":    [0.82, 0.48, 0.98, 0.72],   # far right, beside table
        "sofa":     [0.05, 0.70, 0.45, 0.98],   # bottom-left seating zone
    },

    # ── Diagonal flow — items arranged diagonally top-left to bottom-right ────
    {
        "id": "layout_08",
        "bed":      [0.02, 0.02, 0.38, 0.40],   # top-left
        "table":    [0.35, 0.30, 0.60, 0.58],   # center
        "chair":    [0.55, 0.55, 0.70, 0.75],   # center-right
        "cupboard": [0.68, 0.65, 0.85, 0.98],   # bottom-right
    },

    # ── Studio style — everything along walls, massive open center ────────────
    {
        "id": "layout_09",
        "bed":      [0.00, 0.02, 0.30, 0.45],   # left wall
        "cupboard": [0.00, 0.55, 0.18, 0.98],   # left wall bottom
        "table":    [0.82, 0.02, 0.98, 0.35],   # right wall top
        "chair":    [0.82, 0.40, 0.98, 0.60],   # right wall middle
        "sofa":     [0.30, 0.80, 0.75, 0.98],   # bottom wall center
    },

    # ── Cramped/cozy — many items, small gaps, high density ──────────────────
    {
        "id": "layout_10",
        "bed":      [0.02, 0.02, 0.48, 0.48],   # large, top-left quadrant
        "cupboard": [0.52, 0.02, 0.70, 0.48],   # top-right, beside bed
        "table":    [0.72, 0.02, 0.98, 0.35],   # top-far-right
        "chair":    [0.72, 0.38, 0.88, 0.58],   # mid-right
        "sofa":     [0.02, 0.55, 0.45, 0.80],   # bottom-left
        "shelf":    [0.50, 0.55, 0.98, 0.70],   # bottom-right shelf
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
        self.embedding_matrix = None  # (N × dim) — raw layout embeddings
        self.axes = None              # (K × dim) — SVD discriminative axes
        self.axis_scores = None       # projections of each layout onto each axis
        self._index_path = "layout_index.faiss"
        self._meta_path  = "layout_meta.json"

    def build_index(self, layouts):
        descriptions = [l['description'] for l in layouts]
        self.layout_ids = [l['id'] for l in layouts]

        embeddings = self.embedder.encode(
            descriptions, normalize_embeddings=True, show_progress_bar=True
        )
        self.dim = embeddings.shape[1]
        self.embedding_matrix = embeddings  # save raw for SVD

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.array(embeddings, dtype=np.float32))

        # Compute discriminative axes via SVD
        self._compute_axes()

        faiss.write_index(self.index, self._index_path)
        with open(self._meta_path, "w") as f:
            json.dump({
                "layout_ids":   self.layout_ids,
                "dim":          self.dim,
                "descriptions": {l['id']: l['description'] for l in layouts},
                "embeddings":   self.embedding_matrix.tolist(),
                "axes":         self.axes.tolist(),
                "axis_scores":  self.axis_scores.tolist()
            }, f)
        print(f"[SYSTEM] Index built. {len(descriptions)} layouts | {self.axes.shape[0]} axes\n")

    def load_index(self):
        import os
        if not (os.path.exists(self._index_path) and os.path.exists(self._meta_path)):
            return False
        self.index = faiss.read_index(self._index_path)
        with open(self._meta_path) as f:
            meta = json.load(f)
        self.layout_ids      = meta["layout_ids"]
        self.dim             = meta["dim"]
        self.embedding_matrix = np.array(meta["embeddings"], dtype=np.float32)
        self.axes            = np.array(meta["axes"],         dtype=np.float32)
        self.axis_scores     = np.array(meta["axis_scores"],  dtype=np.float32)
        for layout in LAYOUTS:
            if layout['id'] in meta["descriptions"]:
                layout['description'] = meta["descriptions"][layout['id']]
        print(f"[SYSTEM] Index loaded. {len(self.layout_ids)} layouts | {self.axes.shape[0]} axes\n")
        return True

    def _compute_axes(self, num_axes=None):
        """
        SVD on the centered embedding matrix.
        Each row of self.axes is a direction in embedding space
        that maximally separates the layouts.

        self.axis_scores[i, j] = projection of layout j onto axis i
        → tells us how each layout is positioned along each discriminative direction
        """
        E = self.embedding_matrix
        E_centered = E - E.mean(axis=0, keepdims=True)

        # SVD: E_centered = U @ diag(S) @ Vt
        # Vt rows = principal directions in embedding space
        _, S, Vt = np.linalg.svd(E_centered, full_matrices=False)

        # Keep top-K axes (default: min(N-1, 8) to avoid trivial axes)
        K = num_axes or min(len(self.layout_ids) - 1, 8)
        self.axes        = Vt[:K]                  # (K × dim)
        self.axis_scores = E_centered @ Vt[:K].T   # (N × K) — layout projections
        # axis_scores[j, i] = how far layout j is along axis i

        variances = (S[:K] ** 2) / (S ** 2).sum()
        print(f"[SVD] Top-{K} axes explain {variances.sum()*100:.1f}% of layout variance")
        for i in range(K):
            spread = self.axis_scores[:, i].max() - self.axis_scores[:, i].min()
            print(f"  Axis {i}: variance={variances[i]*100:.1f}%  spread={spread:.4f}")

    def get_axis_contrast_layouts(self, axis_idx):
        """
        Returns the two layouts most separated along the given axis.
        These become the "poles" for question generation.
        High projection → one style. Low projection → opposite style.
        """
        projections = self.axis_scores[:, axis_idx]
        high_idx = int(np.argmax(projections))
        low_idx  = int(np.argmin(projections))
        return (
            self.layout_ids[high_idx], LAYOUTS[high_idx]['description'],
            self.layout_ids[low_idx],  LAYOUTS[low_idx]['description']
        )

    def query(self, preference_text):
        query_vec = self.embedder.encode([preference_text], normalize_embeddings=True)
        similarities, indices = self.index.search(
            np.array(query_vec, dtype=np.float32), len(self.layout_ids)
        )
        return {
            self.layout_ids[idx]: float(sim)
            for sim, idx in zip(similarities[0], indices[0])
        }

    def compute_axis_reward(self, preference_text, axis_idx):
        """
        Reward = how much the user's preference vector aligns with this axis.

        If the user's answer projects strongly onto axis i, it means
        the question about axis i was highly relevant to them.
        High alignment → high reward → bandit learns to probe this
        type of dimension more for future users.

        This is the 'decoder' you imagined:
          embedding space → scalar reward signal per axis
        """
        pref_vec = self.embedder.encode([preference_text], normalize_embeddings=True)[0]
        axis_vec = self.axes[axis_idx]
        # Dot product of normalized vectors = cosine similarity to this axis
        alignment = float(np.dot(pref_vec, axis_vec))
        return abs(alignment)   # abs: both strong positive and negative signal = informative


# ==========================================
# 3. EMBEDDING-SPACE UCB BANDIT
#    Arms = SVD axes of the layout embedding space
#    Reward = how much user's preference aligned with that axis
# ==========================================
class EmbeddingUCBBandit:
    """
    Each arm i corresponds to a discriminative axis in embedding space.
    The bandit learns which geometric directions in the layout space
    are most relevant to this user's preferences.

    Axis 0 = most globally discriminative (highest SVD variance)
    Axis 1 = second most discriminative, orthogonal to axis 0
    ...

    After several sessions/users, the bandit learns:
      "Axis 0 (density) consistently gets high user engagement"
      "Axis 2 (table presence) rarely matters to users"
    → ask density questions first, skip table questions
    """
    def __init__(self, num_axes):
        self.num_arms    = num_axes
        self.counts      = [0] * num_axes
        self.values      = [0.0] * num_axes
        self.total_pulls = 0
        self.asked       = set()

    def select_arm(self):
        candidates = [i for i in range(self.num_arms) if i not in self.asked]
        if not candidates:
            self.asked.clear()
            candidates = list(range(self.num_arms))

        unexplored = [i for i in candidates if self.counts[i] == 0]
        if unexplored:
            # Start with axis 0 — highest SVD variance = most globally discriminative
            return unexplored[0]

        ucb = {
            i: self.values[i] + math.sqrt(2 * math.log(self.total_pulls) / self.counts[i])
            for i in candidates
        }
        return max(ucb, key=ucb.get)

    def update(self, arm, reward):
        self.asked.add(arm)
        self.counts[arm] += 1
        self.total_pulls += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) / n) * self.values[arm] + (1 / n) * reward

    def status(self):
        print("\n[Bandit] Arm status:")
        for i in range(self.num_arms):
            bar = "█" * int(self.values[i] * 20)
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
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=temperature,
            do_sample=(temperature > 0), pad_token_id=self.tokenizer.eos_token_id
        )
        out_ids = [o[len(i):] for i, o in zip(inputs.input_ids, output)]
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

    # ── SETUP PHASE ──────────────────────────────────────────────────────────

    def describe_layout(self, layout_dict):
        system = """You are an expert interior designer writing descriptions for a vector database.
STRICT RULES:
1. ONLY describe furniture present in the coordinates.
2. NEVER mention absent items. NEVER use negation words.
3. Use: left wall / center / right corner / top half / bottom half for positions.
4. Cover: item positions, sizes, spatial relationships, room density, zone layout, style feel.
5. Standard English furniture names only. 5-7 sentences."""
        user = f"""Coordinates [x_min, y_min, x_max, y_max] normalized 0.0-1.0:
{json.dumps(layout_dict, indent=2)}
x: left→right, y: top→bottom. Write description:"""
        return self._generate(system, user, max_tokens=250)

    # ── SESSION PHASE ─────────────────────────────────────────────────────────

    def generate_axis_question(self, high_desc, low_desc, axis_idx):
        """
        THE KEY DECODER: takes two layout descriptions at opposite poles
        of a discriminative axis and generates a natural question
        that probes that contrast.

        This is what makes questions dynamic and layout-count-independent:
          - With 3 layouts: generates 2 axis questions
          - With 100 layouts: generates 8 axis questions, each capturing
            richer, more nuanced contrasts derived from 100 descriptions
        """
        system = (
            "You are a preference elicitation expert. You will be shown two contrasting room layouts. "
            "Generate ONE concise, natural question that would help identify which style a user prefers. "
            "The question must be open-ended (not yes/no, not forced A/B). "
            "Focus on the KEY spatial difference between the two descriptions. "
            "Output the question only. No explanation."
        )
        user = f"""Layout Style A (one pole of contrast axis {axis_idx}):
{high_desc}

Layout Style B (opposite pole):
{low_desc}

What single open-ended question best captures the key preference difference between these two styles?"""
        return self._generate(system, user, max_tokens=60, temperature=0.3)

    def extract_preference_statement(self, question, user_input):
        """Converts free-text answer to positive, FAISS-ready preference statement."""
        system = (
            "You are a spatial preference interpreter for a vector database. "
            "Convert the user's answer into ONE positive, factual sentence about what they WANT. "
            "Rules: remove ALL negations. Translate regional terms to standard English "
            "(almara/almirah → wardrobe, takhat/palang → bed, kursi → chair). "
            "Output ONE sentence only."
        )
        user = f'Question: "{question}"\nUser answered: "{user_input}"\nPositive preference sentence:'
        return self._generate(system, user, max_tokens=80, temperature=0.1)


# ==========================================
# 5. BALANCED REWARD
# ==========================================
def apply_balanced_reward(layouts, similarity_scores, threshold=0.3):
    for layout in layouts:
        sim = similarity_scores.get(layout['id'], 0.0)
        if sim >= threshold:
            layout['score'] += sim
            tag = f"+{sim:.4f} ✅"
        else:
            penalty = (1.0 - sim) * 0.5
            layout['score'] -= penalty
            tag = f"-{penalty:.4f} ❌  (sim={sim:.4f})"
        print(f"   [{layout['id']:8s}] score={layout['score']:+.4f}  {tag}")


# ==========================================
# 6. ORCHESTRATOR
# ==========================================
def main():
    for layout in LAYOUTS:
        layout['score'] = 0.0

    llm = QwenAgent()
    vector_store = LayoutVectorStore()

    # PHASE 1: Setup (skipped if persisted)
    print("=" * 45)
    print(" PHASE 1: SETUP")
    print("=" * 45)

    if not vector_store.load_index():
        print("[SYSTEM] Building index from scratch...")
        for layout in LAYOUTS:
            coord_data = {k: v for k, v in layout.items() if k not in ["id", "score", "description"]}
            layout['description'] = llm.describe_layout(coord_data)
            print(f"[{layout['id']}] {layout['description'][:80]}...\n")
        vector_store.build_index(LAYOUTS)

    # Bandit arms = embedding-space SVD axes (NOT fixed questions)
    num_axes = vector_store.axes.shape[0]
    bandit = EmbeddingUCBBandit(num_axes)
    print(f"[Bandit] Initialized with {num_axes} embedding-space axes\n")

    # Cache generated questions per axis (generate once per axis per session)
    axis_questions = {}

    # PHASE 2: Session
    print("=" * 45)
    print(" PHASE 2: USER SESSION")
    print("=" * 45)

    num_turns = min(num_axes, 5)
    for turn in range(num_turns):
        print(f"\n--- Turn {turn + 1} of {num_turns} ---")

        # UCB picks which embedding axis to probe
        axis_idx = bandit.select_arm()

        # Generate question for this axis if not already cached
        if axis_idx not in axis_questions:
            high_id, high_desc, low_id, low_desc = vector_store.get_axis_contrast_layouts(axis_idx)
            print(f"[System] Generating question for Axis {axis_idx} "
                  f"(contrast: {high_id} ↔ {low_id})...")
            question = llm.generate_axis_question(high_desc, low_desc, axis_idx)
            axis_questions[axis_idx] = question
        else:
            question = axis_questions[axis_idx]

        print(f"Chatbot: {question}")
        user_input = input("    You: ").strip()

        if not user_input:
            print("[System] Empty input — skipping.")
            continue

        # Extract clean preference statement
        preference_text = llm.extract_preference_statement(question, user_input)
        print(f"[System] Preference → \"{preference_text}\"")

        # Query FAISS
        similarity_scores = vector_store.query(preference_text)

        # Reward = how much user's answer aligned with this embedding axis
        reward = vector_store.compute_axis_reward(preference_text, axis_idx)
        bandit.update(axis_idx, reward)
        print(f"[Bandit] Axis {axis_idx} reward (embedding alignment) = {reward:.4f}")

        # Update layout scores
        print("[System] Applying rewards:")
        apply_balanced_reward(LAYOUTS, similarity_scores, threshold=0.3)

    bandit.status()

    # Final results
    print("\n" + "=" * 45)
    print(" FINAL RECOMMENDATIONS")
    print("=" * 45)
    LAYOUTS.sort(key=lambda x: x["score"], reverse=True)
    for i, layout in enumerate(LAYOUTS):
        print(f"\n  Rank {i+1}: '{layout['id']}' | Score: {layout['score']:+.4f}")
        print(f"  {layout['description'][:120]}...")


if __name__ == "__main__":
    main()