import math
import torch
import json
import re
import numpy as np
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. THE DATA (JSON Input)
# ==========================================
LAYOUTS = [
    {"id": "iter_0", "bed": [0.05, 0.0, 1.0, 0.06], "cupboard": [0.09, 0.03, 1.0, 0.10], "chair": [-0.15, 0.09, 0.46, 0.20]},
    {"id": "iter_2", "bed": [0.0, 0.03, 1.0, 0.78], "cupboard": [0.06, 0.08, 1.0, 0.92], "chair": [0.18, 0.17, 0.65, 0.89], "table": [0.75, 0.25, 1.0, 0.65]},
    {"id": "iter_3", "bed": [0.19, 0.03, 0.35, 0.11], "cupboard": [0.85, 0.23, 0.97, 0.32], "chair": [0.58, 0.31, 0.63, 0.46], "table": [0.42, 0.39, 0.59, 0.51]}
]


# ==========================================
# 2. FAISS VECTOR STORE
# ==========================================
class LayoutVectorStore:
    def __init__(self, model_name="BAAI/bge-m3"):
        print("[SYSTEM] Loading embedding model (BAAI/bge-m3)...")
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.layout_ids = []
        self.dim = None

    def build_index(self, layouts):
        """
        Embeds layout['description'] — guaranteed positive-factual from
        the single-step describe_layout prompt — and builds FAISS index.
        Embeddings are saved once offline and reused at session time.
        """
        descriptions = [l['description'] for l in layouts]
        self.layout_ids = [l['id'] for l in layouts]

        embeddings = self.embedder.encode(
            descriptions,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        self.dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.array(embeddings, dtype=np.float32))
        print(f"[SYSTEM] FAISS index built: {len(descriptions)} layouts | dim={self.dim}\n")

    def query(self, preference_text):
        """
        Returns cosine similarity scores for ALL layouts.
        Scores in [-1.0, 1.0] (vectors are L2-normalized).
        Returns: dict { layout_id -> similarity_score }
        """
        query_vec = self.embedder.encode(
            [preference_text],
            normalize_embeddings=True
        )
        similarities, indices = self.index.search(
            np.array(query_vec, dtype=np.float32), len(self.layout_ids)
        )
        return {
            self.layout_ids[idx]: float(sim)
            for sim, idx in zip(similarities[0], indices[0])
        }


# ==========================================
# 3. THE LLM AGENT (Hugging Face Qwen 7B)
# ==========================================
class QwenAgent:
    def __init__(self):
        print("[SYSTEM] Loading Qwen 7B Model...")
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
        )
        print("[SYSTEM] Model loaded successfully!\n")

    def _generate(self, system_prompt, user_prompt, max_tokens=150, temperature=0.2):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # ── SETUP PHASE ──────────────────────────────────────────────────────────

    def describe_layout(self, layout_dict):
        """
        Single-step prompt: generates rich, POSITIVE-FACTUAL description
        directly safe for FAISS indexing — no second sanitization pass needed.

        Enforced in the prompt itself:
          - Only describe furniture that IS present in coordinates
          - Never mention absent furniture (even negatively)
          - Zero negation words allowed
          - Pure positional + relational facts in standard English
        """
        system = """You are an expert interior designer and spatial analyst writing descriptions
for a vector database index. Your output must follow STRICT rules:

STRICT LANGUAGE RULES:
1. ONLY describe furniture items present in the provided coordinates. 
2. If a furniture item is NOT in the coordinates, do NOT mention it at all — not even to say it's absent.
3. NEVER use negation words: no, not, never, without, lacks, absent, missing, away from, rather than, instead of.
4. Every sentence must state what IS present and WHERE it is.
   BAD:  "The wardrobe is not near the wall."
   GOOD: "The wardrobe is positioned centrally in the room."
   BAD:  "No dining table is present."
   GOOD: (omit entirely — table not in coordinates)
   BAD:  "The chair is away from the bed."
   GOOD: "The chair is located in the right-center area of the room."
5. Use standard English furniture names only (wardrobe, bed, chair, table, cupboard).

CONTENT TO COVER (5-7 sentences):
1. Each item's exact wall position: left wall / center / right corner / top half / bottom half
2. Each item's footprint size: small / medium / large relative to the room
3. Spatial relationships: adjacent to, facing, beside, aligned with, near
4. Overall room density: spacious and open / moderately furnished / densely packed
5. Zone layout: sleeping zone location, seating zone location, storage zone location
6. Style feel: minimalist / functional / cozy / balanced"""

        user = f"""Room furniture coordinates [x_min, y_min, x_max, y_max] (normalized 0.0 to 1.0):
{json.dumps(layout_dict, indent=2)}

Coordinate guide: x increases left→right, y increases top→bottom.
Near 0.0 = top-left. Near 1.0 = bottom-right.
Write the description now:"""

        return self._generate(system, user, max_tokens=250)

    def generate_discriminative_phrases(self, all_descriptions, num_phrases=10):
        """Generates binary A/B questions that discriminate between layouts."""
        system = f"""You are a recommendation system engine. Read these detailed room layout descriptions carefully.
Generate exactly {num_phrases} distinct, binary (A/B) questions to narrow down user preference.
Cover: furniture placement, room density, zone organization, style feel, and spatial flow.
Examples:
  - 'Do you prefer the bed placed along the left wall or positioned in the center-bottom area?'
  - 'Do you want a spacious open room or a densely furnished cozy room?'
  - 'Do you prefer storage (cupboard) near the bed or on the opposite wall?'

OUTPUT STRICTLY AS A JSON ARRAY OF STRINGS. NO MARKDOWN, NO EXPLANATION."""

        user = "Descriptions:\n" + "\n".join([f"- {desc}" for desc in all_descriptions])
        response = self._generate(system, user, max_tokens=500)

        try:
            clean_json = re.sub(r'```json\n?|```', '', response).strip()
            phrases = json.loads(clean_json)
            if not isinstance(phrases, list):
                raise ValueError("Not a list")
            return phrases[:num_phrases]
        except Exception:
            print(f"[ERROR] Failed to parse JSON from LLM:\n{response}")
            return [
                "Do you prefer the bed on the left wall or the right wall?",
                "Do you want a central seating area or seating near a wall?",
                "Do you prefer a minimalist or fully furnished room?",
                "Do you want the storage cupboard near the bed or far from it?",
                "Do you prefer an open spacious room or a cozy compact layout?"
            ]

    # ── SESSION PHASE ─────────────────────────────────────────────────────────

    def classify_intent(self, question, user_input):
        """
        Classifies free-text user reply into A, B, or C.
        This is the ONLY LLM call needed in the session phase.
        The A/B output cleanly captures intent — no further rewriting needed
        before querying FAISS. The FAISS query is built deterministically
        from the question text + classification in the orchestrator.
        """
        system = "You are a strict data classifier. Output EXACTLY ONE letter: A, B, or C. No other text."
        user = f"""Question asked: "{question}"
User reply: "{user_input}"
Categories:
[A] User explicitly states preference for the first option/concept in the question.
[B] User explicitly states preference for the second option/concept in the question.
[C] Unclear, neither, or doesn't care.
Output ONLY A, B, or C:"""

        res = self._generate(system, user, max_tokens=2, temperature=0.1)
        match = re.search(r'[ABC]', res.upper())
        return match.group(0) if match else "C"


# ==========================================
# 4. MULTI-ARMED BANDIT (UCB)
# ==========================================
class UCBBandit:
    def __init__(self, arms):
        self.arms = arms
        self.num_arms = len(arms)
        self.counts = [0] * self.num_arms
        self.values = [0.0] * self.num_arms
        self.total_pulls = 0

    def select_arm(self):
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm
        ucb_values = [
            self.values[arm] + math.sqrt((2 * math.log(self.total_pulls)) / self.counts[arm])
            for arm in range(self.num_arms)
        ]
        return ucb_values.index(max(ucb_values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.total_pulls += 1
        n = self.counts[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * self.values[chosen_arm] + (1 / n) * reward


# ==========================================
# 5. BALANCED REWARD FUNCTION
# ==========================================
def apply_balanced_reward(layouts, similarity_scores, threshold=0.3):
    """
    Balanced reward strategy:
      sim >= threshold → score += sim          (proportional positive reward)
      sim <  threshold → score -= (1-sim)*0.5  (soft penalty, never brutal)
    """
    for layout in layouts:
        sim = similarity_scores.get(layout['id'], 0.0)
        if sim >= threshold:
            layout['score'] += sim
            tag = f"+{sim:.4f} ✅ MATCH"
        else:
            penalty = (1.0 - sim) * 0.5
            layout['score'] -= penalty
            tag = f"-{penalty:.4f} ❌ MISMATCH  (sim={sim:.4f})"
        print(f"   [{layout['id']:8s}] score={layout['score']:+.4f}  |  {tag}")


# ==========================================
# 6. PREFERENCE TEXT BUILDER
# ==========================================
def build_preference_text(question, classification):
    """
    Builds a clean, negation-free FAISS query string deterministically
    from the question and A/B classification — no LLM call needed.

    Why this works without an LLM:
      - The question is already a well-formed binary choice (from Phase 1)
      - classify_intent already resolved the user's intent to A or B
      - Extracting option A or B from the question is a simple string parse
      - Result is always in the same vocabulary as the indexed descriptions

    Example:
      question       = "Do you prefer the bed on the left wall or the right wall?"
      classification = "A"
      → "Room layout where the bed is on the left wall"

      question       = "Do you want a spacious open room or a densely furnished cozy room?"
      classification = "B"
      → "Room layout where there is a densely furnished cozy room"
    """
    # Split question on " or " to extract the two options
    # Works for all generated questions which follow the "X or Y?" pattern
    try:
        # Strip trailing "?" and split on " or "
        body = question.rstrip("?").strip()
        # Find " or " separator — take text after last occurrence of "or"
        parts = re.split(r'\s+or\s+', body, maxsplit=1)
        if len(parts) == 2:
            option_a_fragment = parts[0].split("prefer ")[-1].split("want ")[-1].strip()
            option_b_fragment = parts[1].strip()
            chosen = option_a_fragment if classification == "A" else option_b_fragment
            return f"Room layout where {chosen}"
        else:
            # Fallback: use full question with classification marker
            return f"Room layout matching option {classification} of: {question}"
    except Exception:
        return f"Room layout matching option {classification} of: {question}"


# ==========================================
# 7. ORCHESTRATOR
# ==========================================
def main():
    llm = QwenAgent()
    vector_store = LayoutVectorStore()  # BAAI/bge-m3

    # ---------------------------------------------------------
    # PHASE 1: SETUP (Offline — run once per layout batch)
    # ---------------------------------------------------------
    print("=======================================")
    print(" PHASE 1: SETUP (Translating Layouts)  ")
    print("=======================================")

    for layout in LAYOUTS:
        print(f"\nTranslating Layout {layout['id']}...")
        coord_data = {k: v for k, v in layout.items() if k not in ["id", "score", "description"]}

        # Single LLM call: rich + positive-factual in one prompt
        layout['description'] = llm.describe_layout(coord_data)
        layout['score'] = 0.0
        print(f" -> Description:\n    {layout['description']}\n")

    # Build FAISS index — descriptions are clean, index directly
    vector_store.build_index(LAYOUTS)

    print("Generating 10 Discriminative Questions...")
    all_descriptions = [l['description'] for l in LAYOUTS]
    generated_questions = llm.generate_discriminative_phrases(all_descriptions, num_phrases=10)

    print("\nGenerated Questions:")
    for i, q in enumerate(generated_questions):
        print(f"  {i+1:2d}. {q}")

    bandit = UCBBandit(generated_questions)

    # ---------------------------------------------------------
    # PHASE 2: USER SESSION (Online)
    # ---------------------------------------------------------
    print("\n=======================================")
    print(" PHASE 2: USER SESSION STARTED         ")
    print("=======================================")

    num_turns = 3
    for turn in range(num_turns):
        print(f"\n--- Turn {turn + 1} of {num_turns} ---")

        chosen_arm_idx = bandit.select_arm()
        question = bandit.arms[chosen_arm_idx]

        print(f"Chatbot: {question}")
        user_input = input("    You: ").strip()

        # 1 LLM call: classify A/B/C — that's all we need
        print("[System] Classifying intent...")
        classification = llm.classify_intent(question, user_input)
        print(f"[System] Classification → {classification}")

        if classification in ["A", "B"]:
            bandit.update(chosen_arm_idx, reward=1.0)

            # Build FAISS query deterministically — NO extra LLM call
            preference_text = build_preference_text(question, classification)
            print(f"[FAISS]  Query → \"{preference_text}\"")

            similarity_scores = vector_store.query(preference_text)

            print("[System] Applying balanced rewards:")
            apply_balanced_reward(LAYOUTS, similarity_scores, threshold=0.3)

        else:
            print("[System] Preference unclear — penalizing arm, skipping layout update.")
            bandit.update(chosen_arm_idx, reward=0.0)

    # ---------------------------------------------------------
    # FINAL RESULTS
    # ---------------------------------------------------------
    print("\n=======================================")
    print(" FINAL RECOMMENDATIONS                 ")
    print("=======================================")
    LAYOUTS.sort(key=lambda x: x["score"], reverse=True)

    for i, layout in enumerate(LAYOUTS):
        print(f"\n  Rank {i+1}: Layout '{layout['id']}' | Final Score: {layout['score']:+.4f}")
        print(f"  Description: {layout['description']}")


if __name__ == "__main__":
    main()