"""
Agentic RAG — Room Layout Recommendation (VLM + DINOv3 Edition)
================================================================
Phase 1 (K-means Elimination):
  • Embed all layout IMAGES with DINOv3
  • Run K-means with dynamic K (silhouette, capped ≤ 10)
  • Qwen3-VL sees 2 representative images + history + user context
  • Generates spatial preference question from visual differences
  • User answers → VLM deduces which cluster to KEEP
  • Repeat until ≤ DIRECT_THRESHOLD (5) layouts remain

Final Recommendation:
  • All surviving layouts (≤ DIRECT_THRESHOLD) are shown to the user

Pair selection (Phase 1):
  • right_pool = ranked cluster B (centroid-closest first)
  • left_pool  = ranked cluster A (centroid-closest first)
  • On ambiguity: left_pool cycles; when exhausted, right_pool advances

Usage:
  python crs_vlm.py --layouts path/to/layouts.json
"""

import re, json, argparse, base64
import numpy as np
import ollama
from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import torch
from transformers import AutoImageProcessor, AutoModel


# ── Sentinels ──────────────────────────────────────────────────────────────────
KEEP_ALL_SENTINEL = "__KEEP_ALL__"
UNSURE_SENTINEL   = "__UNSURE__"

# ── Default Knobs ──────────────────────────────────────────────────────────────
DIRECT_THRESHOLD = 5
MAX_K            = 10
DINOV3_MODEL     = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
VLM_MODEL_ID     = "qwen3-vl:30b-a3b-instruct"


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

    print(f"[Loader] Loaded {len(layouts)} layouts\n")
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


# ══════════════════════════════════════════════════════════════════════════════
# 3. DINOV3 IMAGE EMBEDDER
# ══════════════════════════════════════════════════════════════════════════════
class DINOv3Embedder:

    def __init__(self, model_name: str = DINOV3_MODEL):
        print(f"[DINOv3] Loading model: {model_name}")
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype     = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name, dtype=self.dtype)
        self.model.eval().to(self.device)
        print(f"[DINOv3] Ready on {self.device} | dtype={self.dtype}\n")

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
        print(f"[VectorStore] Built: {len(self.active_layouts)} layouts | "
              f"dim={self.embeddings.shape[1]}\n")

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
        """All cluster members sorted closest-to-centroid first."""
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


class Qwen3VLAgent:

    def __init__(self, model_id: str = VLM_MODEL_ID):
        print(f"[VLM] Using ollama.chat() — model: {model_id}")
        self.model_id = model_id
        print("[VLM] Ready\n")

    def _generate(self, text: str, img1_path: str, img2_path: str,
                  max_tokens: int = 300, temperature: float = 0.3) -> str:
        img1_b64 = encode_image(img1_path)
        img2_b64 = encode_image(img2_path)

        response = ollama.chat(
            model=self.model_id,
            messages=[
                {
                    "role":    "user",
                    "content": text,
                    "images":  [img1_b64, img2_b64],
                }
            ],
            options={
                "temperature":    temperature,
                "num_predict":    max_tokens,
                "top_p":          0.9,
                "repeat_penalty": 1.1,
            },
        )
        print(response)
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

            "STEP 1 — Identify the single most visually obvious spatial difference between "
            "Layout A and Layout B. Focus on ONE of:\n"
            "  • Position of a major furniture piece relative to a fixed room feature "
            "(door, window)\n"
            "  • Furniture grouping or cluster arrangement\n"
            "  • Traffic flow or walkway direction through the room\n\n"

            "STEP 2 — Anchor all spatial language to a named reference object. "
            "NEVER use bare 'left' or 'right' without a reference "
            "(e.g., say 'left of the bed', 'beside the window', 'against the far wall', "
            "'near the door'). "
            "If you cannot name a reference object, describe by wall proximity instead "
            "(e.g., 'pushed against the top wall', 'centered in the room').\n\n"

            "STEP 3 — Write ONE preference question (max 25 words) that:\n"
            "  • Is a direct this-or-that choice (e.g., 'Do you prefer X or Y?')\n"
            "  • Uses only room-relative, anchored spatial terms from STEP 2\n"
            "  • Is about something a real person would notice and care about\n"
            "  • Does NOT mention 'Layout A', 'Layout B', 'Image 1', or 'Image 2'\n\n"

            f"Previously asked questions — DO NOT ask about the same spatial feature again:\n"
            f"{history_str}\n\n"
            f"Known user preferences (skip features already resolved):\n{ctx_text}\n\n"

            "Output ONLY the question text. No preamble, no numbering, no explanation."
        )

        result = self._first_line(
            self._generate(prompt, img1_path, img2_path, max_tokens=500, temperature=0.4)
        )
        return result

    def deduce_preferred_cluster(
        self,
        img1_path: str, img2_path: str,
        question: str, user_input: str, user_ctx: UserContext,
    ) -> str:
        prompt = (
            "These are two top-view 2D room layout images (Image 1 and Image 2).\n\n"
            f"Question asked to the user:\n{question}\n\n"
            f"User's response:\n{user_input}\n\n"
            f"User context:\n{user_ctx.as_prompt_block()}\n\n"
            "Determine which layout matches the user's answer.\n\n"
            "Rules:\n"
            "1. If the user's answer clearly and specifically matches one image → output that number.\n"
            "2. If the answer is vague, too short, or doesn't address the differentiating feature "
            "in the question → output 'both'.\n"
            "3. If the answer only confirms a feature SHARED by both images (not the difference "
            "being asked about) → output 'both'.\n"
            "4. Example of rule 3: question asks 'wardrobe on left or right?' and user says "
            "'back wall' — 'back wall' is not the differentiator, so output 'both'.\n"
            "Output ONLY: 1, 2, or both. Nothing else."
        )
        raw = self._first_line(
            self._generate(prompt, img1_path, img2_path, max_tokens=10, temperature=0.0)
        ).strip().lower()

        if re.search(r'\b1\b', raw) and not re.search(r'\b2\b', raw):
            return "1"
        elif re.search(r'\b2\b', raw) and not re.search(r'\b1\b', raw):
            return "2"
        else:
            return "both"

    def detect_unsure(self, img1_path: str, img2_path: str,
                      question: str, user_input: str) -> bool:
        prompt = (
            f"Question: {question}\n"
            f"User response: {user_input}\n\n"
            "Is the user uncertain or do they have a clear preference?\n"
            "Output exactly one word: UNSURE or CLEAR."
        )
        raw = self._first_line(
            self._generate(prompt, img1_path, img2_path, max_tokens=10, temperature=0.0)
        ).strip().upper()
        return "UNSURE" in raw

    def update_user_context(
        self,
        img1_path: str, img2_path: str,
        question: str, user_input: str,
        kept_img: str, dropped_img: str,
        user_ctx: UserContext,
    ):
        prompt = (
            "These are two top-view 2D room layout images (Image 1 and Image 2).\n\n"
            f"Question asked: {question}\n"
            f"User answered: {user_input}\n"
            f"Layout they preferred: {kept_img}\n"
            f"Layout they rejected: {dropped_img}\n\n"
            "Extract one LIKE and one DISLIKE (spatial preferences only).\n"
            "Output exactly two lines:\n"
            "LIKE: <short phrase max 10 words>\n"
            "DISLIKE: <short phrase max 10 words>\n"
            "Use 'nothing' if nothing is certain for a category."
        )
        raw = self._generate(prompt, img1_path, img2_path, max_tokens=100, temperature=0.1)
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
# 6. FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
def print_report(
    all_layouts:        list[dict],
    recommended:        list[dict],
    eliminated:         list[str],
    user_ctx:           UserContext,
):
    print("\n" + "=" * 55)
    print(" FINAL RECOMMENDATION")
    print("=" * 55)

    if user_ctx.likes or user_ctx.dislikes:
        print(f"\n[User Profile]\n{user_ctx.as_prompt_block()}")

    print(f"\n✅  Recommended layouts ({len(recommended)}):")
    for i, l in enumerate(recommended, 1):
        print(f"   {i}. [{l['id']}]  →  {l['image_path']}")

    if eliminated:
        print(f"\n🔴  Eliminated during Phase 1 ({len(eliminated)} layouts):")
        elim_set = set(eliminated)
        for l in all_layouts:
            if l["id"] in elim_set:
                print(f"   • {l['id']}  ({l['image_path']})")

    print(f"\n[Summary] {len(recommended)} recommended | "
          f"{len(eliminated)} eliminated | "
          f"{len(all_layouts)} total")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════════════════════
# 7. PAIR SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
class PairSelector:
    """
    Anchor-based pair selector for Phase 1.

      left_pool  = ranked cluster A members (centroid-closest first)
      right_pool = ranked cluster B members (centroid-closest first)

    Initial pair: (left_pool[0], right_pool[0])
    On ambiguity:
      anchor_step advances → next left_pool member, same right anchor
      When left_pool exhausted → pair_index advances → next right anchor, reset left
    On clear answer: anchor_step resets to 0
    """

    def __init__(self):
        self.pair_index  = 0
        self.anchor_step = 0

    def reset(self):
        self.pair_index  = 0
        self.anchor_step = 0

    def get_pair(
        self,
        left_pool:  list[dict],
        right_pool: list[dict],
    ) -> tuple[dict, dict]:
        r_idx = self.pair_index  % len(right_pool)
        l_idx = self.anchor_step % len(left_pool)
        return left_pool[l_idx], right_pool[r_idx]

    def advance_on_ambiguity(
        self,
        left_pool:  list[dict],
        right_pool: list[dict],
    ) -> str:
        """
        Rotate left member. If left_pool exhausted, advance right anchor.
        Returns 'left_rotated' or 'right_advanced'.
        """
        self.anchor_step += 1
        if self.anchor_step >= len(left_pool):
            self.anchor_step = 0
            self.pair_index += 1
            return "right_advanced"
        return "left_rotated"

    def on_clear_answer(self):
        self.anchor_step = 0


# ══════════════════════════════════════════════════════════════════════════════
# 8. CLI + ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Agentic RAG — Room Layout Recommendation (VLM + DINOv3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--layouts",          "-l", type=str, required=True)
    p.add_argument("--direct-threshold",       type=int, default=DIRECT_THRESHOLD)
    p.add_argument("--max-k",                  type=int, default=MAX_K)
    p.add_argument("--dinov3",                 type=str, default=DINOV3_MODEL)
    p.add_argument("--vlm",                    type=str, default=VLM_MODEL_ID)
    return p


def main():
    parser = build_arg_parser()
    args   = parser.parse_args()

    global DIRECT_THRESHOLD, MAX_K
    DIRECT_THRESHOLD = args.direct_threshold
    MAX_K            = args.max_k

    print(f"\n[Loader] Reading: {args.layouts}")
    all_layouts = load_layouts_from_json(args.layouts)

    if len(all_layouts) < 2:
        print("[Error] Need at least 2 layouts.")
        return

    embedder = DINOv3Embedder(model_name=args.dinov3)
    vlm      = Qwen3VLAgent(model_id=args.vlm)
    store    = LayoutVectorStore(embedder=embedder)
    user_ctx = UserContext()

    store.build(all_layouts)

    initial_active_ids = {l["id"] for l in all_layouts}
    turn             = 0
    asked_questions: list[str] = []
    selector         = PairSelector()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — K-means cluster elimination
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 55)
    print(" PHASE 1  —  Cluster-Based Visual Elimination")
    print("=" * 55)

    while store.count > DIRECT_THRESHOLD:
        turn += 1
        print(f"\n--- Turn {turn}  |  {store.count} layouts remaining ---")
        print("[Phase 1] Computing optimal K ...")

        k = store.compute_dynamic_k()
        clusters, km = store.cluster(k)

        print(f"[Phase 1] K={k} | Clusters: { {cid: len(cl) for cid, cl in clusters.items()} }")

        cids         = list(clusters.keys())
        cid_a, cid_b = cids[0], cids[1]

        left_pool  = store.get_ranked_members(clusters[cid_a], cid_a, km)
        right_pool = store.get_ranked_members(clusters[cid_b], cid_b, km)

        selector.reset()

        print(f"  Cluster {cid_a} pool ({len(left_pool)}): "
              + " ".join(l["id"] for l in left_pool))
        print(f"  Cluster {cid_b} pool ({len(right_pool)}): "
              + " ".join(l["id"] for l in right_pool))

        eliminated = False
        while not eliminated:
            rep_a, rep_b = selector.get_pair(left_pool, right_pool)
            img_a = rep_a["image_path"]
            img_b = rep_b["image_path"]

            print(f"\n[Phase 1] Comparing: '{rep_a['id']}' (cluster {cid_a}) "
                  f"vs '{rep_b['id']}' (cluster {cid_b})")
            if user_ctx.likes or user_ctx.dislikes:
                print(f"[Context] {user_ctx.as_prompt_block()}")

            question = vlm.generate_cluster_question(img_a, img_b, asked_questions, user_ctx)
            asked_questions.append(question)

            print(f"\nChatbot: {question}")
            raw_input_str = input("    You: ").strip()

            if not raw_input_str:
                print("[System] Empty input — please type an answer.")
                asked_questions.pop()
                continue

            is_unsure = vlm.detect_unsure(img_a, img_b, question, raw_input_str)
            if is_unsure:
                user_ctx.add_unsure(question[:60] + ("..." if len(question) > 60 else ""))
                print("[System] Got it — skipping that topic and trying something else.")
                move = selector.advance_on_ambiguity(left_pool, right_pool)
                next_a, next_b = selector.get_pair(left_pool, right_pool)
                print(f"[System] Rotating pair ({move}): "
                      f"next → {next_a['id']} vs {next_b['id']}")
                continue

            preferred = vlm.deduce_preferred_cluster(
                img_a, img_b, question, raw_input_str, user_ctx
            )
            print(f"[VLM] Preferred: Image {preferred}")

            if preferred == "both":
                move = selector.advance_on_ambiguity(left_pool, right_pool)
                next_a, next_b = selector.get_pair(left_pool, right_pool)
                print(f"[System] Ambiguous — rotating pair ({move}): "
                      f"next → {next_a['id']} vs {next_b['id']}")
                continue

            selector.on_clear_answer()

            eliminate_cid = cid_b if preferred == "1" else cid_a
            eliminate_ids = [l["id"] for l in clusters[eliminate_cid]]
            print(f"[System] Eliminating cluster {eliminate_cid} "
                  f"({len(eliminate_ids)} layouts)")

            vlm.update_user_context(
                img_a, img_b, question, raw_input_str,
                kept_img    = f"Image {'1' if preferred == '1' else '2'}",
                dropped_img = f"Image {'2' if preferred == '1' else '1'}",
                user_ctx    = user_ctx,
            )
            store.eliminate(eliminate_ids)
            eliminated = True

        if store.count == 0:
            print("[System] ⚠ All layouts eliminated — stopping.")
            return

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL RECOMMENDATION — all surviving layouts
    # ══════════════════════════════════════════════════════════════════════════
    surviving_ids = {l["id"] for l in store.active_layouts}
    eliminated    = [lid for lid in initial_active_ids if lid not in surviving_ids]

    print_report(
        all_layouts  = all_layouts,
        recommended  = store.active_layouts,
        eliminated   = eliminated,
        user_ctx     = user_ctx,
    )


if __name__ == "__main__":
    main()