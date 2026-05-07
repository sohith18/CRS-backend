"""
Debug script — mirrors summarise_cluster() from crs8_nothink.py exactly.
Run this to verify qwen3.6:35b cluster summaries work correctly.

FIX: Use think=False as a top-level client.chat() param instead of
     prepending /no_think to the prompt (which causes empty output).
"""
import re
import random
from ollama import Client


client = Client(host="http://localhost:11434")
MODEL  = "qwen3.6:35b"
DEBUG  = True


def debug(label: str, value: str):
    border = "─" * 60
    print(f"\n🔍 [{label}]\n{border}\n{value}\n{border}")


# ── _generate — think=False passed as top-level param (Ollama native) ──
def _generate(prompt: str, max_new_tokens: int = 500, temperature: float = 0.2) -> str:
    debug("PROMPT SENT", prompt)

    response = client.chat(
        model=MODEL,
        think=False,                          # ✅ correct way to disable thinking
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
    debug("RAW MODEL OUTPUT", raw)

    # Strip any residual <think>...</think> blocks just in case
    stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    debug("AFTER <think> STRIP", repr(stripped))

    stripped = re.sub(r"^[\n\r\s]+", "", stripped).strip()
    debug("AFTER LEADING WHITESPACE STRIP", repr(stripped))

    return stripped


# ── _first_line ────────────────────────────────────────────────
def _first_line(text: str) -> str:
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            return line
    return ""


# ── summarise_cluster — mirrors crs8_nothink.py exactly ────────
def summarise_cluster(descriptions: list[str], n_sample: int = 10) -> str:
    sample = random.sample(descriptions, min(n_sample, len(descriptions)))

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

    raw_result = _generate(prompt, max_new_tokens=500, temperature=0.2)
    first = _first_line(raw_result)
    debug("_first_line RESULT", repr(first))
    return first


# ── Test with dummy layout descriptions ───────────────────────
if __name__ == "__main__":
    dummy_descriptions = [
        "A compact bedroom with the bed against the north wall, wardrobe on the left, and a small desk near the window.",
        "The bed is centred on the east wall with a wardrobe opposite and a desk tucked into the corner near the door.",
        "Open layout with the bed pushed to the far wall, wardrobe beside it, and a sofa facing the window.",
    ]

    print("=" * 60)
    print(f"🧪 Testing summarise_cluster with {MODEL}")
    print("=" * 60)

    summary = summarise_cluster(dummy_descriptions)

    print("\n" + "=" * 60)
    print(f"✅ Final summary returned: {repr(summary)}")
    print("=" * 60)