import ollama
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_preference_questions(img1_path, img2_path, asked_questions=None):
    img1_b64 = encode_image(img1_path)
    img2_b64 = encode_image(img2_path)

    history_str = (
        "\n".join(f"- {q}" for q in asked_questions)
        if asked_questions else "None yet."
    )

    response = ollama.chat(
        model="qwen3-vl:8b",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Previously asked questions (DO NOT repeat these):\n{history_str}\n\n"
                    "These are two top-view 2D room layout images (Image 1 and Image 2).\n"
                    "Analyze the spatial differences in furniture placement \n\n"
                    "Generate 2-3 NEW short questions to ask the user to understand "
                    "which layout they prefer. Focus only on spatial/positional differences."
                ),
                "images": [img1_b64, img2_b64],   # ← pass both images here
            }
        ]
    )

    return response["message"]["content"]


# ── Usage ──────────────────────────────────────────────────────────────
asked = []

questions = generate_preference_questions(
    img1_path="layout1.png",
    img2_path="layout2.png",
    asked_questions=asked
)
print(questions)