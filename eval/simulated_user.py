#!/usr/bin/env python3
"""
simulated_user.py — Synthetic user personas for CRS evaluation.
Each persona has fixed preferences and answers VLM questions consistently.
"""

import re
import base64
import ollama

OLLAMA_HOST  = "http://localhost:11434"
VLM_MODEL_ID = "qwen2.5vl:3b"


class SimulatedUser:
    """
    A synthetic user with a fixed preference profile.
    Answers VLM-generated questions consistently based on their profile.
    5 personas covering distinct room preference archetypes.
    """

    PERSONAS = {
        # ── Persona 1: Open Space Lover ────────────────────────────────────
        "open_space_lover": {
            "description": (
                "Minimalist who values breathing room. Wants clear floor space, "
                "furniture pushed to walls, and an uncluttered center."
            ),
            "preferences": {
                "space_feel":       "open, airy, and uncluttered",
                "bed_position":     "against the far wall away from door to maximise floor space",
                "wardrobe_position":"along a side wall, not blocking pathways",
                "desk_position":    "near the window for natural light",
                "furniture_density":"minimal — fewer pieces, wide gaps between them",
                "traffic_flow":     "clear unobstructed walking path from door to bed",
                "focal_point":      "the window or a single statement wall, not furniture",
                "social_private":   "private retreat — calm and quiet atmosphere",
            },
        },

        # ── Persona 2: Cozy Corner ─────────────────────────────────────────
        "cozy_corner": {
            "description": (
                "Comfort-seeker who likes snug, filled rooms. Prefers furniture "
                "clustered together, cozy nooks, and a warm enclosed feel."
            ),
            "preferences": {
                "space_feel":       "cozy, snug, and filled — not too much empty floor",
                "bed_position":     "tucked into a corner with walls on two sides for security",
                "wardrobe_position":"close to the bed for easy morning access",
                "desk_position":    "in a corner to create a dedicated nook",
                "furniture_density":"high — furniture grouped close together for warmth",
                "traffic_flow":     "intimate — slight navigation around furniture is fine",
                "focal_point":      "the bed as the dominant centrepiece of the room",
                "social_private":   "private and retreat-like — personal sanctuary feel",
            },
        },

        # ── Persona 3: Functional Organiser ────────────────────────────────
        "functional_organiser": {
            "description": (
                "Pragmatic person who values clear zones and logical flow. "
                "Wants sleep, work, and storage areas clearly separated."
            ),
            "preferences": {
                "space_feel":       "organised and zoned — each area has a clear purpose",
                "bed_position":     "against the top wall opposite the door — clear sightline on entry",
                "wardrobe_position":"near the door for a logical get-dressed-and-leave flow",
                "desk_position":    "separated from the bed to keep work and sleep zones distinct",
                "furniture_density":"moderate — enough furniture for function, not excess",
                "traffic_flow":     "straight clear path from door to bed without detours",
                "focal_point":      "no single focal point — balanced zoning matters more",
                "social_private":   "practical and efficient — form follows function",
            },
        },

        # ── Persona 4: Social Entertainer ──────────────────────────────────
        "social_entertainer": {
            "description": (
                "Outgoing person who uses their room for more than just sleep. "
                "Wants open, welcoming layouts that feel social and spacious."
            ),
            "preferences": {
                "space_feel":       "spacious and welcoming — room for multiple people",
                "bed_position":     "against a side wall to leave the centre open",
                "wardrobe_position":"tucked away discreetly — storage should not dominate",
                "desk_position":    "facing the room, not a wall — to feel engaged",
                "furniture_density":"low-to-moderate — prioritise open floor area",
                "traffic_flow":     "wide easy movement paths — nothing blocking the centre",
                "focal_point":      "the centre of the room — open, inviting, unobstructed",
                "social_private":   "social and inviting — feels open to guests",
            },
        },

        # ── Persona 5: Symmetry Seeker ─────────────────────────────────────
        "symmetry_seeker": {
            "description": (
                "Design-conscious person who values visual balance and symmetry. "
                "Prefers formal, centred arrangements and paired furniture."
            ),
            "preferences": {
                "space_feel":       "balanced, formal, and visually harmonious",
                "bed_position":     "centred on the main wall with equal space on both sides",
                "wardrobe_position":"balanced — either symmetrically placed or hidden entirely",
                "desk_position":    "centred or directly opposite another piece for balance",
                "furniture_density":"moderate — every piece placed intentionally and symmetrically",
                "traffic_flow":     "equal paths on both sides of the main furniture",
                "focal_point":      "the bed centred as the clear symmetrical anchor",
                "social_private":   "formal and composed — elegant rather than casual",
            },
        },
    }

    def __init__(self, persona_name: str, model_id: str = VLM_MODEL_ID,
                 host: str = OLLAMA_HOST):
        if persona_name not in self.PERSONAS:
            raise ValueError(
                f"Unknown persona '{persona_name}'. "
                f"Choose from: {list(self.PERSONAS.keys())}"
            )
        self.persona_name = persona_name
        self.persona      = self.PERSONAS[persona_name]
        self.model_id     = model_id
        self.client       = ollama.Client(host=host)
        print(f"[SimUser] Loaded persona '{persona_name}': {self.persona['description']}")

    def answer(self, question: str, img_a_path: str, img_b_path: str) -> str:
        """
        Given a question and two layout images, produce a consistent natural-
        language answer in character with this persona's preference profile.
        """
        pref_block = "\n".join(
            f"  • {k.replace('_', ' ')}: {v}"
            for k, v in self.persona["preferences"].items()
        )

        prompt = (
            f"You are roleplaying as a real person looking for a room layout. "
            f"Your character description: {self.persona['description']}\n\n"
            f"Your fixed preferences:\n{pref_block}\n\n"
            f"You are shown two top-down room layout floor plans and asked:\n"
            f"\"{question}\"\n\n"
            "Answer IN CHARACTER as this person would naturally respond. Rules:\n"
            "  • Give a short natural answer (1 sentence, max 20 words)\n"
            "  • Be consistent with your preferences above\n"
            "  • Do NOT mention 'Layout A', 'Layout B', 'Image 1', 'Image 2'\n"
            "  • Do NOT explain your reasoning — just answer like a real person\n"
            "  • If the question uses 'open/airy vs cozy/filled', answer based on "
            "your space_feel preference\n"
            "  • If the question uses wall positions, answer based on your "
            "bed/wardrobe/desk position preferences\n"
            "  • If genuinely unsure (question is unclear), say 'I'm not sure' or "
            "'either works for me'\n\n"
            "Output ONLY the answer sentence. Nothing else."
        )

        with open(img_a_path, "rb") as fa, open(img_b_path, "rb") as fb:
            img_a_b64 = base64.b64encode(fa.read()).decode()
            img_b_b64 = base64.b64encode(fb.read()).decode()

        response = self.client.chat(
            model=self.model_id,
            think=False,
            messages=[{
                "role":    "user",
                "content": prompt,
                "images":  [img_a_b64, img_b_b64],
            }],
            options={
                "temperature": 0.2,
                "num_predict": 50,
                "top_p":       0.9,
            },
        )
        print(f"[SimUser] Raw response: {response['message']['content']}")
        raw = response["message"]["content"]
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*",          "", raw, flags=re.DOTALL)
        return raw.strip()

    @classmethod
    def list_personas(cls) -> list[str]:
        return list(cls.PERSONAS.keys())

    @classmethod
    def describe_all(cls):
        for name, data in cls.PERSONAS.items():
            print(f"  {name}: {data['description']}")


if __name__ == "__main__":
    print("Available personas:")
    SimulatedUser.describe_all()
