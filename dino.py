import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# ── Load model & processor ─────────────────────────────────────────────
model_name = "facebook/dinov3-vith16plus-pretrain-lvd1689m"


processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)  # ← changed
model.eval()
model = model.to("cuda" if torch.cuda.is_available() else "cpu")


# ── Embed a single image ───────────────────────────────────────────────
def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token → global spatial-aware embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]   # shape: (1, 1280)
    return cls_embedding.squeeze().cpu().float().numpy()  # shape: (1280,)


# ── Embed multiple images (batch) ──────────────────────────────────────
def embed_images_batch(image_paths, batch_size=16):
    all_embeddings = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = processor(images=images, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (B, 1280)
        all_embeddings.append(cls_embeddings.cpu().float().numpy())

    return np.vstack(all_embeddings)  # (N, 1280)


# ── Dummy run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Single image
    emb = embed_image("layout1.png")
    print(f"Single embedding shape : {emb.shape}")        # (1280,)
    print(f"Embedding sample values: {emb[:5]}")

    # Batch of images
    image_paths = ["layout1.png", "layout2.png"]
    embeddings = embed_images_batch(image_paths, batch_size=8)
    print(f"Batch embeddings shape : {embeddings.shape}") # (2, 1280)