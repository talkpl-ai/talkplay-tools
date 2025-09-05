import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

class SigLIP2:
    def __init__(self, ckpt_path="google/siglip2-base-patch16-224", device="cuda", dtype=torch.bfloat16):
        self.ckpt_path = ckpt_path
        self.device = device
        self.dtype = dtype
        self.model, self.processor = self._load_model()
        self.model.to(self.device).to(self.dtype).eval()

    def _load_model(self):
        model = AutoModel.from_pretrained(self.ckpt_path)
        processor = AutoProcessor.from_pretrained(self.ckpt_path, use_fast=True)
        return model, processor

    def get_image_embedding(self, image_path):
        """Embed a single image."""
        try:
            image = load_image(image_path)
        except:
            import pdb; pdb.set_trace()
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)
        return image_embeddings.to(self.dtype)

    def get_text_embedding(self, text):
        """Embed a single text."""
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs)
        return text_embeddings.to(self.dtype)

# model = SigLIP2()
# print(model.get_image_embedding("/workspace/dataset/spotify/images/0/0/00JYyme0U7qzgb1Bc7RbZY.jpg"))
# print(model.get_text_embedding("Hello, how are you?"))
