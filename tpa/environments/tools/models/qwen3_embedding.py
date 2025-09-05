import torch
from transformers import AutoTokenizer, AutoModel

class Qwen3Embedding:
    def __init__(self, ckpt_path="Qwen/Qwen3-Embedding-0.6B",device="cuda", dtype=torch.bfloat16):
        self.ckpt_path = ckpt_path
        self.device = device
        self.dtype = dtype
        self.model, self.tokenizer = self._load_model()
        self.model.to(self.device).to(self.dtype).eval()

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.ckpt_path, padding_side="left")
        model = AutoModel.from_pretrained(
            self.ckpt_path,
            torch_dtype=self.dtype
        )
        return model, tokenizer

    def _last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_text_embedding(self, text: str):
        """Embed a single text string."""
        batch_dict = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        batch_dict.to(self.device)
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embedding = self._last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        return embedding.to(self.dtype)

# ### Debug Code
# model = Qwen3Embedding()
# print(model.get_text_embedding("Hello, how are you?"))
