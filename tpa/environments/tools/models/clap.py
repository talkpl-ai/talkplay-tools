import os
import torch
from tqdm import tqdm
import numpy as np


class CLAP:
    def __init__(self, cache_dir="./cache", device="cuda", dtype=torch.bfloat16):
        self.model = self._load_model(f"{cache_dir}/encoder/models/music_audioset_epoch_15_esc_90.14.pt")
        self.device = device
        self.dtype = dtype
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, ckpt_path):
        import laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", tmodel="roberta")
        model.load_ckpt(ckpt=ckpt_path, verbose=False)  # download
        return model

    def get_audio_embedding(self, audio_path):
        """Embed a batch of audio files"""
        with torch.no_grad():
            outputs = self.model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=True)
        return outputs.to(self.dtype)

    def get_text_embedding(self, text):
        """Embed a batch of text"""
        with torch.no_grad():
            outputs = self.model.get_text_embedding(x=[text], use_tensor=True)
        return outputs.to(self.dtype)

# ### Debug Code
# model = CLAP()
# modelity = "audio"
# path = f"/workspace/dataset/spotify/{modelity}/0/0/00JYyme0U7qzgb1Bc7RbZY.mp3"
# print(model.get_audio_embedding(path))
# print(model.get_text_embedding("Hello, how are you?"))
