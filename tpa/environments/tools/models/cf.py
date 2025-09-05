import os
import numpy as np
import torch
import torch.nn as nn
from glob import glob
from tqdm import tqdm

class CollaborativeFiltering:
    def __init__(self, cache_dir="./cache", device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        if os.path.exists(f"{self.cache_dir}/encoder/models/cf_model.pt"):
            cf_model = torch.load(f"{self.cache_dir}/encoder/models/cf_model.pt")
            self.model = cf_model["model"]
            self.user_ids = cf_model["user_ids"]
            self.item_ids = cf_model["item_ids"]
        else:
            self.model, self.user_ids, self.item_ids = self._load_model()

    def _load_model(self):
        model = {
            "item": {},
            "user": {}
        }
        user_ids, item_ids = set([]), set([])
        item_files = glob(os.path.join("/workspace/dataset/spotify/embs", "cf_item", "**","*.npy"), recursive=True)
        user_files = glob(os.path.join("/workspace/dataset/spotify/embs", "cf_user", "*.npy"))
        for file in tqdm(item_files):
            item_id = os.path.basename(file).split(".")[0]
            embedding = np.load(file)
            model["item"][item_id] = torch.from_numpy(embedding)
            item_ids.add(item_id)
        for file in tqdm(user_files):
            user_id = os.path.basename(file).split(".")[0]
            embedding = np.load(file)
            model["user"][user_id] = torch.from_numpy(embedding)
            user_ids.add(user_id)
        os.makedirs(os.path.join(self.cache_dir, "encoder", "models"), exist_ok=True)
        torch.save({
            "model": model,
            "user_ids": user_ids,
            "item_ids": item_ids
        }, f"{self.cache_dir}/encoder/models/cf_model.pt")
        return model, user_ids, item_ids

    def get_item_embedding(self, item_id):
        if item_id not in self.item_ids:
            print(f"Item {item_id} is cold-start item case")
            return None
        item_emb = self.model["item"][item_id].to(self.device).to(self.dtype)
        if item_emb.ndim == 1:
            item_emb = item_emb.unsqueeze(0)
        return item_emb

    def get_user_embedding(self, user_id):
        if user_id not in self.user_ids:
            print(f"User {user_id} is cold-start user case")
            return None
        user_emb = self.model["user"][user_id].to(self.device).to(self.dtype)
        if user_emb.ndim == 1:
            user_emb = user_emb.unsqueeze(0)
        return user_emb
