"""Embedding-based retrieval utilities for text, audio, image, and CF modalities."""
import os
import json
import torch
import numpy as np
from tpa.environments.tools.models.qwen3_embedding import Qwen3Embedding
from tpa.environments.tools.models.clap import CLAP
from tpa.environments.tools.models.siglip2 import SigLIP2
from tpa.environments.tools.models.cf import CollaborativeFiltering
from tpa.environments.tools.utils import entity_str

class EmbeddingTool:
    """Load embedding models and vector DB, and perform similarity search."""
    def __init__(self, cache_dir="./cache", device="cuda", dtype=torch.bfloat16):
        self.cache_dir = cache_dir
        self.device = device
        self.dtype = dtype
        self.retriever = {
            "text": Qwen3Embedding(device=device, dtype=dtype), # 1024
            "audio": CLAP(device=device, dtype=dtype),  # 512
            "image": SigLIP2(device=device, dtype=dtype),  # 768
            "cf": CollaborativeFiltering(device=device, dtype=dtype) # 128
        }
        self.vector_db = self.load_model()
        self.modality_list = list(self.vector_db.keys())
        self.original_matrices, self.original_id_lists = self._build_matrices_from_vector_db()
        self._init_track_pool() # init

    def _init_track_pool(self):
        """Reset in-memory matrices and id lists to the full corpus."""
        self.current_matrices = self.original_matrices.copy()
        self.current_id_lists = self.original_id_lists.copy()

    def _update_track_pool(self, track_pool: list[str]):
        """Restrict search to a provided candidate set of track ids.

        Args:
            track_pool (list[str]): Allowed track ids.
        """
        current_index = {}
        for modality in self.modality_list:
            indices = []
            for track_id in track_pool:
                id_lists = self.original_id_lists[modality]
                if track_id in id_lists:
                    indices.append(id_lists.index(track_id))
            current_index[modality] = indices
        self.current_matrices = self.original_matrices.copy()
        self.current_id_lists = self.original_id_lists.copy()
        for modality in self.modality_list:
            self.current_matrices[modality] = self.original_matrices[modality][current_index[modality]]
            self.current_id_lists[modality] = [self.original_id_lists[modality][i] for i in current_index[modality]]

    def load_model(self):
        """Load vector database from disk.

        Returns:
            dict | None: Modality to id→embedding mapping, or None if missing.
        """
        path = os.path.join(self.cache_dir, "encoder", "vector_db.pt")
        if not os.path.exists(path):
            return None
        vector_db = torch.load(path, map_location="cpu")
        return vector_db

    def _build_matrices_from_vector_db(self):
        """Construct stacked matrices and id lists from the vector DB.

        Returns:
            tuple[dict, dict]: modality→matrix and modality→id_list mappings.
        """
        matrices = {}
        id_lists = {}
        for modality, id_to_vec in self.vector_db.items():
            ids, vecs = [], []
            for track_id, emb in id_to_vec.items():
                emb = emb.detach().cpu().squeeze()
                if emb.ndim != 1:
                    emb = emb.view(-1)
                ids.append(track_id)
                vecs.append(emb)
            id_lists[modality] = ids
            matrices[modality] = torch.stack(vecs, dim=0)
        return matrices, id_lists

    @torch.no_grad()
    def build_index(self):
        """Build vector DB from cached metadata and encoders, then persist to disk."""
        test_metadata = json.load(open(os.path.join(self.cache_dir, "metadata", "test_metadata.json"), "r"))
        metadata_db, lyrics_db, attributes_db, audio_db, image_db, cf_db = {}, {}, {}, {}, {}, {}
        for track_id, meta_info in test_metadata.items():
            audio_path = f"/workspace/dataset/spotify/audio/{track_id[0]}/{track_id[1]}/{track_id}.mp3"
            image_path = f"/workspace/dataset/spotify/images/{track_id[0]}/{track_id[1]}/{track_id}.jpg"
            metadata_db[track_id] = self.retriever["text"].get_text_embedding(entity_str(meta_info)).detach().cpu()
            lyrics_db[track_id] = self.retriever["text"].get_text_embedding(meta_info['lyrics']).detach().cpu()
            attributes_db[track_id] = self.retriever["text"].get_text_embedding(",".join(meta_info['tag_list'])).detach().cpu()
            audio_db[track_id] = self.retriever["audio"].get_audio_embedding(audio_path).detach().cpu()
            image_db[track_id] = self.retriever["image"].get_image_embedding(image_path).detach().cpu()
            cf_embedding = self.retriever["cf"].get_item_embedding(track_id)
            if cf_embedding is None:
                continue
            cf_db[track_id] = cf_embedding.detach().cpu()
        os.makedirs(os.path.join(self.cache_dir, "encoder"), exist_ok=True)
        vector_db = {
            "metadata": metadata_db,
            "lyrics": lyrics_db,
            "attributes": attributes_db,
            "audio": audio_db,
            "image": image_db,
            "cf": cf_db
        }
        torch.save(vector_db, os.path.join(self.cache_dir, "encoder", "vector_db.pt"))
        # Refresh in-memory structures
        self.vector_db = vector_db
        self._build_matrices_from_vector_db()

    def text_to_item_similarity(self, modality_type: str, corpus_type: str, query: str, topk: int):
        """Compute text-to-item similarity for the specified corpus.

        Args:
            modality_type (str): One of {"text", "audio", "image"}.
            corpus_type (str): One of {"metadata", "lyrics", "attributes", "audio", "image"}.
            query (str): Input text query.
            topk (int): Number of results.

        Returns:
            list[str]: Ranked track ids.
        """
        if corpus_type not in {"metadata", "lyrics", "attributes", "audio", "image"}:
            raise ValueError(f"Invalid corpus_type: {corpus_type}")
        if modality_type not in {"text", "audio", "image"}:
            raise ValueError(f"Invalid modality_type: {modality_type}")
        # enforce compatible pairs
        if corpus_type in {"metadata", "lyrics", "attributes"} and modality_type != "text":
            modality_type = "text"
        if modality_type == "audio" and corpus_type != "audio":
            corpus_type = "audio"
        if modality_type == "image" and corpus_type != "image":
            corpus_type = "image"
        mat = self.current_matrices.get(corpus_type)
        ids = self.current_id_lists.get(corpus_type, [])
        try:
            encoder = self.retriever[modality_type]
            q = encoder.get_text_embedding(query).detach().cpu().squeeze()
            k = min(topk, mat.size(0))
            scores = torch.matmul(mat, q.to(mat.dtype))
            top_indices = torch.topk(scores, k=k).indices.tolist()
            return [ids[i] for i in top_indices]
        except Exception as e:
            raise ValueError(f"Error in text_to_item_similarity: {e}")

    def item_to_item_similarity(self, modality_type: str, corpus_type:str, item_id: str, topk: int):
        """Compute item-to-item similarity within a corpus.

        Args:
            modality_type (str): One of {"audio", "image", "cf"}.
            corpus_type (str): One of {"audio", "image", "cf"}.
            item_id (str): Anchor track id.
            topk (int): Number of results.

        Returns:
            list[str]: Ranked track ids.
        """
        if modality_type not in {"audio", "image", "cf"}:
            raise ValueError(f"Invalid modality type: {modality_type}")
        if corpus_type not in {"audio", "image", "cf"}:
            raise ValueError(f"Invalid corpus_type for item similarity: {corpus_type}")
        mat = self.current_matrices.get(corpus_type)
        ids = self.current_id_lists.get(corpus_type, [])
        try:
            if item_id not in self.vector_db.get(modality_type, {}):
                raise ValueError(f"Item {item_id} not found in the database")
            item_emb = self.vector_db[modality_type][item_id].detach().cpu().squeeze()
            k = min(topk, mat.size(0))
            scores = torch.matmul(mat, item_emb.to(mat.dtype))
            top_indices = torch.topk(scores, k=k).indices.tolist()
            return [ids[i] for i in top_indices]
        except Exception as e:
            raise ValueError(f"Error in item_to_item_similarity: {e}")

    def user_to_item_similarity(self, user_id: str, topk: int):
        """Compute user-to-item similarity using the CF model.

        Args:
            user_id (str): User identifier.
            topk (int): Number of results.

        Returns:
            list[str]: Ranked track ids personalized for the user.
        """
        # Use CF model to get user embedding; DB contains item embeddings
        mat = self.current_matrices.get("cf")
        ids = self.current_id_lists.get("cf", [])
        try:
            user_emb = self.retriever["cf"].get_user_embedding(user_id)
            if user_emb is None:
                raise ValueError(f"User {user_id} not found in the database, cold start user")
            q = user_emb.detach().cpu().squeeze()
            k = min(topk, mat.size(0))
            scores = torch.matmul(mat, q.to(mat.dtype))
            top_indices = torch.topk(scores, k=k).indices.tolist()
            return [ids[i] for i in top_indices]
        except Exception as e:
            raise ValueError(f"Error in user_to_item_similarity: {e}")

# emb_tool = EmbeddingTool()
# print(emb_tool.text_to_item_similarity("text", "metadata", "cold as ice", 5))
