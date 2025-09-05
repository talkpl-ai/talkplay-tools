"""Semantic ID matching tool for quantized embeddings across modalities."""
import os
import json
import torch
import numpy as np
from vector_quantize_pytorch import ResidualVQ
import polars as pl

class SemanticIDTool:
    """Perform semantic-ID matching using residual VQ code indices per modality."""
    def __init__(self, cache_dir="./cache", device="cuda", num_of_residual=4, num_of_cluster=64, dtype=torch.bfloat16):
        self.cache_dir = cache_dir
        self.device = device
        self.dtype = dtype
        self.num_of_residual = num_of_residual
        self.num_of_cluster = num_of_cluster
        # Initialize empty lists for each column
        track_ids = []
        modality_cols = {}
        sid_data = self.load_model()
        self.original_sid = {}
        for modality in ["audio", "image", "metadata", "lyrics", "attributes", "cf_item"]:
            track_ids = list(sid_data[modality].keys())
            modality_cols = {}
            for r in range(self.num_of_residual):
                col_name = f"r{r}"
                modality_cols[col_name] = [sid_data[modality][tid][r] for tid in track_ids]
            self.original_sid[modality] = pl.DataFrame({"track_id": track_ids, **modality_cols})
        self._init_track_pool()

    def _init_track_pool(self):
        """Reset current view to include all tracks."""
        self.current_sid = self.original_sid.copy()

    def _update_track_pool(self, track_pool: list):
        """Filter the current SID tables by a candidate track pool.

        Args:
            track_pool (list): Allowed track ids.
        """
        for modality in self.original_sid.keys():
            sid_db = self.original_sid[modality]
            sid_db = sid_db.filter(pl.col("track_id").is_in(track_pool))
            self.current_sid[modality] = sid_db

    def load_model(self):
        """Load quantized code indices from disk for each modality."""
        sid_db = {}
        for modality in ["audio", "image", "metadata", "lyrics", "attributes", "cf_item"]:
            path = os.path.join(self.cache_dir, "semantic_id", f"rvq_r{self.num_of_residual}_c{self.num_of_cluster}", "indices", f"{modality}.pt")
            sid_db[modality] = torch.load(path, map_location="cpu")
        return sid_db

    def semantic_id_matching(self, modality_type: str, indices: list[int], topk: int):
        """Retrieve items whose residual-VQ code indices match the query codes.

        Args:
            modality_type (str): One of {"audio", "image", "metadata", "lyrics", "attributes", "cf_item"}.
            indices (list[int]): Quantized code indices for each residual level.
            topk (int): Number of results.

        Returns:
            list[str]: Ranked track ids by match score.
        """
        try:
            sid_db = self.current_sid[modality_type]
            matches = []
            for r in range(self.num_of_residual):
                matches.append(pl.col(f"r{r}") == indices[r])
            match_expr = pl.sum_horizontal([m.cast(pl.Int8) for m in matches])
            results = sid_db.with_columns(match_score=match_expr).sort("match_score", descending=True)
            topk_results = results.select("track_id").head(topk).get_column("track_id").to_list()
        except Exception as e:
            raise ValueError(f"Error in semantic_id_matching: {e}")
        return topk_results
