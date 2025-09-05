"""Item catalog access for TalkPlay environments.

Exposes `MusicCatalog` for reading track metadata and semantic id indices.
"""
import os
import torch
import json

class MusicCatalog:
    """Read-only access to track metadata and semantic ID indices.

    Args:
        cache_dir (str): Directory where metadata and semantic-id indices live.
        sid (str): Name/version of the semantic-id run (e.g., "rvq_r4_c64").
    """
    def __init__(self, cache_dir: str = "./cache", sid = "rvq_r4_c64"):
        self.cache_dir = cache_dir
        self.db = json.load(open(f"{cache_dir}/metadata/item_metadata.json", 'r'))
        self.track_pool = list(self.db.keys())
        self.sid_db = {}
        for modality in ["audio", "image", "metadata", "lyrics", "attributes", "cf_item"]:
            path = os.path.join(self.cache_dir, "semantic_id", sid, "indices", f"{modality}.pt")
            self.sid_db[modality] = torch.load(path, map_location="cpu")

    def id_to_semantic_id(self, track_id: str):
        """Get per-modality semantic-id indices for a track.

        Args:
            track_id (str): The track identifier.

        Returns:
            dict: Mapping of modality name to list of code indices for the track.
        """
        semantic_id = {}
        for modality in ["audio", "image", "metadata", "lyrics", "attributes", "cf_item"]:
            if track_id in self.sid_db[modality]:
                semantic_id[f"{modality}:semantic_id"] = self.sid_db[modality][track_id]
        return semantic_id

    def id_to_metadata(self, track_id: str, use_semantic_id: bool = False):
        """Format a compact metadata string for a given track.

        Args:
            track_id (str): The track identifier.
            use_semantic_id (bool): Whether to append semantic-id summary.

        Returns:
            str: A single-line entity description suitable for prompts.
        """
        metadata = self.db[track_id]
        track_id = metadata['track_id']
        track_name = metadata['track_name'][0].lower()
        artist_name = ", ".join(metadata['artist_name']).lower()
        album_name = ", ".join(metadata['album_name']).lower()
        release_date = metadata['track_release_date_spotify']
        tag_list = metadata['tag_list']
        semantic_id = self.id_to_semantic_id(track_id)
        entity_str = f"track_id: {track_id}, title: {track_name}, artist: {artist_name}, album: {album_name}"
        if len(tag_list) > 0:
            entity_str += f", tags: {', '.join(tag_list)}"
        if len(metadata['tempo']) > 0:
            tempo = float(metadata['tempo'][0])
            entity_str += f", tempo: {tempo}"
        if len(metadata['key']) > 0:
            key = metadata['key'][0]
            entity_str += f", key: {key}"
        if release_date:
            entity_str += f", release_date: {release_date}"
        if semantic_id and use_semantic_id:
            entity_str += f", {semantic_id}"
        return entity_str

# music_catalog = MusicCatalog()
# print(list(music_catalog.db.keys())[0:10])
# print(music_catalog.id_to_metadata(list(music_catalog.db.keys())[0]))
