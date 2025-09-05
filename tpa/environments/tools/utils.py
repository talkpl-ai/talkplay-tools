"""Utility helpers for tools layer."""

def entity_str(metadata: dict):
    """Create a compact entity string from track metadata for text encoders.

    Args:
        metadata (dict): Item metadata with `track_name`, `artist_name`, `album_name`.

    Returns:
        str: A formatted description string.
    """
    track_name = metadata['track_name'][0].lower()
    artist_name = ", ".join(metadata['artist_name']).lower()
    album_name = ", ".join(metadata['album_name']).lower()
    entity_str = f"title: {track_name}, artist: {artist_name}, album: {album_name}"
    return entity_str
