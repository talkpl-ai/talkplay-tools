"""User profile database access for TalkPlay environments.

Loads user profiles from cached metadata and exposes simple lookup helpers.
"""
import os
import json
import random
class UserProfileDB:
    """Lightweight interface to user profile metadata.

    Args:
        cache_dir (str): Root directory containing cached metadata files.
    """
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        self.user_profiles = json.load(open(os.path.join(self.cache_dir, "metadata", "user_profiles.json"), "r", encoding="utf-8"))

    def user_id_to_profile(self, user_id: str):
        """Return a normalized profile for a given user id.

        The returned dict includes a sampled list of recent tracks as
        `previous_history` when available.

        Args:
            user_id (str): Unique identifier of the user.

        Returns:
            dict: Profile fields and recent listening history.
        """
        user_profile = self.user_profiles[user_id]
        if len(user_profile['last_track_ids']) > 0:
            previous_history = random.sample(user_profile['last_track_ids'], 5)
        else:
            previous_history = []
        return {
            "user_id": user_id,
            "user_type": user_profile['user_type'],
            "age_group": user_profile['age_group'],
            "country_name": user_profile['country_name'],
            "gender": user_profile['gender'],
            "previous_history": previous_history
        }
