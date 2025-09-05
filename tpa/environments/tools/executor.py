"""Aggregates and executes retrieval tools for music recommendation.

Provides a single `ToolExecutor` class that manages SQL, BM25, embedding-based
similarity, and semantic-ID matching tools, and exposes them as callable
functions.
"""
import torch
from tpa.environments.tools.bm25 import BM25Tool
from tpa.environments.tools.sql import SQLTool
from tpa.environments.tools.embedding_sim import EmbeddingTool
from tpa.environments.tools.semantic_id import SemanticIDTool

class ToolExecutor:
    """
    A tool execution engine that manages and executes various music recommendation tools.

    This class provides a unified interface for executing different types of recommendation
    tools including SQL queries, BM25 text search, embedding-based similarity, and
    semantic ID matching. It manages tool instances and handles tool execution with
    proper error handling and track pool management.

    Args:
        cache_dir (str, optional): Directory path for caching tool data and models.
            Defaults to "./cache"
        device (str, optional): Device to run computations on (e.g., "cuda", "cpu").
            Defaults to "cuda"
        dtype (torch.dtype, optional): Data type for tensor operations.
            Defaults to torch.bfloat16
    """
    def __init__(self, cache_dir: str = "./cache", device="cuda", dtype=torch.bfloat16):
        self.cache_dir = cache_dir
        self.sql_tool = SQLTool()
        self.bm25_tool = BM25Tool()
        self.semantic_id_tool = SemanticIDTool()
        self.embedding_tool = EmbeddingTool()

        self.tools = {
            "sql": {
                "class": self.sql_tool,
                "function": self.sql,
                },
            "bm25": {
                "class": self.bm25_tool,
                "function": self.bm25,
            },
            "text_to_item_similarity": {
                "class": self.embedding_tool,
                "function": self.text_to_item_similarity,
            },
            "item_to_item_similarity": {
                "class": self.embedding_tool,
                "function": self.item_to_item_similarity,
            },
            "user_to_item_similarity": {
                "class": self.embedding_tool,
                "function": self.user_to_item_similarity,
            },
            "semantic_id_matching": {
                "class": self.semantic_id_tool,
                "function": self.semantic_id_matching,
            },
        }

    def execute(self, tool_name: str, tool_args: dict, track_pool: list[str]):
        """Execute a registered tool with arguments constrained to a track pool.

        Args:
            tool_name (str): Name of the tool to invoke (e.g., "sql", "bm25").
            tool_args (dict): Keyword arguments forwarded to the tool function.
            track_pool (list[str]): Candidate track ids to constrain retrieval.

        Returns:
            tuple[bool, list[str]]: Success flag and resulting list of track ids.
        """
        tool_call_success, tool_call_results = False, []
        tool_config = self.tools.get(tool_name, "")
        if tool_config == "": # error case 1: wrong tool name
            return tool_call_success, tool_call_results
        tool_class = tool_config["class"]
        tool_function = tool_config["function"]
        tool_class._update_track_pool(track_pool)
        tool_call_results = tool_function(**tool_args)
        if len(tool_call_results) == 0: # error case 2: tool call failed
            return tool_call_success, tool_call_results
        tool_call_success = True
        tool_class._init_track_pool() # == reset track pool
        return tool_call_success, tool_call_results

    def sql(self, sql_query: str, topk: int):
        """
        Execute an SQL query for boolean matching. Good for retrieval stage and filtering queries. Must retrun results with track_id (SELECT track_id FROM tracks WHERE ...).
        SQL Schema:
        track_id TEXT PRIMARY KEY, title TEXT, artist TEXT, album TEXT, popularity INTEGER, release_date DATETIME (YYYY-MM-DD), tempo REAL, key TEXT
        'title': {'description': 'title column is string data type, the examples are "Hello", "Shape of You", "Despacito"', 'type': 'string'},
        'artist': {'description': 'artist column is string data type, the examples are "Ed Sheeran", "Drake", "Ariana Grande"', 'type': 'string'},
        'album': {'description': 'album column is string data type, the examples are "Divide", "More Life", "DAMN."', 'type': 'string'},
        'popularity': {'description': 'popularity column is integer data type, the range is from 0 to 93, the average is 40.34', 'type': 'integer'},
        'release_date': {'description': 'release_date column is datetime data type, the range is from 1900-01-01 to 2017-11-10', 'type': 'integer'},
        'tempo': {'description': 'tempo column is integer data type, the range is from 41.21 to 208.31, the average is 105.41', 'type': 'integer'},
        'key': {'description': "key column is string data type, example: 'G major', 'C major', 'D# minor', 'Db major'", 'type': 'string'}}
        Args:
            sql_query: SQL query string to execute.
            topk: track_ids to return.
        """
        results = self.sql_tool.sql_retrieval(sql_query, topk)
        return results

    def bm25(self, query: str, corpus_type: str, topk: int):
        """
        Perform BM25 retrieval for lexical matching. lowercase all the input strings.
        Good for retrieval stage and lexical matching. Good for text queries where typos are common and exact string matching is difficult.
        BM25 Schema:
        'title': lowercase of track name
        'artist': lowercase of artist name
        'album': lowercase of album name
        'lyrics': lowercase of lyrics
        'attributes': lowercase of genre, instrument, mood, theme, usage, etc.
        Args:
            query: Search query string.
            corpus_type: One of {"title", "artist", "album", "lyrics", "attributes"}.
            topk: Maximum number of track_ids to return.
        """
        results = self.bm25_tool.bm25_retrieval(corpus_type, query, topk)
        return results

    def text_to_item_similarity(self, query: str, modality_type: str, corpus_type: str, topk: int):
        """
        Perform text-to-item similarity retrieval.
        Good for retrieval stage and reranking stage. Good for semantic queries.
        if corpus_type is "metadata", "lyrics", "attributes", "cf_item", use text modality.
        if corpus_type is "audio", "image", use audio/image modality, respectively.
        Args:
            query: Search query string.
            modality_type: One of {"text", "audio", "image"}.
            corpus_type: One of {"metadata", "lyrics", "attributes", "audio", "image"}.
            topk: Maximum number of track_ids to return.
        """
        results = self.embedding_tool.text_to_item_similarity(modality_type, corpus_type, query, topk)
        return results

    def item_to_item_similarity(self, track_id: str, modality_type: str, corpus_type: str, topk: int):
        """
        Perform item-to-item similarity retrieval. track_id is a 22 character string.
        Good for both retrieval and reranking stage. Good for similarity queries (similar track, similar artist, similar album, etc.).
        Args:
            track_id: unique track identifier (string).
            modality_type: One of {"audio", "image", "cf"}.
            corpus_type: One of {"audio", "image", "cf"}.
            topk: Maximum number of track_ids to return.
        """
        results = self.embedding_tool.item_to_item_similarity(modality_type, corpus_type, track_id, topk)
        return results

    def user_to_item_similarity(self, user_id: str, topk: int):
        """
        Perform user-to-item similarity retrieval. for this tool, use only user_id in demographic information. if user_type is "cold_start", don't select this tool.
        Good for both retrieval and reranking stage. Good for personalization.
        Args:
            user_id: unique user identifier (string).
            topk: Maximum number of results to return.
        """
        results = self.embedding_tool.user_to_item_similarity(user_id, topk)
        return results


    def semantic_id_matching(self, modality_type: str, indices: list[int], topk: int):
        """
        Perform semantic ID matching.
        Args:
            modality_type: One of {"audio", "image", "metadata", "lyrics", "attributes", "cf_item"}.
            indices: List of indices.
            topk: Maximum number of results to return.
        """
        results = self.semantic_id_tool.semantic_id_matching(modality_type, indices, topk)
        return results

# tool_executor = ToolExecutor()
# track_pool = ['78zL2RIYaIyV7wT65ntOrg', '1RxaEvQCmT36nIZFYjpk5F', '7tpkQLjgZwNHQzagkvAJ7h', '4nQNNwnIUUmFzVOcVVJCk1', '40LYL1Z6xgCn5cBybo5K0D', '49xSKuTCVjDyFYWZD2p6r9', '5q9o521FdgonxiKsvAFIPJ', '3yYgm1bUjfSu5b9N800geT', '4H4VLLPVVsPvPb9H1gGR72', '50e8KL7TnyNKLaWfCSr0xf']
# sql_results = tool_executor.execute("sql", {"sql_query": "SELECT track_id FROM tracks ORDER BY release_date", "topk": 1}, ['3gpLPDbqIreZmGHn7LHTli', '23GuZ9crtV50nrB4XuiF7E', '2bBAAVvOHsr7vxWFTuYxH8'])
# bm25_results = tool_executor.execute("bm25", {"query": "drake", "corpus_type": "metadata", "topk": 5}, track_pool=track_pool)
# semantic_id_results = tool_executor.execute("semantic_id_matching", {"modality_type": "metadata", "indices": [41,2,5,1], "topk": 5}, track_pool=track_pool)
# text_embedding_results = tool_executor.execute("text_to_item_similarity", {"query": "drake", "modality_type": "text", "corpus_type": "metadata", "topk": 5}, track_pool=track_pool)
# user_embedding_results = tool_executor.execute("user_to_item_similarity", {"user_id": "123", "topk": 5}, track_pool=track_pool)
# item_embedding_results = tool_executor.execute("item_to_item_similarity", {"track_id": "78zL2RIYaIyV7wT65ntOrg", "modality_type": "audio", "corpus_type": "audio", "topk": 5}, track_pool=track_pool)

# print(sql_results)
# print(bm25_results)
# print(semantic_id_results)
# print(text_embedding_results)
# print(user_embedding_results)
# print(item_embedding_results)
