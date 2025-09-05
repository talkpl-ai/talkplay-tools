"""Agents package entry point for constructing a configured TalkPlay music
recommendation agent.

Exposes `load_talkplay_agent` for convenient creation of a
`MusicRecommendationAgent`, including its `LLM`, tools, and data stores.
"""
from tpa.agents.agent_class import MusicRecommendationAgent
from tpa.agents.model import LLM
from tpa.environments import ToolExecutor, MusicCatalog, UserProfileDB

__all__ = ["MusicRecommendationAgent", "load_talkplay_agent"]

def load_talkplay_agent(cache_dir: str = "./cache", model_name: str = "Qwen/Qwen3-4B"):
    """Load and initialize a TalkPlay music recommendation agent.
    Args:
        cache_dir (str, optional): Directory path for caching data and models.
            Defaults to "./cache".
        model_name (str, optional): Name of the language model to use for the agent.
            Defaults to "Qwen/Qwen3-4B-AWQ".
    Returns:
        MusicRecommendationAgent: A fully configured music recommendation agent ready
            for use.
    """
    tool_executor = ToolExecutor(cache_dir=cache_dir)
    llm = LLM(model_name=model_name, tools=tool_executor.tools)
    user_profiler = UserProfileDB(cache_dir=cache_dir)
    music_catalog = MusicCatalog(cache_dir=cache_dir)
    agent = MusicRecommendationAgent(
        tool_executor=tool_executor,
        llm=llm,
        user_db=user_profiler,
        item_db=music_catalog,
        cache_dir=cache_dir,
    )
    return agent
