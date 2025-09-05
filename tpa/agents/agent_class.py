"""Agent implementations used by TalkPlay for music recommendation.

Contains `MusicRecommendationAgent`, which orchestrates tool calling and
response generation using an LLM and domain-specific tools and databases.
"""
# reference: https://github.com/QwenLM/Qwen-Agent
from typing import List, Dict, Any
import os
import json
import random
from time import time
from tpa.agents.agent_utils import parsing_multiple_tool_response

class MusicRecommendationAgent:
    """
    A music recommendation agent that provides personalized track recommendations
    through tool-based interactions and conversational responses.
    Args:
        tool_executor: Tool execution engine for music recommendation tools
        llm: Language model for generating responses and tool calls
        user_db: User profile database for storing and retrieving user information
        item_db: Music catalog database containing track metadata and information
        cache_dir (str, optional): Directory for caching data. Defaults to "./cache"
    """
    def __init__(self, tool_executor, llm, user_db, item_db, cache_dir: str = "./cache"):
        """Initialize the music recommendation agent.
        Args:
            tool_executor: Executor that runs music recommendation tools.
            llm: Language model wrapper used for tool calling and response generation.
            user_db: User profile store for retrieving user information.
            item_db: Music catalog providing track metadata and a track pool.
            cache_dir (str, optional): Directory path for caching data.
        """
        self.cache_dir = cache_dir
        self.tool_executor = tool_executor
        self.llm = llm
        self.user_db = user_db
        self.item_db = item_db
        self.track_pool = item_db.track_pool
        self.prompts_dir = os.path.join(os.path.dirname(__file__), "system_prompts")
        self.session_memory = []
        self.user_info = {}
        self.prompts = {
            "role_play": open(f"{self.prompts_dir}/roleplay.txt", "r", encoding="utf-8").read(),
            "personalization": open(f"{self.prompts_dir}/personalization.txt", "r", encoding="utf-8").read(),
            "goal_tool_calling": open(f"{self.prompts_dir}/goal_tool_calling.txt", "r", encoding="utf-8").read(),
            "goal_response_generation": open(f"{self.prompts_dir}/goal_response_generation.txt", "r", encoding="utf-8").read(),
        }

    def _load_user_profile(self, user_id: str= None, explicit_user_info: dict = None):
        """Load user profile into memory.
        One of `user_id` or `explicit_user_info` must be provided.
        Args:
            user_id (str, optional): Identifier of the user whose profile is fetched.
            explicit_user_info (dict, optional): Pre-fetched user info to use directly.
        Raises:
            ValueError: If neither `user_id` nor `explicit_user_info` is provided.
        """
        if explicit_user_info:
            self.user_info = explicit_user_info
        elif user_id:
            self.user_info = self.user_db.user_id_to_profile(user_id)
        else:
            raise ValueError("user_id or explicit_user_info is required")

    def _personalize_profile(self):
        """Construct a personalization prompt segment for the current user.
        Returns:
            str: A formatted string encoding demographics and in-context examples.
        """
        base_propmt = self.prompts["personalization"]
        demographics = f"[DEMOGRAPHIC INFORMATION]\n user_id: {self.user_info['user_id']}, user_type: {self.user_info['user_type']}, age_group: {self.user_info['age_group']}, gender: {self.user_info['gender']}"
        if len(self.user_info['previous_history']) > 0:
            incontext_examples = random.sample(self.user_info['previous_history'], 5)
            incontext_examples = [self.item_db.id_to_metadata(track_id, use_semantic_id=True) for track_id in incontext_examples]
            incontext_prompt = f"[PREVIOUS SESSION EXAMPLES]: Examples of tracks that the user_id: {self.user_info['user_id']} has listened to in the last session:\n{incontext_examples}"
        else:
            incontext_prompt = ""
        return base_propmt + "\n" + demographics + "\n" + incontext_prompt

    def _reset_session_memory(self):
        """Clear the in-memory chat session history."""
        self.session_memory = []

    def tool_calling(self, tool_response: str):
        """Execute tool calls produced by the LLM and collect results.
        Args:
            tool_response (str): Raw tool call markup emitted by the model.
        Returns:
            tuple[int, list[dict]]: Number of tool calls and a list of per-call
            result dictionaries containing tool name, args, success flag,
            candidate recommendation ids, and error message if any.
        """
        tool_response = parsing_multiple_tool_response(tool_response)
        num_of_tool_calls = len(tool_response)
        recommend_track_ids = self.track_pool.copy()
        tool_call_results = []
        for tool_call in tool_response:
            current_recommend_track_ids = recommend_track_ids.copy()
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]
            try:
                tool_call_success, recommend_track_ids = self.tool_executor.execute(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    track_pool = current_recommend_track_ids
                )
                error_message = ""
            except Exception as e:
                tool_call_success = False
                recommend_track_ids = []
                error_message = str(e)
            if tool_call_success == True: # error case 1: tool call failed
                current_recommend_track_ids = recommend_track_ids
            tool_call_results.append({
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_call_success": tool_call_success,
                "recommend_track_ids": current_recommend_track_ids,
                "error_message": error_message
            })
        return num_of_tool_calls, tool_call_results

    def _return_error_response(self, message, num_of_tool_calls, tool_call_results, tool_input_text, tool_input_token_len, tool_calling_cot, tool_calling_time):
        """Format an error payload when tool calling cannot proceed.
        Returns:
            dict: Structured error response payload.
        """
        return {
            "user_query": message,
            "tool_input_text": tool_input_text,
            "tool_input_token_len": tool_input_token_len,
            "tool_calling_cot": tool_calling_cot,
            "num_of_tool_calls": num_of_tool_calls,
            "tool_calling_time": tool_calling_time,
            "tool_call_results": tool_call_results
        }

    def chat(self, message: str):
        """Run a full recommendation cycle and generate a user-facing answer.
        This performs three stages: tool-calling prompt → tool execution → answer generation prompt.
        Args:
            message (str): The user's message or query.
        Returns:
            dict: A structured payload containing inputs/outputs and timings for
            tool calling, tool execution, and response generation stages.
        """
        self.session_memory.append({"role": "user", "content": message})
        # stae 0. use personalization
        personalization_prompt = self._personalize_profile()
        personalization_tool_calling = "\n PERSONALIZATION: For reranking stage, prioritize user_to_item_similarity when user_type is \"warm_start\". For \"cold_start\" users, don't use user_to_item_similarity."
        tool_calling_prompt = self.prompts["role_play"] + personalization_prompt + self.prompts["goal_tool_calling"] + personalization_tool_calling
        response_prompt = self.prompts["role_play"] + personalization_prompt + self.prompts["goal_response_generation"]
        # stage 1. tool calling
        start = time()
        tool_input_text, tool_input_token_len, tool_calling_cot, tool_response = self.llm.tool_calling_chat_completion(
            prompt=tool_calling_prompt,
            chat_history=self.session_memory,
            message=message,
            incontext_examples=None,
            max_new_tokens=32768
        )
        tool_calling_time = time() - start
        # stage 2, tool calling
        start = time()
        num_of_tool_calls, tool_call_results = self.tool_calling(tool_response)
        recommend_track_metadata = []
        if len(tool_call_results) == 0:
            return self._return_error_response(message, num_of_tool_calls, tool_call_results, tool_input_text, tool_input_token_len, tool_calling_cot, tool_calling_time)
        top1_recommend_track_id = tool_call_results[-1]['recommend_track_ids'][0]
        top1_recommend_track_metadata = self.item_db.id_to_metadata(top1_recommend_track_id)
        tool_execution_time = time() - start
        # stage 3. response generation
        start = time()
        answer_input_text, answer_input_token_len, answer_cot, answer_response = self.llm.response_chat_completion(
            prompt=response_prompt,
            chat_history=self.session_memory,
            message=message,
            recommend_track_metadata=top1_recommend_track_metadata,
            max_new_tokens=32768
        )
        answer_generation_time = time() - start
        self.session_memory.append({"role": "assistant", "content": f"recommended track: {top1_recommend_track_metadata}" + "\n" + answer_response})
        return {
            # User Input
            "user_query": message,
            # Tool Calling Stage
            "tool_input_text": tool_input_text,
            "tool_input_token_len": tool_input_token_len,
            "tool_calling_cot": tool_calling_cot,
            "num_of_tool_calls": num_of_tool_calls,
            "tool_calling_time": tool_calling_time,
            "tool_call_results": tool_call_results,
            # Tool Execution Stage
            "top1_recommend_track_id": top1_recommend_track_id,
            "tool_execution_time": tool_execution_time,
            # Response Generation Stage
            "answer_input_text": answer_input_text,
            "answer_input_token_len": answer_input_token_len,
            "answer_cot": answer_cot,
            "answer_response": answer_response,
            "answer_generation_time": answer_generation_time,
        }
