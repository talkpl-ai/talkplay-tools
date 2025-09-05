"""Utility helpers for agent operations.

Currently includes parsing functions for extracting tool calls from model
outputs.
"""
import re
import json

def parsing_multiple_tool_response(text: str):
    """
    Parse the text to a list of tool calls.
    Args:
        text: The text to parse.
    Returns:
        list_of_tool_calls: A list of tool calls.
    """
    list_of_tool_calls = []
    pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    for match in pattern.finditer(text):
        json_block = match.group(1).strip()
        start_idx = json_block.find("{")
        end_idx = json_block.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            continue
        candidate = json_block[start_idx:end_idx + 1]
        for attempt in [candidate, re.sub(r",\s*([}\]])", r"\1", candidate)]:
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                    list_of_tool_calls.append(obj)
                    break
            except json.JSONDecodeError:
                continue
    return list_of_tool_calls
