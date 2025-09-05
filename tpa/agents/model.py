"""Model abstractions and logits processing utilities for TalkPlay agents.

Provides `LLM` for chat completions and tool calling, and a
`InsertToolTokenProcessor` to inject tool markers during generation.
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import get_json_schema
from transformers import LogitsProcessor, LogitsProcessorList


class InsertToolTokenProcessor(LogitsProcessor):
	"""Logits processor that inserts a token sequence after a trigger id.
	This is used to inject the `<tools>` token sequence immediately after the
	Qwen end-of-thought token so the model emits tool calls.
	Args:
		trigger_id (int): Token id that starts insertion (e.g., end-of-think).
		insert_ids (list): Token ids to insert sequentially.
		prefix_len (int): Input prefix length in tokens at generation start.
	"""
	def __init__(self, trigger_id: int, insert_ids: list, prefix_len: int):
		self.trigger_id = trigger_id
		self.insert_ids = insert_ids
		self.prefix_len = prefix_len
		self.inserting = False
		self.idx = 0
	def __call__(self, input_ids, scores):
		"""Apply constrained decoding to insert tokens.
		Args:
			input_ids (torch.Tensor): Current sequence ids of shape (1, seq_len).
			scores (torch.Tensor): Logits for the next token of shape (1, vocab).
		Returns:
			torch.Tensor: Potentially modified scores to force target token ids.
		"""
		# batch size assumed 1
		seq_len = input_ids.shape[1]
		last_id = input_ids[0, -1].item()
		if self.inserting and self.insert_ids:
			target_id = self.insert_ids[self.idx]
			scores[:] = -float("inf")
			scores[0, target_id] = 0.0
			self.idx += 1
			if self.idx >= len(self.insert_ids):
				self.inserting = False
		return scores

class LLM:
    """
    A language model wrapper for music recommendation with tool calling capabilities.
    This class provides a unified interface for interacting with large language models
    to generate tool calls and responses for music recommendation tasks.
    Args:
        tools (list): List of available tools/functions that the model can call
        model_name (str, optional): HuggingFace model identifier. Defaults to "Qwen/Qwen3-4B-AWQ"
        device (str, optional): Device to run the model on. Defaults to "cuda"
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 8192
    """
    def __init__(self,
        tools: list,
        model_name="Qwen/Qwen3-4B",
        device="cuda",
        max_new_tokens=8192,
    ):
        self.tools = tools
        self.model_name = model_name
        self.start_of_tools = "<tools>"
        self.end_of_tools = "</tools>"
        self.tool_functions = [get_json_schema(tool_config["function"]) for tool_config in self.tools.values()]
        self.model, self.tokenizer = self._load_model()
        self.device = device
        self.model.to(self.device)
        self.end_of_think = self.tokenizer.convert_tokens_to_ids("</think>")  # qwen end of thought token
        self.tools_token_ids = self.tokenizer.encode(self.start_of_tools, add_special_tokens=False)
        self.qwen_sampling_params = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "do_sample": True,
        }

    def _load_model(self):
        """Load the model and tokenizer.
        Returns:
            tuple: `(model, tokenizer)` ready for inference.
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model.eval()
        return model, tokenizer

    def tool_calling_chat_completion(self, prompt, chat_history, message, incontext_examples=None, max_new_tokens=8192):
        """Generate tool-call content given a prompt and chat history.
        Args:
            prompt (str): System prompt to steer tool-calling behavior.
            chat_history (list[dict]): Prior conversation messages.
            message (str): Current user message.
            incontext_examples (str | None): Optional in-context examples.
            max_new_tokens (int): Maximum tokens to generate.
        Returns:
            tuple[str, int, str, str]: Model input text, token length, model
            thinking content, and raw tool-call markup.
        """
        messages = [{"role": "system", "content": prompt}]
        messages.extend(chat_history)
        if incontext_examples is None:
            messages.append({"role": "user", "content": message})
        else:
            messages.append({"role": "user", "content": message + "\n" + incontext_examples})
        model_input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            tools=self.tool_functions,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = self.tokenizer(model_input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        model_input_token_len = model_inputs.input_ids.shape[1]
        logits_processor = LogitsProcessorList([
            InsertToolTokenProcessor(
                trigger_id=self.end_of_think,
                insert_ids=self.tools_token_ids,
                prefix_len=model_input_token_len
            )
        ])
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_processor,
                **self.qwen_sampling_params
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        think_index = len(output_ids) - output_ids[::-1].index(self.end_of_think)
        thinking_content = self.tokenizer.decode(output_ids[:think_index], skip_special_tokens=True).strip("\n")
        tool_content = self.tokenizer.decode(output_ids[think_index:], skip_special_tokens=True).strip("\n")
        return model_input_text, model_input_token_len, thinking_content, tool_content

    def response_chat_completion(self, prompt, chat_history, message, recommend_track_metadata, max_new_tokens=8192):
        """Generate a natural-language response grounded in recommendation data.
        Args:
            prompt (str): System prompt to steer answer generation.
            chat_history (list[dict]): Prior conversation messages.
            message (str): Current user message.
            recommend_track_metadata (dict | str): Top recommendation metadata.
            max_new_tokens (int): Maximum tokens to generate.
        Returns:
            tuple[str, int, str, str]: Model input text, token length, model thinking content, and final answer text.
        """
        messages = [{"role": "system", "content": prompt}]
        messages.extend(chat_history)
        messages.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"{recommend_track_metadata}"}
        ])
        model_input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = self.tokenizer(model_input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        model_input_token_len = model_inputs.input_ids.shape[1]
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                **self.qwen_sampling_params
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        think_index = len(output_ids) - output_ids[::-1].index(self.end_of_think)
        thinking_content = self.tokenizer.decode(output_ids[:think_index], skip_special_tokens=True).strip("\n")
        answer_content = self.tokenizer.decode(output_ids[think_index:], skip_special_tokens=True).strip("\n")
        return model_input_text, model_input_token_len, thinking_content, answer_content
