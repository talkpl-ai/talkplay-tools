import json
import gradio as gr
from tpa.agents import load_talkplay_agent


def respond(message, history, age_group, gender, country_name, model_name, agent_state):
    """
    Chat handler for the TalkPlay agent. Maintains per-session agent state.
    """
    try:
        agent = agent_state
        if agent is None or getattr(agent, "_model_name", None) != model_name:
            agent = load_talkplay_agent(model_name=model_name)
            agent._model_name = model_name
        agent._load_user_profile(
            explicit_user_info={
                "user_id": "NEW_USER",
                "age_group": age_group,
                "gender": gender,
                "country_name": country_name,
                "user_type": "cold_start",
                "previous_history": [],
            }
        )
        results = agent.chat(message)
        user_query = results.get("user_query", "")
        tool_calling_cot = results.get("tool_calling_cot", "")
        tool_call_results = results.get("tool_call_results", [])
        answer_cot = results.get("answer_cot", "")
        answer_response = results.get("answer_response", "")
        answer_generation_time = results.get("answer_generation_time", "")
        # Build response with recommended track link (if available)
        track_url = None
        tool_calls = results.get("tool_call_results", [])
        if tool_calls and tool_calls[-1].get("recommend_track_ids"):
            track_id_list = tool_calls[-1]["recommend_track_ids"]
            if track_id_list:
                track_url = f"https://open.spotify.com/track/{track_id_list[0]}"
        # Convert tool_call_results to properly formatted JSON string
        if tool_call_results:
            tool_call_results_str = json.dumps(tool_call_results, indent=2, ensure_ascii=False)
        else:
            tool_call_results_str = "[]"
        final_answer = ""
        final_answer += f"[Tool Calling CoT]\n{tool_calling_cot}\n\n"
        final_answer += f"[Tool Call Results]\n{tool_call_results}\n\n"
        final_answer += f"[Answer CoT]\n{answer_cot}\n\n"
        final_answer += f"[Answer Response]\n{answer_response}\n\n"
        if track_url:
            final_answer += f"[Spotify]\n{track_url}\n\n"
        return final_answer, agent

    except Exception as e:
        return f"❌ Error: {str(e)}", agent_state


with gr.Blocks(
    title="TalkPlay Music Recommendation",
    css="""
    html, body, #root, .gradio-container { height: 100%; }
    #app_row { height: calc(100vh - 140px); }
    #left_controls { height: 100%; overflow: auto; }
    #chat_col { height: 100%; display: flex; flex-direction: column; }
    /* Make chatbot fill available vertical space */
    .gradio-container .gr-chatbot { flex: 1 1 auto; height: 100%; min-height: 0; }
    """
) as demo:
    gr.Markdown("# 🎵 TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling")
    gr.Markdown(
        "Chat with the agent to find music you’ll like. Adjust demographics/model as needed; "
        "the conversation remembers context across turns."
    )

    with gr.Row(elem_id="app_row"):
        with gr.Column(scale=1, elem_id="left_controls"):
            age_group = gr.Dropdown(
                choices=["10s", "20s", "30s", "40s", "50s+"],
                value="20s",
                label="Age Group",
            )
            gender = gr.Dropdown(
                choices=["male", "female"],
                value="male",
                label="Gender",
            )
            country_name = gr.Textbox(
                value="United States",
                label="Country",
            )
            model_name = gr.Dropdown(
                choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B"],
                value="Qwen/Qwen3-4B",
                label="Model",
            )
            agent_state = gr.State(value=None)

        with gr.Column(scale=9, elem_id="chat_col"):
            gr.ChatInterface(
                fn=respond,
                additional_inputs=[age_group, gender, country_name, model_name, agent_state],
                additional_outputs=[agent_state]
            )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=1024,
        share=False,
        debug=True,
    )
