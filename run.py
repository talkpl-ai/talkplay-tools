import os
import json
import argparse
from tpa.agents import load_talkplay_agent

def main(args):
    agent = load_talkplay_agent(model_name=args.model_name)
    # Simple Cold-start user case
    agent._load_user_profile(
        explicit_user_info={
            "user_id": "NEW_USER",
            "age_group": args.age_group,
            "gender": args.gender,
            "country_name": args.country_name,
            "user_type": args.user_type,
            "previous_history": []
        }
    )
    results = agent.chat(args.user_query)
    os.makedirs(args.save_path, exist_ok=True)
    with open(f"{args.save_path}/results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    track_id = results['tool_call_results'][-1]["recommend_track_ids"][0]
    track_url = f"https://open.spotify.com/track/{track_id}"
    print("-"*100)
    print(f"🎵 Music: {track_url}")
    print("🤖 Assistant Response:")
    print(results["answer_response"])
    print("-"*100)
    print(f"More detail results (Chain of Thought / Tool Calling / Response) are saved in {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--age_group", type=str, default="20s")
    parser.add_argument("--gender", type=str, default="male")
    parser.add_argument("--country_name", type=str, default="United States")
    parser.add_argument("--user_type", type=str, default="cold_start")
    parser.add_argument("--user_query", type=str, default="I'm looking for something chill and relaxed, slow tempo piano music.")
    parser.add_argument("--save_path", type=str, default="./demo/static")
    args = parser.parse_args()
    main(args)
