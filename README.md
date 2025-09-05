# TalkPlay-Tools
[![arXiv](https://img.shields.io/badge/arXiv-2410.03264-blue.svg)](#)
[![Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2)

An LLM-powered music recommendation system that uses tool calling to orchestrate a unified retrieval → reranking pipeline over SQL, BM25, embeddings (text/audio/image/CF), and semantic IDs.

<p align="center">
  <img src="https://i.imgur.com/sWgWXkb.png" alt="TalkPlay Tools Overview">
</p>

## Features

- **Agentic pipeline**: LLM plans tool calls, executes retrieval, and generates a grounded response.
- **Multi-tool retrieval**: SQL filtering, BM25 lexical search, text/audio/image/CF embeddings, semantic-ID matching.
- **Personalization**: Warm/cold-start aware strategies with user-item similarity when applicable.
- **Repro-friendly**: Lightweight test indices for quick demos; cache-first design for tools and models.

## Requirements

- Python 3.11
- Linux, macOS, or WSL; GPU recommended for embedding/LLM models (CPU works but slower)

### Setup

```bash
python -m pip install uv
uv venv .venv --python 3.11
source .venv/bin/activate
uv add torch torchvision torchaudio
uv pip install laion_clap
uv pip install -e .
```

## Fetch Tool Environments

Prebuilt demo indices are expected under `./cache`. You can download a prepared bundle and extract it:

```bash
wget UPDATE_SOON/{tool_env.tar.gz}
tar -xzvf tool_env.tar.gz -C ./cache
```

Expected subdirectories (after extraction):

- `cache/metadata` (test metadata files)
- `cache/bm25` (BM25 indices and `track_index.json`)
- `cache/encoder` (vector DB for embeddings)
- `cache/semantic_id` (RVQ indices per modality)
- `cache/sql` (SQLite DB for tracks)

## Quick Start

Run an example query with the provided test data (cold-start user case). This demo uses only **6,744 tracks** from the test split of [TalkPlayData-2](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2). Due to licensing constraints, the system returns Spotify links instead of direct audio files.

```bash
python run.py --user_query "I'm looking for something chill and relaxed, slow tempo piano music."
```

Example output:

```bash
----------------------------------------------------------------------------------------------------
🎵 Music: https://open.spotify.com/track/6lNUewdE3ZY4vUMxXpHtIC
🤖 Assistant Response:
I’ve found a perfect match for you! **"White Lake" by Deaf Center** is a slow-tempo (61.7 BPM) piano piece with an atmospheric, calm, and melancholic vibe. It falls under experimental and alternative genres, but its soothing, dreamy quality makes it ideal for relaxing. The Bb minor key adds a subtle emotional depth, while the track’s soft, instrumental nature keeps it chill and easy to listen to.

This track seems to align well with your request for slow, relaxed piano music. Would you like to explore similar tracks or need recommendations for different moods? 😊
----------------------------------------------------------------------------------------------------
More detail results (Chain of Thought / Tool Calling / Response) are saved in ./demo/static
```

## Demo App

```
python app.py
```


### Configuration

- Default LLM: Qwen3-4B (you can customize in `tpa/agents/__init__.py` or via flags if you extend `run.py`).
- Tools and models read from `./cache` by default; set a different path by changing the constructor args when building the agent.

## Project Structure

```
tpa/
  agents/            # Agent, LLM wrapper, prompts
  environments/      # Tool executor, tools, DBs, preprocessing
  evaluation/        # Offline metrics and examples
run.py               # CLI demo entry point
app.py               # Gardio App for demo
```

## Dataset

- Demo/test data: `TalkPlayData-2` on Hugging Face
  - https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2

## License

This project is released under the CC-BY-NC 4.0 license.

## Citation

If this project helps your research, please consider citing our work.

```bibtex
% Coming soon
```
