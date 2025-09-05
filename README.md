# TalkPlay-Tools
[![arXiv](https://img.shields.io/badge/arXiv-####-blue.svg)](#)
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

## Requirements / Setup

- Python 3.11
- Linux, macOS, or WSL; GPU recommended for embedding/LLM models (CPU works but slower)

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
wget https://huggingface.co/datasets/talkpl-ai/TalkPlayTools-Env/resolve/main/tool_env.tar.gz
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
python run.py --user_query "I'm looking for calm and slow tempo piano music."
```

Example output:

```bash
----------------------------------------------------------------------------------------------------
🎵 Music: https://open.spotify.com/track/00CXUMREit80f2McJsjcIz
🤖 Assistant Response:
I’ve found a perfect match for your request!
**"Lieder ohne Worte (Songs without Words), Book 2, Op. 30: No. 7 in E-flat major"**
by **Felix Mendelssohn**, performed by **Péter Nagy**.
This classical piano piece features a **slow tempo (65.79 BPM)**, **mellow melodies**,
and a **romantic, emotional tone** that exudes calm and introspection.

The track’s **F# major key** and **instrumental, melancholic style**
make it ideal for a relaxed, reflective mood. It’s a beautifully
crafted piece that aligns perfectly with your request for calm and slow piano music.

Would you like to explore similar tracks, or need recommendations for different moods?
I’m here to help! 🎹
----------------------------------------------------------------------------------------------------
More detail results (Chain of Thought / Tool Calling / Response) are saved in ./demo/static
```

## Demo App
We've implemented an interactive Gradio web interface for multi-turn conversations with the TalkPlay agent. Note that multi-turn conversation capabilities will be updated in future releases.

To launch the demo interface:
```
python app.py
```

<p align="center">
  <img src="https://i.imgur.com/uyCUWwF.png" alt="Gardio Demo for Tool Calling">
</p>


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
