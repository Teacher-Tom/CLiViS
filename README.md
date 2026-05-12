# [CVPR 2026] CLiViS: Unleashing Cognitive Map through Linguistic-Visual Synergy for Embodied Visual Reasoning

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Accepted to appear in CVPR 2026!**

This repository contains the official implementation of our paper *Unleashing Cognitive Map through Linguistic-Visual Synergy for Embodied Visual Reasoning*.

## 📄 Abstract

Embodied Visual Reasoning (EVR) seeks to follow complex, free-form instructions based on egocentric video, enabling semantic understanding and spatiotemporal reasoning in dynamic environments. Despite its promising potential, EVR encounters significant challenges stemming from the diversity of complex instructions and the intricate spatiotemporal dynamics in long-term egocentric videos. Prior solutions either employ Large Language Models (LLMs) over static video captions, which often omit critical visual details, or rely on end-to-end Vision-Language Models (VLMs) that struggle with stepwise compositional reasoning.

Considering the complementary strengths of LLMs in reasoning and VLMs in perception, we propose **CLiViS**, a training-free framework that leverages LLMs for high-level task planning and orchestrates VLM-driven open-world visual perception to iteratively update the scene context. The core of CLiViS is a dynamic **Cognitive Map** that evolves throughout the reasoning process, constructing a structured representation of the embodied scene and bridging low-level perception with high-level reasoning. Extensive experiments across multiple benchmarks demonstrate the effectiveness and generality of CLiViS, especially in handling long-term visual dependencies.

## Usage

Run inference from the repository root:

```bash
python main.py --video path/to/video.mp4 --question "What happens after the person picks up the cup?"
```

You can also invoke the package entry point:

```bash
python -m clivis --video path/to/video.mp4 --question "Your question"
```

Useful options:

```bash
python main.py \
  --video path/to/video.mp4 \
  --question "Your question" \
  --output-path output_segments/output \
  --fps 1 \
  --max-rounds 15 \
  --save-result outputs/result.json
```

## Configuration

Runtime configuration is loaded from environment variables in `clivis/preference.py`. For local development, copy `.env.example` to `.env` and fill in the values you need:

```bash
copy .env.example .env
```

On PowerShell, you can also set variables for the current session:

```powershell
$env:QWEN_API_KEY="your-key"
$env:DEEPSEEK_API_KEY="your-key"
$env:NEO4J_PASSWORD="your-password"
```

### CLI Options

| Option | Purpose | Default |
| --- | --- | --- |
| `--video` | Input video path. | required |
| `--question` | Question to answer from the video. | required |
| `--output-path` | Output prefix for generated segment files. | `output_segments/output` |
| `--fps` | Base video sampling FPS for VLM calls. | `1` |
| `--max-pixels` | Maximum pixel budget passed to supported VLM calls. | `172800` |
| `--max-rounds` | Maximum iterative reasoning rounds. | `15` |
| `--save-result` | Optional JSON file path for saving answer and metadata. | unset |

## Code Structure

```text
CLiViS/
├── main.py                         # Thin launcher for the command-line interface
├── clivis/
│   ├── cli.py                      # CLI argument parsing and result serialization
│   ├── preference.py               # Environment-based runtime configuration
│   ├── utils.py                    # Shared parsing and utility helpers
│   ├── pipeline/
│   │   ├── time_inference_od_neo4j.py  # Main iterative reasoning pipeline
│   │   └── time_utils.py           # Time range parsing, segmentation, and FPS helpers
│   ├── graph/
│   │   ├── scene_graph.py          # Cognitive Map orchestration over scene entities
│   │   ├── navigation_graph.py     # Temporal segment and navigation representation
│   │   ├── relation_graph.py       # Object/action/relation graph operations
│   │   └── common.py               # Shared graph data structures
│   ├── memory/
│   │   └── time_working_memory.py  # Iterative evidence and rationale memory
│   ├── models/
│   │   ├── llm.py                  # LLM client wrappers
│   │   └── vlm.py                  # VLM client wrappers
│   └── video/
│       └── spilit_video.py         # Video duration, splitting, and time-range utilities
├── requirements.txt                # Python dependencies
├── .env.example                    # Example environment variables
└── README.md
```

At a high level, `clivis.cli` parses the request and calls the main inference pipeline in `clivis.pipeline`. The pipeline first splits the input video into temporal segments, builds a Cognitive Map through the graph modules, and then iteratively alternates between LLM planning and VLM perception. Evidence collected during this loop is tracked in `clivis.memory`, while model calls are isolated behind the wrappers in `clivis.models`.

## Requirements

- Python 3.10
- PyTorch 2.6.0+ with CUDA-capable hardware recommended
- Transformers 4.50.0.dev0 or newer
- Neo4j database for the graph-backed relation pipeline
- API keys:
  - Qwen / DashScope API key
  - Optional DeepSeek API key if you use the alternative LLM backend
- Core Python packages:
  - `openai`
  - `torch`, `torchvision`, `torchaudio`
  - `transformers`
  - `Pillow`
  - `numpy`
  - `decord` for video loading in some VLM paths
  - `neo4j`
  - `pyyaml`
  - `qwen-vl-utils`
  - `ultralytics`
  - `scikit-learn`
  - `safetensors`
  - `tokenizers`
  - `bitsandbytes`
  - `accelerate`
  - `flash-attn` if you use `attn_implementation="flash_attention_2"`

## 📚 Citation

If you use this code or build upon our work, please cite:

```bibtex
@misc{li2025cliviscognitivemapguided,
      title={CLiViS: Unleashing Cognitive Map through Linguistic-Visual Synergy for Embodied Visual Reasoning}, 
      author={Kailing Li and Qi'ao Xu and Tianwen Qian and Yuqian Fu and Yang Jiao and Xiaoling Wang},
      year={2025},
      eprint={2506.17629},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.17629}
}
```

## ⏳ Note from the Authors

We are cleaning up and organizing the full codebase and will release it progressively. Thank you for your interest and patience.


## 📞 Contact

For any questions or issues, feel free to open an issue or contact us at 51275901046@stu.ecnu.edu.cn.
