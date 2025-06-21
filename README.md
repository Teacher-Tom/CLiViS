# CLiViS: Unleashing Cognitive Map through Linguistic-Visual Synergy for Embodied Visual Reasoning

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of our paper *Unleashing Cognitive Map through Linguistic-Visual Synergy for Embodied Visual Reasoning*.

## 📄 Abstract

  Embodied Visual Reasoning (EVR) seeks to follow complex, free-form instructions based on egocentric video, enabling semantic understanding and spatiotemporal reasoning in dynamic environments. Despite its promising potential, EVR encounters significant challenges stemming from the diversity of complex instructions and the intricate spatiotemporal dynamics in long-term egocentric videos. Prior solutions either employ Large Language Models (LLMs) over static video captions, which often omit critical visual details, or rely on end‑to‑end Vision-Language Models (VLMs) that struggle with stepwise compositional reasoning. Consider the complementary strengths of LLMs in reasoning and VLMs in perception, we propose **CLiViS**. It is a novel training-free framework that leverages LLMs for high-level task planning and orchestrates VLM‑driven open‑world visual perception to iteratively update the scene context. Building on this synergy, the core of CLiViS is a dynamic **Cognitive Map** that evolves throughout the reasoning process. This map constructs a structured representation of the embodied scene, bridging low-level perception and high-level reasoning. Extensive experiments across multiple benchmarks demonstrate the effectiveness and generality of CLiViS, especially in handling long‑term visual dependencies.

## 📚 Citation

If you use this code or build upon our work, please cite:

```bibtex
@misc{li2025cliviscognitivemapguided,
      title={CLiViS: Unleashing Cognitive Map through Linguistic-Visual Synergy for Embodied Visual Reasoning}, 
      author={Kailing Li and Qi'ao Xu and Tianwen Qian and Yuqian Fu and Yang Jiao and Xiaoling Wang},
      year={2025},
      eprint={2506.xxxx},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.xxxx}
}
```

## ⏳ Note from the Authors

We are currently preparing the full codebase and plan to upload it shortly. Thank you for your interest and patience!

## 📞 Contact

For any questions or issues, feel free to open an issue or contact us at [51275901046@stu.ecnu.edu.cn].

