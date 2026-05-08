"""
Runtime configuration.

Values are read from environment variables so the public repository does not
contain API keys, passwords, or machine-specific paths. Copy `.env.example` to
`.env` for local development, or export the variables in your shell.
"""

import os

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at import time.
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _env(name, default=""):
    return os.getenv(name, default)


# 大模型相关
QWEN_API_KEY = _env("QWEN_API_KEY")
LLM_MODEL_NAME = _env("LLM_MODEL_NAME", "qwen-max-latest")
LLM_MODEL_NAME_2 = _env("LLM_MODEL_NAME_2", LLM_MODEL_NAME)
VLM_MODEL_NAME = _env("VLM_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
DS_R1_API_KEY = _env("DS_R1_API_KEY")
DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY")

VLM_MODEL_NAME_QWEN25VL = _env("VLM_MODEL_NAME_QWEN25VL", "Qwen/Qwen2.5-VL-7B-Instruct")
VLM_MODEL_NAME_QWEN2VL = _env("VLM_MODEL_NAME_QWEN2VL", "Qwen/Qwen2-VL-7B-Instruct")
VLM_MODEL_NAME_QWEN3VL = _env("VLM_MODEL_NAME_QWEN3VL", "Qwen/Qwen3-VL-4B-Instruct")
VLM_MODEL_NAME_VIDEOLLAMA3 = _env("VLM_MODEL_NAME_VIDEOLLAMA3", "DAMO-NLP-SG/VideoLLaMA3-7B")
VLM_MODEL_NAME_INTERNVL25 = _env("VLM_MODEL_NAME_INTERNVL25", "OpenGVLab/InternVL2.5-8B")
VLM_MODEL_NAME_INTERNVL3 = _env("VLM_MODEL_NAME_INTERNVL3", "OpenGVLab/InternVL3-8B")


# 图数据库相关
NEO4J_URI = _env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = _env("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = _env("NEO4J_PASSWORD")
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)
