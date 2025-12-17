"""
Central configuration for the Akademie KI Suite
"""
import os

# ==========================================
# PATHS
# ==========================================
APP_DIR = "/var/www/transkript_app"
LOG_FILE = os.path.join(APP_DIR, "app.log")
STATIC_DIR = os.path.join(APP_DIR, "static")
IMAGES_DIR = os.path.join(APP_DIR, "generated_images")
STORAGE_MOUNT_POINT = "/mnt/akademie_storage"
JOB_STATE_DIR = os.path.join(APP_DIR, "jobs")

# ==========================================
# DATABASE
# ==========================================
DATABASE_URL = "sqlite:///akademie_suite.db"

# ==========================================
# API KEYS
# ==========================================
API_KEYS = {
    "GLADIA": os.environ.get("GLADIA_API_KEY", "your_key"),
    "SCALEWAY": os.environ.get("SCALEWAY_API_KEY", "your_key"),
    "NEBIUS": os.environ.get("NEBIUS_API_KEY", "your_key"),
    "MISTRAL": os.environ.get("MISTRAL_API_KEY", "your_key"),
    "OPENROUTER": os.environ.get("OPENROUTER_API_KEY", "your_key"),
    "GROQ": os.environ.get("GROQ_API_KEY", "your_key"),
    "POE": os.environ.get("POE_API_KEY", "your_poe_key_here"),
    "DEEPGRAM": os.environ.get("DEEPGRAM_API_KEY", "your_key"), 
    "ASSEMBLYAI": os.environ.get("ASSEMBLYAI_API_KEY", "your_key"),
    "OPENAI": os.environ.get("OPENAI_API_KEY", "your_key"),
    "COHERE": os.environ.get("COHERE_API_KEY", "your_key"),
    "TOGETHER": os.environ.get("TOGETHER_API_KEY", "your_key"),
    "OVH": os.environ.get("OVH_API_KEY", "your_key"),
    "CEREBRAS": os.environ.get("CEREBRAS_API_KEY", "your_key"),
    "GOOGLEAI": os.environ.get("GOOGLEAI_API_KEY", "your_key"),
    "ANTHROPIC": os.environ.get("ANTHROPIC_API_KEY", "your_key"),
}

# ==========================================
# SECURITY
# ==========================================
EU_ONLY_MODE = True

RESTRICTED_PROVIDERS = {
    "OpenAI": "🇺🇸 US-Server",
    "Anthropic": "🇺🇸 US-Server", 
    "OpenRouter": "🇺🇸 US-Server",
    "Groq": "🇺🇸 US-Server",
    "Poe": "🇺🇸 US-Server",
    "Cohere": "🇺🇸 US-Server",
    "Together": "🇺🇸 US-Server",
    "Cerebras": "🇺🇸 US-Server",
}

# ==========================================
# PROVIDERS CONFIG
# ==========================================
PROVIDERS = {
    "Scaleway": {
        "base_url": "https://api.scaleway.ai/v1",
        "key_name": "SCALEWAY",
        "badge": "🇫🇷 <b>DSGVO-Konform</b>",
        "chat_models": [
            "gpt-oss-120b", 
            "mistral-small-3.2-24b-instruct-2506", 
            "gemma-3-27b-it", 
            "qwen3-235b-a22b-instruct-2507", 
            "llama-3.3-70b-instruct", 
            "deepseek-r1-distill-llama-70b"
        ],
        "vision_models": ["pixtral-12b-2409", "mistral-small-3.1-24b-instruct-2503"],
        "audio_models": ["whisper-large-v3"],
        "image_models": ["pixtral-12b-2409"],
        "context_limits": {
            "gpt-oss-120b": 32768,
            "mistral-small-3.2-24b-instruct-2506": 32768,
            "gemma-3-27b-it": 96000,
            "qwen3-235b-a22b-instruct-2507": 131072,
            "llama-3.3-70b-instruct": 131072,
            "deepseek-r1-distill-llama-70b": 8192,
            "pixtral-12b-2409": 32768,
            "mistral-small-3.1-24b-instruct-2503": 96000,
            "whisper-large-v3": 16384,
        }
    },
    
    "Gladia": {
        "base_url": "https://api.gladia.io/v2",
        "key_name": "GLADIA",
        "badge": "🇫🇷 <b>DSGVO-Konform</b>",
        "audio_models": ["gladia-v2"],
        "context_limits": {
            "gladia-v2": 1000000 
        }
    },
    
    "Nebius": {
        "base_url": "https://api.tokenfactory.nebius.com/v1",
        "key_name": "NEBIUS",
        "badge": "🇪🇺 <b>DSGVO-Konform</b>",
        "chat_models": [
            "deepseek-ai/DeepSeek-R1-0528",
            "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
            "openai/gpt-oss-120b",
            "moonshotai/Kimi-K2-Instruct",
            "moonshotai/Kimi-K2-Thinking",
            "zai-org/GLM-4.5",
            "meta-llama/Llama-3.3-70B-Instruct"
        ],
        "image_models": ["black-forest-labs/flux-schnell", "black-forest-labs/flux-dev"],
        "vision_models": ["google/gemma-3-27b-it", "Qwen/Qwen2.5-VL-72B-Instruct", "nvidia/Nemotron-Nano-V2-12b"],
        "context_limits": {
            "deepseek-ai/DeepSeek-R1-0528": 163840,
            "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1": 131072,
            "openai/gpt-oss-120b": 32768,
            "moonshotai/Kimi-K2-Instruct": 128000,
            "moonshotai/Kimi-K2-Thinking": 128000,
            "zai-org/GLM-4.5": 128000,
            "meta-llama/Llama-3.3-70B-Instruct": 131072,
            "black-forest-labs/flux-schnell": 4096,
            "black-forest-labs/flux-dev": 4096,
        }
    },
    
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "key_name": "MISTRAL",
        "badge": "🇫🇷 <b>DSGVO-Konform</b>",
        "chat_models": [
            "mistral-large-latest",
            "mistral-medium-2508",
            "magistral-medium-2509",
            "open-mistral-nemo-2407"
        ],
        "vision_models": ["pixtral-large-2411", "pixtral-12b-2409", "mistral-ocr-latest"],
        "audio_models": ["voxtral-mini-latest"],
        "context_limits": {
            "mistral-large-latest": 128000,
            "mistral-medium-2508": 128000,
            "magistral-medium-2509": 128000,
            "open-mistral-nemo-2407": 128000,
            "pixtral-large-2411": 128000,
            "pixtral-12b-2409": 32768,
            "mistral-ocr-latest": 32768,
            "voxtral-mini-latest": 16384,
        }
    },
    
    "Deepgram": {
        "base_url": "https://api.eu.deepgram.com/v1",
        "key_name": "DEEPGRAM",
        "badge": "🇪🇺 <b>EU-Server, US-Firma</b>",
        "audio_models": ["nova-3-general", "nova-2-general", "nova-2"],
        "context_limits": {
            "nova-3-general": 16384,
            "nova-2-general": 16384,
            "nova-2": 16384,
        }
    },
    
    "AssemblyAI": {
        "base_url": "https://api.eu.assemblyai.com/v2",
        "key_name": "ASSEMBLYAI",
        "badge": "🇪🇺 <b>EU-Server, US-Firma</b>",
        "audio_models": ["universal", "slam-1"],
        "context_limits": {
            "universal": 16384,
            "slam-1": 16384,
        }
    },
    
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_name": "OPENROUTER",
        "badge": "🇺🇸 <b>US-Server</b>!",
        "chat_models": [
            # 1M+ Context
            "google/gemini-2.0-pro-exp-02-05:free",
            "google/gemini-2.0-flash-thinking-exp:free",
            "google/gemini-2.0-flash-exp:free",
            "google/gemini-2.5-pro-exp-03-25:free",
            "google/gemini-flash-1.5-8b-exp",
            # 100K+ Context
            "deepseek/deepseek-r1-zero:free",
            "deepseek/deepseek-r1:free",
            "deepseek/deepseek-v3-base:free",
            "deepseek/deepseek-chat-v3-0324:free",
            "deepseek/deepseek-chat:free",
            "google/gemma-3-4b-it:free",
            "google/gemma-3-12b-it:free",
            "qwen/qwen2.5-vl-72b-instruct:free",
            "nvidia/llama-3.1-nemotron-70b-instruct:free",
            "meta-llama/llama-3.2-1b-instruct:free",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mistral-nemo:free",
            # 64K-100K Context
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "google/gemma-3-27b-it:free",
            "qwen/qwen2.5-vl-3b-instruct:free",
            "qwen/qwen-2.5-vl-7b-instruct:free",
            # 32K-64K Context
            "google/learnlm-1.5-pro-experimental:free",
            "qwen/qwq-32b:free",
            "google/gemini-2.0-flash-thinking-exp-1219:free",
            "bytedance-research/ui-tars-72b:free",
            "google/gemma-3-1b-it:free",
            "mistralai/mistral-small-24b-instruct-2501:free",
            "qwen/qwen-2.5-coder-32b-instruct:free",
            "qwen/qwen-2.5-72b-instruct:free",
            # 8K-32K Context
            "meta-llama/llama-3.2-3b-instruct:free",
            "qwen/qwq-32b-preview:free",
            "deepseek/deepseek-r1-distill-qwen-32b:free",
            "qwen/qwen2.5-vl-32b-instruct:free",
            "deepseek/deepseek-r1-distill-llama-70b:free",
            "qwen/qwen-2-7b-instruct:free",
            "google/gemma-2-9b-it:free",
            "mistralai/mistral-7b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "meta-llama/llama-3-8b-instruct:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            # 4K Context
            "huggingfaceh4/zephyr-7b-beta:free",
        ],
        "vision_models": [
            "google/gemini-2.0-pro-exp-02-05:free",
            "google/gemini-2.0-flash-thinking-exp:free",
            "google/gemini-2.0-flash-exp:free",
            "google/gemini-2.5-pro-exp-03-25:free",
            "google/gemini-flash-1.5-8b-exp",
            "qwen/qwen2.5-vl-72b-instruct:free",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "google/gemma-3-27b-it:free",
            "qwen/qwen2.5-vl-3b-instruct:free",
            "qwen/qwen-2.5-vl-7b-instruct:free",
            "bytedance-research/ui-tars-72b:free",
            "qwen/qwen2.5-vl-32b-instruct:free",
        ],
        "audio_models": [
            "google/gemini-2.0-flash-lite-001",
            "mistralai/voxtral-small-24b-2507",
            "google/gemini-2.5-flash-lite"
        ],
        "image_models": [
            "google/gemini-2.5-flash-image",
            "openai/gpt-5-image-mini",
            "google/gemini-3-pro-image-preview",
            "black-forest-labs/flux.2-pro",
            "black-forest-labs/flux.2-flex"
        ],
        "context_limits": {
            # 1M+ Context
            "google/gemini-2.0-pro-exp-02-05:free": 2000000,
            "google/gemini-2.0-flash-thinking-exp:free": 1048576,
            "google/gemini-2.0-flash-exp:free": 1048576,
            "google/gemini-2.5-pro-exp-03-25:free": 1000000,
            "google/gemini-flash-1.5-8b-exp": 1000000,
            # 100K+ Context
            "deepseek/deepseek-r1-zero:free": 163840,
            "deepseek/deepseek-r1:free": 163840,
            "deepseek/deepseek-v3-base:free": 131072,
            "deepseek/deepseek-chat-v3-0324:free": 131072,
            "deepseek/deepseek-chat:free": 131072,
            "google/gemma-3-4b-it:free": 131072,
            "google/gemma-3-12b-it:free": 131072,
            "qwen/qwen2.5-vl-72b-instruct:free": 131072,
            "nvidia/llama-3.1-nemotron-70b-instruct:free": 131072,
            "meta-llama/llama-3.2-1b-instruct:free": 131072,
            "meta-llama/llama-3.2-11b-vision-instruct:free": 131072,
            "meta-llama/llama-3.1-8b-instruct:free": 131072,
            "mistralai/mistral-nemo:free": 128000,
            # 64K-100K Context
            "mistralai/mistral-small-3.1-24b-instruct:free": 96000,
            "google/gemma-3-27b-it:free": 96000,
            "qwen/qwen2.5-vl-3b-instruct:free": 64000,
            "qwen/qwen-2.5-vl-7b-instruct:free": 64000,
            # 32K-64K Context
            "google/learnlm-1.5-pro-experimental:free": 40960,
            "qwen/qwq-32b:free": 40000,
            "google/gemini-2.0-flash-thinking-exp-1219:free": 40000,
            "bytedance-research/ui-tars-72b:free": 32768,
            "google/gemma-3-1b-it:free": 32768,
            "mistralai/mistral-small-24b-instruct-2501:free": 32768,
            "qwen/qwen-2.5-coder-32b-instruct:free": 32768,
            "qwen/qwen-2.5-72b-instruct:free": 32768,
            # 8K-32K Context
            "meta-llama/llama-3.2-3b-instruct:free": 20000,
            "qwen/qwq-32b-preview:free": 16384,
            "deepseek/deepseek-r1-distill-qwen-32b:free": 16000,
            "qwen/qwen2.5-vl-32b-instruct:free": 8192,
            "deepseek/deepseek-r1-distill-llama-70b:free": 8192,
            "qwen/qwen-2-7b-instruct:free": 8192,
            "google/gemma-2-9b-it:free": 8192,
            "mistralai/mistral-7b-instruct:free": 8192,
            "microsoft/phi-3-mini-128k-instruct:free": 8192,
            "meta-llama/llama-3-8b-instruct:free": 8192,
            "meta-llama/llama-3.3-70b-instruct:free": 8000,
            # 4K Context
            "huggingfaceh4/zephyr-7b-beta:free": 4096,
            # Audio/Image
            "google/gemini-2.0-flash-lite-001": 1000000,
            "mistralai/voxtral-small-24b-2507": 32768,
            "google/gemini-2.5-flash-lite": 1000000,
            "google/gemini-2.5-flash-image": 1000000,
            "openai/gpt-5-image-mini": 128000,
            "google/gemini-3-pro-image-preview": 1000000,
            "black-forest-labs/flux.2-pro": 4096,
            "black-forest-labs/flux.2-flex": 4096,
        }
    },
    
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_name": "GROQ",
        "badge": "🇺🇸 <b>US-Server</b>!",
        "chat_models": [
            "deepseek-r1-distill-llama-70b",
            "deepseek-r1-distill-qwen-32b",
            "gemma2-9b-it",
            "llama-3.1-8b-instant",
            "llama-3.2-1b-preview",
            "llama-3.2-3b-preview",
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview",
            "llama-3.3-70b-specdec",
            "llama-3.3-70b-versatile",
            "llama-guard-3-8b",
            "llama3-8b-8192",
            "llama3-70b-8192",
            "mistral-saba-24b",
            "qwen-2.5-32b",
            "qwen-2.5-coder-32b",
            "qwen-qwq-32b",
        ],
        "audio_models": [
            "distil-whisper-large-v3-en",
            "whisper-large-v3",
            "whisper-large-v3-turbo"
        ],
        "vision_models": [
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview"
        ],
        "context_limits": {
            "deepseek-r1-distill-llama-70b": 8192,
            "deepseek-r1-distill-qwen-32b": 8192,
            "gemma2-9b-it": 8192,
            "llama-3.1-8b-instant": 131072,
            "llama-3.2-1b-preview": 131072,
            "llama-3.2-3b-preview": 131072,
            "llama-3.2-11b-vision-preview": 131072,
            "llama-3.2-90b-vision-preview": 131072,
            "llama-3.3-70b-specdec": 131072,
            "llama-3.3-70b-versatile": 131072,
            "llama-guard-3-8b": 8192,
            "llama3-8b-8192": 8192,
            "llama3-70b-8192": 8192,
            "mistral-saba-24b": 32768,
            "qwen-2.5-32b": 32768,
            "qwen-2.5-coder-32b": 32768,
            "qwen-qwq-32b": 32768,
            "distil-whisper-large-v3-en": 16384,
            "whisper-large-v3": 16384,
            "whisper-large-v3-turbo": 16384,
        }
    },
    
    "Poe": {
        "base_url": "https://api.poe.com/v1",
        "key_name": "POE",
        "badge": "🌐 <b>US-Server</b>!",
        "chat_models": [
            "gpt-5.1-instant",
            "claude-sonnet-4.5",
            "gemini-3-pro",
            "gpt-5.1",
            "gpt-4o",
            "claude-3.5-sonnet",
            "deepseek-r1",
            "grok-4"
        ],
        "vision_models": [
            "claude-sonnet-4.5",
            "gpt-5.1",
            "gemini-3-pro",
            "gpt-4o",
            "claude-3.5-sonnet"
        ],
        "image_models": [
            "gpt-image-1",
            "flux-pro-1.1-ultra",
            "ideogram-v3",
            "dall-e-3",
            "playground-v3"
        ],
        "audio_models": [
            "elevenlabs-v3",
            "sonic-3.0"
        ],
        "video_models": [
            "kling-2.5-turbo-pro",
            "runway-gen-4-turbo",
            "veo-3.1"
        ],
        "supports_system": True,
        "supports_streaming": True,
        "context_limits": {
            "gpt-5.1-instant": 128000,
            "claude-sonnet-4.5": 200000,
            "gemini-3-pro": 2000000,
            "gpt-5.1": 128000,
            "gpt-4o": 128000,
            "claude-3.5-sonnet": 200000,
            "deepseek-r1": 163840,
            "grok-4": 131072,
            "gpt-image-1": 4096,
            "flux-pro-1.1-ultra": 4096,
            "ideogram-v3": 4096,
            "dall-e-3": 4096,
            "playground-v3": 4096,
            "elevenlabs-v3": 4096,
            "sonic-3.0": 4096,
            "kling-2.5-turbo-pro": 4096,
            "runway-gen-4-turbo": 4096,
            "veo-3.1": 4096,
        }
    },
    
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "key_name": "OPENAI",
        "badge": "🇺🇸 <b>US-Server</b>!",
        "chat_models": [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "o1-preview",
            "o1-mini"
        ],
        "vision_models": [
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "o1-preview",
            "o1-mini"
        ],
        "context_limits": {
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-0125": 16385,
            "gpt-3.5-turbo-instruct": 4096,
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "o1-preview": 128000,
            "o1-mini": 128000,
        }
    },
    
    "Cohere": {
        "base_url": "https://api.cohere.ai/v1",
        "key_name": "COHERE",
        "badge": "🇺🇸 <b>US-Server</b>!",
        "chat_models": [
            "command-r-plus-08-2024",
            "command-r-plus",
            "command-r-08-2024",
            "command-r",
            "command",
            "c4ai-aya-expanse-8b",
            "c4ai-aya-expanse-32b",
        ],
        "context_limits": {
            "command-r-plus-08-2024": 131072,
            "command-r-plus-04-2024": 131072,
            "command-r-plus": 131072,
            "command-r-08-2024": 131072,
            "command-r-03-2024": 131072,
            "command-r": 131072,
            "command": 4096,
            "command-nightly": 131072,
            "command-light": 4096,
            "command-light-nightly": 4096,
            "c4ai-aya-expanse-8b": 8192,
            "c4ai-aya-expanse-32b": 131072,
        }
    },
    
    "Together": {
        "base_url": "https://api.together.xyz/v1",
        "key_name": "TOGETHER",
        "badge": "🇺🇸 <b>US-Server</b>!",
        "chat_models": [
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        ],
        "vision_models": ["meta-llama/Llama-Vision-Free"],
        "context_limits": {
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": 131072,
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free": 8192,
            "meta-llama/Llama-Vision-Free": 8192,
            "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": 8192,
        }
    },
    
    "OVH": {
        "base_url": "https://llama-3-1-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1",
        "key_name": "OVH",
        "badge": "🇫🇷 <b>DSGVO-Konform</b>",
        "chat_models": [
            "ovh/codestral-mamba-7b-v0.1",
            "ovh/deepseek-r1-distill-llama-70b",
            "ovh/llama-3.1-70b-instruct",
            "ovh/llama-3.1-8b-instruct",
            "ovh/llama-3.3-70b-instruct",
            "ovh/mistral-7b-instruct-v0.3",
            "ovh/mistral-nemo-2407",
            "ovh/mixtral-8x7b-instruct",
            "ovh/qwen2.5-coder-32b-instruct",
        ],
        "vision_models": [
            "ovh/llava-next-mistral-7b",
            "ovh/qwen2.5-vl-72b-instruct"
        ],
        "context_limits": {
            "ovh/codestral-mamba-7b-v0.1": 131072,
            "ovh/deepseek-r1-distill-llama-70b": 8192,
            "ovh/llama-3.1-70b-instruct": 131072,
            "ovh/llama-3.1-8b-instruct": 131072,
            "ovh/llama-3.3-70b-instruct": 131072,
            "ovh/llava-next-mistral-7b": 8192,
            "ovh/mistral-7b-instruct-v0.3": 32768,
            "ovh/mistral-nemo-2407": 131072,
            "ovh/mixtral-8x7b-instruct": 32768,
            "ovh/qwen2.5-coder-32b-instruct": 32768,
            "ovh/qwen2.5-vl-72b-instruct": 131072,
        }
    },
    
    "Cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "key_name": "CEREBRAS",
        "badge": "🇺🇸 <b>US-Server</b>!",
        "chat_models": [
            "llama3.1-8b",
            "llama-3.3-70b"
        ],
        "context_limits": {
            "llama3.1-8b": 8192,
            "llama-3.3-70b": 8192,
        }
    },
    
    "GoogleAI": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "key_name": "GOOGLEAI",
        "badge": "🇺🇸 <b>US-Server</b>?",
        "chat_models": [
            "gemini-1.0-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-pro",
            "gemini-2.5-pro"
        ],
        "vision_models": [
            "gemini-1.5-pro",
            "gemini-1.0-pro",
            "gemini-1.5-flash",
            "gemini-2.0-pro",
            "gemini-2.5-pro"
        ],
        "context_limits": {
            "gemini-1.0-pro": 32768,
            "gemini-1.5-flash": 1000000,
            "gemini-1.5-pro": 1000000,
            "gemini-2.0-pro": 2000000,
            "gemini-2.5-pro": 2000000,
        }
    },
    
    "Anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "key_name": "ANTHROPIC",
        "badge": "🇺🇸 <b>US-Server</b>!",
        "chat_models": [
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20240307",
            "claude-3-opus-20240229",
        ],
        "vision_models": [
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20240307",
            "claude-3-opus-20240229",
        ],
        "context_limits": {
            "claude-3-7-sonnet-20250219": 128000,
            "claude-3-5-sonnet-20241022": 200000,
            "claude-3-5-haiku-20240307": 200000,
            "claude-3-5-sonnet-20240620": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
            "claude-3-sonnet-20240229": 200000,
        }
    },
    
    "HuggingFace": {
        "base_url": "https://api-inference.huggingface.co/models",
        "key_name": "HUGGINGFACE",
        "badge": "🌐 <b>US-Server</b>?",
        "chat_models": [
            "microsoft/phi-3-mini-4k-instruct",
            "microsoft/Phi-3-mini-128k-instruct",
            "HuggingFaceH4/zephyr-7b-beta",
            "deepseek-ai/DeepSeek-Coder-V2-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "microsoft/Phi-3.5-mini-instruct",
            "google/gemma-2-2b-it",
            "Qwen/Qwen2.5-7B-Instruct",
            "tiiuae/falcon-7b-instruct",
            "Qwen/QwQ-32B-preview",
        ],
        "vision_models": [
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/qwen2.5-vl-3b-instruct",
            "Qwen/qwen2.5-vl-32b-instruct",
            "Qwen/qwen2.5-vl-72b-instruct",
        ],
        "context_limits": {
            "microsoft/phi-3-mini-4k-instruct": 4096,
            "microsoft/Phi-3-mini-128k-instruct": 131072,
            "HuggingFaceH4/zephyr-7b-beta": 8192,
            "deepseek-ai/DeepSeek-Coder-V2-Instruct": 8192,
            "mistralai/Mistral-7B-Instruct-v0.3": 32768,
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": 32768,
            "microsoft/Phi-3.5-mini-instruct": 4096,
            "google/gemma-2-2b-it": 2048,
            "openai-community/gpt2": 1024,
            "microsoft/phi-2": 2048,
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2048,
            "Qwen/Qwen2.5-7B-Instruct": 131072,
            "tiiuae/falcon-7b-instruct": 8192,
            "Qwen/QwQ-32B-preview": 32768,
            "Qwen/Qwen2.5-VL-7B-Instruct": 64000,
            "Qwen/qwen2.5-vl-3b-instruct": 64000,
            "Qwen/qwen2.5-vl-32b-instruct": 8192,
            "Qwen/qwen2.5-vl-72b-instruct": 131072,
        }
    },
}

# ==========================================
# DEFAULTS
# ==========================================
DEFAULT_CHAT_PROVIDER = "Mistral"
DEFAULT_CHAT_MODEL = "mistral-large-latest"

# ==========================================
# YOUTUBE WHITELIST
# ==========================================
YOUTUBE_CHANNEL_WHITELIST = [
    # Format: Channel-Identifier (wird automatisch normalisiert)
    "AkademieKanal",           # /user/AkademieKanal
    "@theologisches-forum",     # /@theologisches-forum
    "grenzfragen",              # /grenzfragen
    
    # Wir können auch Channel-IDs verwenden (am sichersten):
    # "UCxyz123abc...",         # /channel/UCxyz123abc...
    
    # Weitere genehmigte Kanäle hier hinzufügen
]
# ==========================================
# PROMPT TEMPLATES
# ==========================================
TRANSCRIPT_PROMPTS = {
    "Veranstaltungsrückblick": """...""",
    # ... rest
}

# ==========================================
# UI CONSTANTS
# ==========================================
PWA_HEAD = f"""
<!-- Existing PWA meta tags -->
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#ffffff">

<!-- ✅ LOAD CUSTOM CSS FROM STATIC FILE -->
<link rel="stylesheet" href="/file=/var/www/transkript_app/static/custom.css">

<!-- Existing PWA manifest and icons -->
<link rel="manifest" href="/file=/var/www/transkript_app/static/manifest.json">
<link rel="apple-touch-icon" href="/file=/var/www/transkript_app/static/icon-192.png">
"""

CUSTOM_CSS = """

/* ==========================================
   ACCORDION COMPACTING
   ========================================== */

.block:has(> .label-wrap) {
    padding: 0 !important;
    margin-bottom: 0 !important;
    border: none !important;
    overflow: hidden !important;
}

.block > .label-wrap { 
    padding: 0px 8px !important; 
    margin: 0 !important;
    min-height: 32px !important; 
    height: 32px !important; 
    display: flex !important;
    align-items: center !important;
    background-color: transparent !important;
    border: none !important;
}

.block > .label-wrap > span {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 0.9rem !important;
}

.block > .label-wrap .icon {
    margin: 0 !important;
    transform: scale(0.8);
}

/* ==========================================
   HEADER ALIGNMENT
   ========================================== */

#user-status-row {
    justify-content: flex-end !important;
    text-align: right !important;
    padding-right: 10px !important;
}

/* ==========================================
   DESKTOP TWEAKS
   ========================================== */

label span { 
    font-size: 0.85rem !important; 
    font-weight: 600 !important; 
    margin-bottom: 2px !important;
    opacity: 1 !important; 
}

/* ==========================================
   BADGE STYLING
   ========================================== */

.badge-col .prose { 
    border: none !important; 
    background: transparent !important; 
    padding: 0 !important; 
    margin: 0 !important;
    box-shadow: none !important;
}

.badge-col { 
    border: none !important; 
    box-shadow: none !important; 
    background: transparent !important; 
}

.custom-badge {
    font-family: "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji", Arial, sans-serif !important;
    font-size: 0.85rem !important;
    line-height: 1.2 !important;
    white-space: nowrap !important; 
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0 10px;
    text-align: center;
    height: 42px; 
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    font-variant-emoji: emoji;
    -webkit-font-smoothing: antialiased;
}

.compact-row { 
    gap: 8px !important; 
    align-items: end !important; 
}

.compact-row .form { 
    border: none !important; 
    background: transparent !important; 
}

/* ==========================================
   MOBILE RESPONSIVE (< 768px)
   ========================================== */

@media (max-width: 768px) {
    
    /* ===================================
       GLOBAL SPACING REDUCTION
       =================================== */
    
    .gradio-container {
        padding: 0 !important;
        margin: 0 !important;
        width: 100vw !important;
        max-width: 100vw !important;
        overflow-x: hidden !important;
    }
    
    .contain {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    
    .gr-row {
        gap: 4px !important;
        margin: 2px 0 !important;
    }
    
    .gr-column {
        gap: 4px !important;
        padding: 2px !important;
    }
    
    /* ===================================
       COMPACT HEADER
       =================================== */
    
    .compact-header { 
        padding: 4px 8px !important; 
        min-height: 40px !important;
        gap: 4px !important;
    }
    
    .compact-header h3 { 
        margin: 0 !important; 
        font-size: 1.1rem !important; 
    }
    
    /* ===================================
       TABS: ICON ONLY (WORKING METHOD)
       =================================== */
    
    /* Target all tab buttons */
    .icon-nav > .tab-nav > button {
        padding: 10px 4px !important;
        min-width: 50px !important;
        height: 50px !important;
        font-size: 0 !important;  /* Hide text */
        overflow: hidden !important;
        text-overflow: clip !important;
        white-space: nowrap !important;
    }
    
    /* Show ONLY first character (the emoji) */
    .icon-nav > .tab-nav > button::first-letter {
        font-size: 1.5rem !important;
        line-height: 1 !important;
    }
    
    /* Selected tab styling */
    .icon-nav > .tab-nav > button.selected {
        border-bottom: 3px solid #2563eb !important;
        background: #f3f4f6 !important;
    }
    
    .dark .icon-nav > .tab-nav > button.selected {
        background: #374151 !important;
    }
    
    /* ===================================
       BUTTONS: ICON ONLY (WORKING METHOD)
       =================================== */
    
    /* Target buttons with mobile-icon-only class */
    .mobile-icon-only {
        min-width: 44px !important;
        height: 44px !important;
        padding: 8px !important;
        font-size: 0 !important;  /* Hide all text */
        overflow: hidden !important;
        text-overflow: clip !important;
        white-space: nowrap !important;
    }
    
    /* Show ONLY first character (the emoji) */
    .mobile-icon-only::first-letter {
        font-size: 1.3rem !important;
        line-height: 1 !important;
    }
    
    /* Special styling for primary send button */
    #btn-send { 
        background-color: #2563eb !important; 
    }
    
    #btn-send::first-letter { 
        color: white !important; 
    }
    
    /* Secondary button styling */
    .btn-secondary::first-letter { 
        color: #374151 !important; 
    }
    
    /* ===================================
       CHAT OPTIMIZATION
       =================================== */
    
    #chat_window { 
        height: 60vh !important; 
        max-height: 60vh !important;
        margin: 0 !important;
        padding: 4px !important;
    }
    
    /* ===================================
       FORM ELEMENTS COMPACT
       =================================== */
    
    input, textarea, select {
        padding: 6px 8px !important;
        font-size: 14px !important;
    }
    
    label {
        margin-bottom: 2px !important;
        font-size: 13px !important;
    }
    
    /* ===================================
       BADGE COMPACT
       =================================== */
    
    .custom-badge { 
        font-size: 0.7rem !important; 
        padding: 2px 6px !important; 
        height: 32px !important; 
    }
    
    /* ===================================
       HIDE FOOTER
       =================================== */
    
    footer, 
    .footer,
    .gradio-container footer { 
        display: none !important; 
    }
    
    /* ===================================
       COMPACT ACCORDIONS
       =================================== */
    
    .accordion-header {
        padding: 6px 8px !important;
        font-size: 14px !important;
    }
    
    .accordion-content {
        padding: 8px !important;
    }
}

/* ==========================================
   VERY SMALL SCREENS (< 400px)
   ========================================== */

@media (max-width: 400px) {
    .gr-row {
        gap: 2px !important;
    }
    
    .mobile-icon-only {
        min-width: 40px !important;
        height: 40px !important;
    }
    
    .mobile-icon-only::first-letter {
        font-size: 1.2rem !important;
    }
    
    .icon-nav > .tab-nav > button {
        min-width: 44px !important;
        padding: 8px 4px !important;
    }
    
    .icon-nav > .tab-nav > button::first-letter {
        font-size: 1.4rem !important;
    }
    
    .custom-badge { 
        font-size: 0.65rem !important; 
        padding: 2px 4px !important; 
        height: 28px !important; 
    }
}

"""

SENSITIVE_FILE_WARNING = """
⚠️ **DATENSCHUTZ-WARNUNG**

**Bitte KEINE der folgenden Dateien hochladen:**
- ❌ Personalakten oder Bewerbungen
- ❌ Seelsorge-Protokolle oder Beichtgespräche  
- ❌ Medizinische Unterlagen
- ❌ Dokumente mit Gesundheitsdaten
- ❌ Interne strategische Dokumente (Verschlusssachen)
- ❌ Passwörter oder Zugangsdaten

**Art. 9 DS-GVO:** Besondere Kategorien (Religion, Gesundheit) sind besonders zu schützen!

✅ **Erlaubt:** Öffentliche Texte, anonymisierte Daten, allgemeine Fachliteratur
"""