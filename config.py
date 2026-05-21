"""Configuration loading from .env and models.yaml."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = Path(__file__).parent


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_keys: list[str] = field(default_factory=list)
    rpm_limit: int = 0  # 0 = unlimited
    rpd_limit: int = 0  # 0 = unlimited

    def __post_init__(self) -> None:
        self._key_index: int = 0
        self.disabled_keys: list[str] = []  # auto-disabled by health checker

    @property
    def api_key(self) -> str:
        """Current active API key (round-robin)."""
        if not self.api_keys:
            return ""
        return self.api_keys[self._key_index % len(self.api_keys)]

    @api_key.setter
    def api_key(self, value: str) -> None:
        """Allow setting a single key (backward compatibility)."""
        if value:
            if not self.api_keys:
                self.api_keys = [value]
            elif value not in self.api_keys:
                self.api_keys[self.active_key_index] = value

    @property
    def active_key_index(self) -> int:
        return self._key_index % len(self.api_keys) if self.api_keys else 0

    @property
    def total_keys(self) -> int:
        return len(self.api_keys)

    def rotate_key(self) -> str:
        """Advance to next key (round-robin) and return it."""
        if self.api_keys:
            self._key_index = (self._key_index + 1) % len(self.api_keys)
        return self.api_key


@dataclass
class ModelFallback:
    provider: str
    model: str
    enabled: bool = True
    rpm_limit: int = 0  # 0 = use provider default
    rpd_limit: int = 0
    tpm_limit: int = 0
    tpd_limit: int = 0


@dataclass
class ModelCapabilities:
    supports_tools: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True


@dataclass
class ModelConfig:
    unified_name: str
    fallbacks: list[ModelFallback] = field(default_factory=list)
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    monthly_token_budget: int = 0  # 0 = unlimited
    intelligence_rank: int = 0  # 0 = unknown, higher = smarter
    speed_rank: int = 0  # 0 = unknown, higher = faster
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig:
    master_key: str
    host: str
    port: int
    default_rpm_limit: int
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    models: dict[str, ModelConfig] = field(default_factory=dict)


# Provider definitions: name -> (env_key_for_api_key, base_url_template)
PROVIDER_DEFS: dict[str, tuple[str, str]] = {
    # ── Original providers ──
    "openrouter": ("OPENROUTER_KEY", "https://openrouter.ai/api/v1"),
    "github": ("GITHUB_KEY", "https://models.inference.ai.azure.com"),
    "groq": ("GROQ_KEY", "https://api.groq.com/openai/v1"),
    "cerebras": ("CEREBRAS_KEY", "https://api.cerebras.ai/v1"),
    "cloudflare": (
        "CLOUDFLARE_KEY",
        "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
    ),
    "huggingface": ("HUGGINGFACE_KEY", "https://api-inference.huggingface.co/models"),
    "nvidia": ("NVIDIA_KEY", "https://integrate.api.nvidia.com/v1"),
    "siliconflow": ("SILICONFLOW_KEY", "https://api.siliconflow.cn/v1"),
    "cohere": ("COHERE_KEY", "https://api.cohere.com/v2"),
    "google_gemini": ("GOOGLE_GEMINI_KEY", "https://generativelanguage.googleapis.com/v1beta"),
    "mistral": ("MISTRAL_KEY", "https://api.mistral.ai/v1"),
    "kilo": ("KILO_KEY", "https://api.kilo.ai/api/gateway"),
    "llm7": ("LLM7_KEY", "https://api.llm7.io/v1"),
    "ollama": ("OLLAMA_KEY", "https://api.ollama.com"),
    # ── New providers (from freellmapi comparison) ──
    "deepseek": ("DEEPSEEK_KEY", "https://api.deepseek.com/v1"),
    "together": ("TOGETHER_KEY", "https://api.together.xyz/v1"),
    "fireworks": ("FIREWORKS_KEY", "https://api.fireworks.ai/inference/v1"),
    "sambanova": ("SAMBANOVA_KEY", "https://api.sambanova.ai/v1"),
    "chutes": ("CHUTES_KEY", "https://chutes.ai/app/api/v1"),
    "anthropic": ("ANTHROPIC_KEY", "https://api.anthropic.com"),
    "openai": ("OPENAI_KEY", "https://api.openai.com/v1"),
    "perplexity": ("PERPLEXITY_KEY", "https://api.perplexity.ai"),
    "xai": ("XAI_KEY", "https://api.x.ai/v1"),
    "novita": ("NOVITA_KEY", "https://api.novita.ai/v3/openai"),
}

# Providers that use OpenAI-compatible chat/completions endpoints
OPENAI_COMPATIBLE = {
    "openrouter", "github", "groq", "cerebras", "nvidia",
    "siliconflow", "mistral", "llm7", "ollama",
    "deepseek", "together", "fireworks", "sambanova", "chutes",
    "openai", "perplexity", "xai", "novita",
}

# Providers needing special request formatting
SPECIAL_PROVIDERS = {
    "cloudflare", "huggingface", "cohere", "google_gemini", "kilo",
    "anthropic",
}


def _load_provider_keys(env_key: str) -> list[str]:
    """Load API keys for a provider.

    Supports:
      - Comma-separated: OPENROUTER_KEY="k1,k2,k3"
      - Indexed: OPENROUTER_KEY_1="k1", OPENROUTER_KEY_2="k2"
    """
    keys: list[str] = []
    # Comma-separated keys
    raw = os.environ.get(env_key, "")
    if raw:
        keys.extend(k.strip() for k in raw.split(",") if k.strip())
    # Indexed keys (KEY_1, KEY_2, ... up to KEY_10)
    for i in range(1, 11):
        indexed = os.environ.get(f"{env_key}_{i}", "")
        if indexed and indexed not in keys:
            keys.append(indexed.strip())
    return keys


def _load_providers() -> dict[str, ProviderConfig]:
    providers: dict[str, ProviderConfig] = {}
    for name, (env_key, base_url_tpl) in PROVIDER_DEFS.items():
        api_keys = _load_provider_keys(env_key)
        base_url = base_url_tpl
        if "{account_id}" in base_url:
            account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
            base_url = base_url.replace("{account_id}", account_id)
        providers[name] = ProviderConfig(
            name=name,
            base_url=base_url,
            api_keys=api_keys,
            rpm_limit=int(os.environ.get("DEFAULT_RPM_LIMIT", "0")),
        )
    return providers


def _parse_rate_limit(rate_str: str) -> dict[str, int]:
    """Parse a rate limit string like '30 RPM, 14,400 RPD' into structured limits.

    Handles: RPM, RPD, TPM, TPD, RPS (×60), req/hr (÷60), K/M suffixes,
    tilde prefix, commas in numbers, parenthetical notes.
    Returns dict with rpm/rpd/tpm/tpd keys (0 for unparsed).
    """
    import re

    result: dict[str, int] = {"rpm": 0, "rpd": 0, "tpm": 0, "tpd": 0}
    if not rate_str:
        return result

    # Remove parenthetical notes and semicolon sections
    cleaned = rate_str.split("(")[0].split(";")[0].strip()
    # Remove tilde prefixes
    cleaned = cleaned.replace("~", "")

    # Match patterns like "30 RPM", "14,400 RPD", "500K TPM", "1M TPD", "1 RPS", "200 req/hr"
    pattern = re.compile(
        r"([\d,]+\.?\d*)\s*([KM]?)\s*(RPM|RPD|TPM|TPD|RPS|REQ/HR|REQ/H)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(cleaned):
        num_str = m.group(1).replace(",", "")
        suffix = m.group(2).upper()
        unit = m.group(3).upper()

        multiplier = 1
        if suffix == "K":
            multiplier = 1_000
        elif suffix == "M":
            multiplier = 1_000_000

        try:
            value = int(float(num_str) * multiplier)
        except (ValueError, TypeError):
            continue

        if unit == "RPS":
            result["rpm"] = max(result["rpm"], value * 60)
        elif unit == "RPM":
            result["rpm"] = max(result["rpm"], value)
        elif unit == "RPD":
            result["rpd"] = max(result["rpd"], value)
        elif unit == "TPM":
            result["tpm"] = max(result["tpm"], value)
        elif unit == "TPD":
            result["tpd"] = max(result["tpd"], value)
        elif unit in ("REQ/HR", "REQ/H"):
            result["rpm"] = max(result["rpm"], max(1, value // 60))

    return result


def _load_models() -> dict[str, ModelConfig]:
    models_file = BASE_DIR / "models.yaml"
    if not models_file.exists():
        return {}

    with open(models_file) as f:
        data = yaml.safe_load(f) or {}

    models: dict[str, ModelConfig] = {}
    for model_name, model_data in data.get("models", {}).items():
        if isinstance(model_data, list):
            # Old format: list of fallbacks directly
            fb_list = [
                ModelFallback(provider=fb["provider"], model=fb["model"])
                for fb in model_data
            ]
            capabilities = ModelCapabilities()
            meta = {}
        elif isinstance(model_data, dict):
            # New format: {capabilities: {...}, fallbacks: [...], _meta: {...}}
            fb_list = [
                ModelFallback(provider=fb["provider"], model=fb["model"])
                for fb in model_data.get("fallbacks", [])
            ]
            caps_data = model_data.get("capabilities", {})
            capabilities = ModelCapabilities(
                supports_tools=caps_data.get("supports_tools", False),
                supports_vision=caps_data.get("supports_vision", False),
                supports_streaming=caps_data.get("supports_streaming", True),
            )
            meta = model_data.get("_meta", {})
            # Parse rate limits from _meta into each fallback
            rate_limits = _parse_rate_limit(meta.get("rate_limit", ""))
            if any(rate_limits.values()):
                fb_list = [
                    ModelFallback(
                        provider=fb.provider,
                        model=fb.model,
                        rpm_limit=rate_limits["rpm"] or fb.rpm_limit,
                        rpd_limit=rate_limits["rpd"] or fb.rpd_limit,
                        tpm_limit=rate_limits["tpm"] or fb.tpm_limit,
                        tpd_limit=rate_limits["tpd"] or fb.tpd_limit,
                    )
                    for fb in fb_list
                ]
        else:
            continue
        # Get explicit ranks from YAML, or infer from model name
        explicit_intel = int(model_data.get("intelligence_rank", 0)) if isinstance(model_data, dict) else 0
        explicit_speed = int(model_data.get("speed_rank", 0)) if isinstance(model_data, dict) else 0
        inferred_intel, inferred_speed = _infer_ranks(model_name)
        models[model_name] = ModelConfig(
            unified_name=model_name,
            fallbacks=fb_list,
            capabilities=capabilities,
            monthly_token_budget=int(model_data.get("monthly_token_budget", 0)) if isinstance(model_data, dict) else 0,
            intelligence_rank=explicit_intel or inferred_intel,
            speed_rank=explicit_speed or inferred_speed,
            meta=meta if isinstance(meta, dict) else {},
        )
    return models


def _infer_ranks(model_name: str) -> tuple[int, int]:
    """Infer intelligence and speed ranks from model name heuristics.

    Returns (intelligence_rank, speed_rank). Both 0-10 scale.
    Higher intelligence = smarter. Higher speed = faster inference.
    """
    name = model_name.lower()

    # ── Intelligence ranks ──
    # Tier 1: Top reasoning models (9-10)
    if any(k in name for k in ("o3-mini", "o4-mini", "deepseek-r1-0528", "qwq", "qwen3-coder")):
        return 10, 5
    if any(k in name for k in ("gpt-4.1-", "gpt-4o", "deepseek-r1", "qwen3.5-35b")):
        return 9, 5
    if any(k in name for k in ("gpt-4.1:", "deepseek-v3", "kimi-k2", "qwen3-next")):
        return 9, 4

    # Tier 2: Strong large models (7-8)
    if any(k in name for k in ("nemotron-super", "nemotron-ultra", "hermes-3-405b")):
        return 8, 3
    if any(k in name for k in ("llama-3.3-70b", "llama-4-maverick", "qwen3-32b", "qwen-3-235b")):
        return 7, 4
    if any(k in name for k in ("gemma-4-31b", "mistral-large", "command-a-", "qwen2.5-72b")):
        return 7, 4
    if any(k in name for k in ("llama-4-scout", "minimax-m2", "pixtral-large")):
        return 7, 5

    # Tier 3: Mid-range models (5-6)
    if any(k in name for k in ("gemma-4-26b", "gemma-3-27b", "mistral-medium", "command-r-plus")):
        return 6, 6
    if any(k in name for k in ("mistral-small-3", "qwen2.5-coder", "codestral")):
        return 6, 7
    if any(k in name for k in ("dolphin-mistral-24b", "gpt-oss-120b", "glm-4.5")):
        return 5, 5

    # Tier 4: Small/fast models (3-4)
    if any(k in name for k in ("llama-3.1-8b", "gemma-3-12b", "command-r-", "open-mistral-nemo")):
        return 4, 8
    if any(k in name for k in ("mistral-7b", "phi-3.5", "glm-4", "qwen2.5-7b", "qwen3-8b")):
        return 4, 8
    if any(k in name for k in ("nemotron-nano", "gpt-oss-20b")):
        return 4, 8

    # Tier 5: Tiny models (2)
    if any(k in name for k in ("gemma-3-4b", "command-r7b", "gemma-3-1b")):
        return 2, 9

    # ── Catch remaining by model family ──
    if "gemini-2.5-flash" in name:
        return 8, 7
    if "gpt-4.1" in name:
        return 9, 5
    if "mistral-small" in name:
        return 5, 8
    if "llama3.1" in name or "llama-3.1" in name:
        return 4, 8
    if "glm-4" in name or "glm-4." in name:
        return 5, 7
    if "ling-" in name:
        return 4, 8
    # Cloudflare worker models (fast inference)
    if name.startswith("@cf/"):
        return 5, 8

    # Named provider/model patterns
    if "mixtral-8x7b" in name:
        return 6, 7
    if "nemotron-3-super-120b" in name:
        return 8, 4
    if "nemotron-3-nano" in name:
        return 3, 9
    if "qwen3.5-27b" in name:
        return 7, 6
    if "qwen3.6-plus" in name:
        return 7, 6
    if "deepseek-chat-v3" in name:
        return 8, 5
    if "deepseek-ocr" in name:
        return 6, 5
    if "devstral" in name:
        return 5, 8
    if "grok-code-fast" in name:
        return 6, 9
    if "dola-seed" in name or "trinity-large" in name:
        return 7, 5

    # ── Speed overrides for known fast providers ──
    # Cerebras = extremely fast
    if "cerebras" in name:
        return 4, 10
    # Groq = very fast
    if "groq" in name:
        return 4, 9

    # Default: unknown
    return 0, 0


def load_config() -> AppConfig:
    return AppConfig(
        master_key=os.environ.get("MASTER_KEY", ""),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        default_rpm_limit=int(os.environ.get("DEFAULT_RPM_LIMIT", "0")),
        providers=_load_providers(),
        models=_load_models(),
    )


async def discover_models(
    client: httpx.AsyncClient, config: AppConfig
) -> dict[str, ModelConfig]:
    """Query each provider with an API key for available models and merge into config.

    Models from models.yaml take priority. Auto-discovered models are added only
    if no manual definition exists. Returns the set of newly discovered models.
    """
    discovered: dict[str, ModelConfig] = {}
    existing_names = set(config.models.keys())

    for name, provider in config.providers.items():
        if not provider.api_key:
            continue

        models = await _fetch_provider_models(client, provider)
        if not models:
            continue

        for m in models:
            model_id = m.get("id", "")
            if not model_id or model_id in existing_names:
                continue

            # Filter: only include free models per provider
            if not _is_free_model(name, model_id, m):
                continue

            # Create a unified model name: provider/model-id
            unified = model_id
            fb = ModelFallback(provider=name, model=model_id)
            discovered[unified] = ModelConfig(
                unified_name=unified,
                fallbacks=[fb],
            )
            existing_names.add(unified)

    # Merge into config (manual definitions always win)
    for unified_name, model_cfg in discovered.items():
        if unified_name not in config.models:
            config.models[unified_name] = model_cfg

    logger.info(
        "Auto-discovered %d new models from %d providers",
        len(discovered),
        sum(1 for p in config.providers.values() if p.api_key),
    )
    return discovered


def _is_free_model(provider: str, model_id: str, model_data: dict) -> bool:
    """Check if a model is available on a free tier.
    
    Rules per provider:
    - OpenRouter: must have ':free' suffix
    - NVIDIA: all models free (free developer tier)
    - Groq: all models free (free tier)
    - Cerebras: all models free (free tier)
    - GitHub Models: all models free (free tier)
    - Mistral: check pricing field or skip (experiment plan)
    - Cohere: trial plan (1000 calls/month)
    - Google Gemini: free tier models only
    - Others: include by default
    """
    # OpenRouter: only free models have ':free' suffix
    if provider == "openrouter":
        return model_id.endswith(":free")
    
    # These providers offer all listed models on free tiers
    FREE_PROVIDERS = {"nvidia", "groq", "cerebras", "github", "llm7", "ollama"}
    if provider in FREE_PROVIDERS:
        return True
    
    # Mistral: experiment plan includes these models
    if provider == "mistral":
        return True  # All listed on /v1/models are on experiment plan
    
    # Cohere: trial key gives 1000 calls/month
    if provider == "cohere":
        return True
    
    # Google Gemini: Flash/Lite models are free, Pro may have limits
    if provider == "google_gemini":
        model_name = model_id.lower()
        return "flash" in model_name or "lite" in model_name or "pro" in model_name
    
    # SiliconFlow: permanently free models
    if provider == "siliconflow":
        return True  # Free models listed on their free tier
    
    # Cloudflare Workers AI: all free tier
    if provider == "cloudflare":
        return True
    
    # Default: include
    return True


async def _fetch_provider_models(
    client: httpx.AsyncClient, provider: ProviderConfig
) -> list[dict[str, Any]]:
    """Fetch available models from a provider's /models endpoint."""
    try:
        headers = {"Authorization": f"Bearer {provider.api_key}"}

        if provider.name in OPENAI_COMPATIBLE:
            url = f"{provider.base_url}/models"
        elif provider.name == "kilo":
            url = f"{provider.base_url}/v1/models"
        elif provider.name == "cloudflare":
            url = f"{provider.base_url}/models"
        elif provider.name == "google_gemini":
            url = f"{provider.base_url}/models?key={provider.api_key}"
            headers = {}
        elif provider.name == "cohere":
            url = f"{provider.base_url}/models"
        else:
            return []

        resp = await client.get(url, headers=headers, timeout=15.0)
        if resp.status_code >= 400:
            logger.debug("Provider %s models endpoint returned %d", provider.name, resp.status_code)
            return []

        data = resp.json()

        # OpenAI-compatible format: {"data": [...]}
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        # Gemini format: {"models": [...]}
        if isinstance(data, dict) and "models" in data:
            return data["models"]
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        logger.debug("Failed to fetch models from %s: %s", provider.name, e)
        return []
