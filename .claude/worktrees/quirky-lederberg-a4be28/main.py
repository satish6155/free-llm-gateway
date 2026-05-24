"""Free LLM Gateway — unified OpenAI-compatible API server."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from benchmark import BenchmarkRunner
from cache import response_cache
from config import discover_models, load_config
from custom_combos import combo_manager
from gateway_auth import GatewayAuthManager
from health import HealthChecker
from key_encryptor import EncryptedKeyStore
from key_manager import KeyManager
from oauth_manager import oauth_manager
from providers.base import has_tool_calling  # noqa: F401 — used by routes
from quota_tracker import quota_tracker
from rate_limiter import RateLimiter
from rate_tracker import PROVIDER_FREE_LIMITS, per_key_rate_tracker
from request_db import request_db
from request_queue import request_queue
from router import Router
from smart_default import SmartDefault
from smart_router import SmartRouter
from sticky_sessions import sticky_sessions
from tracking import usage_tracker

from routes import admin, analytics, chat, fallbacks, health, keys, management, other
from routes.state import init_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Shared instances ──────────────────────────────────────────────────────────

config = load_config()
rate_limiter = RateLimiter()
health_checker = HealthChecker()
router = Router(config, rate_limiter, health_checker)
key_manager = KeyManager(config.master_key or "default-key")
smart_router = SmartRouter(config.models)
benchmark_runner = BenchmarkRunner(config)
benchmark_results = benchmark_runner.get_results()
smart_default = SmartDefault(config.models, benchmark_results)
encrypted_key_store = EncryptedKeyStore(config.master_key or "default-key")

# Gateway auth — initialise with master key if available
if config.master_key:
    import gateway_auth as _gw_mod
    _gw_mod.gateway_auth = GatewayAuthManager(encryption_key=config.master_key)
    gateway_auth = _gw_mod.gateway_auth
else:
    gateway_auth = None

# Populate route module shared state immediately so imports resolve at request time
init_state(
    config=config,
    rate_limiter=rate_limiter,
    health_checker=health_checker,
    router=router,
    key_manager=key_manager,
    smart_router=smart_router,
    benchmark_runner=benchmark_runner,
    smart_default=smart_default,
    encrypted_key_store=encrypted_key_store,
    gateway_auth=gateway_auth,
    client=None,  # set in lifespan
    response_cache=response_cache,
    request_queue=request_queue,
    usage_tracker=usage_tracker,
    combo_manager=combo_manager,
    quota_tracker=quota_tracker,
    oauth_manager=oauth_manager,
    sticky_sessions=sticky_sessions,
    benchmark_results=benchmark_results,
)

# ── Lifespan ──────────────────────────────────────────────────────────────────

_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = httpx.AsyncClient(http2=True, follow_redirects=True)

    # Expose the httpx client to route modules
    from routes import state as _st
    _st.client = _client

    # Auto-discover models from providers
    try:
        discovered = await discover_models(_client, config)
        if discovered:
            logger.info(
                "Discovered %d models: %s",
                len(discovered),
                ", ".join(list(discovered.keys())[:10]),
            )
    except Exception as e:
        logger.warning("Model discovery failed: %s", e)

    # Start background health checks
    health_checker.start(_client, config.providers)
    await health_checker.check_all(_client, config.providers)

    # Sync key_manager keys into config providers
    _st._sync_keys_to_config()

    # Merge custom combos into model config
    combo_manager.merge_into_config(config.models)

    # Initialize per-key rate limits from known free tier limits
    for prov_name, model_limits in PROVIDER_FREE_LIMITS.items():
        for model_key, limits in model_limits.items():
            provider = config.providers.get(prov_name)
            if provider and provider.api_key:
                per_key_rate_tracker.set_limits(
                    prov_name, model_key, provider.api_key,
                    rpm=limits.rpm, rpd=limits.rpd,
                    tpm=limits.tpm, tpd=limits.tpd,
                )

    # Start sticky session cleanup loop
    sticky_sessions.start_cleanup_loop()

    # Initialize SQLite request log
    request_db.init()

    # Start request queue workers
    request_queue.set_router(router)
    await request_queue.start_workers(num_workers=3)

    logger.info(
        "Gateway started — %d models, %d providers configured",
        len(config.models),
        sum(1 for p in config.providers.values() if p.api_key),
    )
    yield

    # Shutdown
    await request_queue.stop_workers()
    await sticky_sessions.stop_cleanup()
    usage_tracker.flush()
    request_db.close()
    await health_checker.stop()
    if _client:
        await _client.aclose()


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(title="Free LLM Gateway", version="1.0.0", lifespan=lifespan)

app.include_router(chat.router)
app.include_router(health.router)
app.include_router(keys.router)
app.include_router(analytics.router)
app.include_router(fallbacks.router)
app.include_router(admin.router)
app.include_router(management.router)
app.include_router(other.router)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level="info",
    )
