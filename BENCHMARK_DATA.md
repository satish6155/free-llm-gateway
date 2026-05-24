# Free LLM Gateway — Model Benchmark Data

> **Last updated**: May 24, 2026
> **Sources**: LMSYS Chatbot Arena, OpenAI/Google/Meta model cards, Artificial Analysis, HuggingFace Open LLM Leaderboard, provider docs
> **Legend**: `~` = approximate from aggregated data. `[E]` = extrapolated from model family trends. `?` = unknown/insufficient data.

---

## Quick Navigation

- [Quality Benchmarks by Model Family](#quality-benchmarks-by-model-family)
- [Provider Speed Benchmarks](#provider-speed-benchmarks)
- [Unified Comparison Table](#unified-comparison-table)
- [Data Confidence Levels](#data-confidence-levels)
- [Sources](#sources)

---

## Quality Benchmarks by Model Family

### OpenAI Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **GPT-5** | ~1420-1460 [E] | ~92-94% [E] | ~95%+ [E] | ~92%+ [E] | ~68% [E] | ~200K-1M | GitHub Models | 10 RPM, 50 RPD |
| **GPT-4.1** | ~1350-1370 | 90.2% | 92.0% | 84.0% | 60.0% | 1,047,576 | GitHub Models | 10 RPM, 50 RPD |
| **o4-mini** | ~1370-1400 | ~88% | ~92% | ~91% | ~63% | 200,000 | GitHub Models | 10 RPM, 50 RPD |
| **o3-mini** | ~1320-1340 | 86.5% | 91.0% | 87.3% | 59.0% | 200,000 | GitHub Models | 10 RPM, 50 RPD |
| **GPT-4o** | 1287 | 88.7% | 90.2% | 76.6% | 53.6% | 128,000 | GitHub Models | 10 RPM, 50 RPD |
| **GPT-4o-mini** | 1177 | 82.0% | 87.2% | 70.2% | 40.2% | 128,000 | llm7.io | 30 RPM (120 w/token) |
| **GPT-4.1-mini** | ~1250-1280 | 85.2% | 87.5% | 77.0% | 52.0% | 1,047,576 | GitHub Models | 15 RPM, 150 RPD |
| **GPT-OSS-120B** | ~1320-1360 [E] | ~88-90% [E] | ~90% [E] | ~82% [E] | ~56% [E] | ~128K-256K | OpenRouter, Cerebras, Groq | 20 RPM, 200 RPD (OR) |
| **GPT-OSS-20B** | ~1180-1240 [E] | ~80-84% [E] | ~82% [E] | ~70% [E] | ~45% [E] | ~128K | OpenRouter | 20 RPM, 200 RPD |

### Google Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **Gemini 3 Flash (Preview)** | ~1350-1400 [E] | ~89-91% [E] | ~91% [E] | ~88% [E] | ~62% [E] | ~1M | Google Gemini API | Preview limits |
| **Gemini 2.5 Pro** | ~1380-1410 | 90.0% | 92.0% | 90.2% | 65.0% | 1,048,576 | Google Gemini API | 5 RPM, 100 RPD |
| **Gemini 2.5 Flash** | ~1300-1340 | 88.0% | 89.0% | 83.0% | 57.0% | 1,048,576 | Google Gemini API | 10 RPM, 250 RPD |
| **Gemini 2.5 Flash-Lite** | ~1200-1240 [E] | ~82-85% [E] | ~83-86% [E] | ~72-76% [E] | ~45% [E] | 1,048,576 | Google Gemini API, llm7.io | 15 RPM, 1,000 RPD |
| **Gemma 4 31B** | ~1230-1280 [E] | ~84-87% [E] | ~80% [E] | ~65% [E] | ~42% [E] | ~256K-1M | OpenRouter, NVIDIA | 20 RPM, 200 RPD (OR) |
| **Gemma 4 26B** | ~1200-1260 [E] | ~82-85% [E] | ~78% [E] | ~62% [E] | ~40% [E] | ~256K | OpenRouter, NVIDIA, Cloudflare | 20 RPM, 200 RPD (OR) |
| **Gemma 3 27B** | ~1170-1210 | 80.5% | ~76% | ~60% | ~35% | 128,000 | OpenRouter | 20 RPM, 200 RPD |
| **Gemma 3 12B** | ~1100-1140 | 73.5% | ~65% | ~48% | ~30% | 128,000 | OpenRouter | 20 RPM, 200 RPD |
| **Gemma 3 4B** | ~1020-1060 | 62.5% | ~52% | ~35% | ~22% | 128,000 | OpenRouter | 20 RPM, 200 RPD |

### xAI Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **Grok-4.3** | ~1380-1420 [E] | ~90-92% [E] | ~92% [E] | ~90%+ [E] | ~65% [E] | ~1M | xAI API | Credit-based |
| **Grok-4.1-fast** | ~1340-1380 [E] | ~88-90% [E] | ~90%+ [E] | ~86-89% [E] | ~60% [E] | ~2M | xAI API | Credit-based |
| **Grok-3-mini** | ~1270-1320 | ~85% | ~86% | ~85% | ~55% | 131,072 | xAI API | Credit-based |

### DeepSeek Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **DeepSeek-R1-0528** | ~1370-1400 [E] | ~91-92% [E] | ~93% [E] | ~85%+ [E] | ~73%+ [E] | ~128K | OpenRouter, llm7.io, DeepSeek API | 20 RPM (OR), Dynamic (API) |
| **DeepSeek-R1** | ~1350-1380 | 90.8% | 91.6% | 79.8% | 71.5% | 128,000 | OpenRouter, Groq, GitHub, NVIDIA NIM, Nebius, Nscale, OVHcloud | Varies |
| **DeepSeek-R1-Distill-70B** | ~1280-1320 | 86.7% | 86.5% | 70.0% | 57.0% | 128,000 | Groq, GitHub, NVIDIA NIM, OVHcloud | Varies |
| **DeepSeek-V3.2 / deepseek-chat** | ~1340-1380 [E] | ~90%+ [E] | ~85% [E] | ~80% [E] | ~62% [E] | ~128K | DeepSeek API | Dynamic |
| **DeepSeek-V3.1** | ~1320-1360 [E] | ~89-90% [E] | ~83% [E] | ~81% [E] | ~60% [E] | ~128K | Ollama Cloud | Session limits |
| **DeepSeek-V3** | ~1300-1340 | 88.5% | 82.6% | 79.2% | 59.1% | 128,000 | OpenRouter, NVIDIA NIM, llm7.io | Varies |
| **DeepSeek-R1-Distill-Qwen-7B** | ~1060-1100 | 73.2% | 64.0% | 52.9% | 30.0% | 32K-128K | SiliconFlow, Cloudflare | 1,000 RPM (SF) |
| **DeepSeek-R1-Distill-Qwen-32B** | ~1160-1200 [E] | ~82% [E] | ~78% [E] | ~65% [E] | ~45% [E] | 32K | Cloudflare | 10K neurons/day |

### Qwen Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **Qwen3-Coder-480B** | ~1350-1400 [E] | ~88-90% [E] | ~93%+ [E] | ~85% [E] | ~58% [E] | ~128K | OpenRouter, NVIDIA | 20 RPM (OR) |
| **Qwen3-Max** | ~1350-1380 | ~89-91% [E] | ~89% | ~86% | ~58% [E] | 128K | Alibaba Cloud | Tiered |
| **Qwen3-235B-A22B** | ~1330-1370 | 89.5% | 89.0% | 88.0% | 60.0% | 131,072 | OpenRouter, Cerebras, Nebius | 20 RPM (OR) |
| **Qwen3.6-Plus** | ~1300-1350 [E] | ~87-89% [E] | ~87% [E] | ~82% [E] | ~55% [E] | ~131K | OpenRouter | 20 RPM, 200 RPD |
| **QwQ-Plus** | ~1300-1340 | ~86% | ~84% | ~85% | ~55% | 131K | Alibaba Cloud | Tiered |
| **Qwen3-Plus** | ~1280-1320 | ~86-88% [E] | ~84% [E] | ~80-83% [E] | ~52% [E] | ~131K | Alibaba Cloud | Tiered |
| **Qwen3-VL-Plus** | ~1300-1340 [E] | ~87-89% [E] | ~85% [E] | ~80% [E] | ~53% [E] | 128K | Alibaba Cloud | Tiered |
| **Qwen3-Coder-Plus** | ~1300-1350 [E] | ~86% [E] | ~90%+ | ~82% [E] | ~54% [E] | 256K | Alibaba Cloud | Tiered |
| **Qwen3-Coder-30B** | ~1240-1280 [E] | ~82-85% [E] | ~90%+ [E] | ~78% [E] | ~48% [E] | 262K | NVIDIA NIM, OVHcloud, Nscale | Varies |
| **Qwen3.5-27B** | ~1220-1270 [E] | ~83-86% [E] | ~82% [E] | ~75-80% [E] | ~46% [E] | ~131K | ModelScope | 2,000 RPD |
| **Qwen3.5-35B-A3B** | ~1200-1250 [E] | ~82-85% [E] | ~80% [E] | ~74% [E] | ~44% [E] | ~131K | ModelScope | 2,000 RPD |
| **Qwen3-32B** | ~1260-1300 | 86.5% | 84.0% | 82.0% | 50.0% | 131,072 | Groq, Cerebras | 30 RPM (Groq) |
| **Qwen3-8B** | ~1110-1150 | 76.5% | 72.0% | 62.0% | ~35% | 131,072 | SiliconFlow | 1,000 RPM, 50K TPM |
| **Qwen2.5-72B** | ~1190-1220 | 85.8% | 80.0% | 72.0% | 42.0% | 131,072 | NVIDIA NIM | ~40 RPM |
| **Qwen2.5-7B** | ~1060-1090 | 74.3% | 68.0% | 54.0% | ~30% | 131,072 | HuggingFace | ~1,000 RPD |
| **Qwen2.5-Coder-32B** | ~1190-1240 [E] | ~80% [E] | ~85% | ~70% [E] | ~38% [E] | ~131K | llm7.io | 30 RPM (120 w/token) |

### Meta / Llama Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **Llama 4 Maverick (17B-128E)** | ~1237 | 84.4% | 85.0% | 62.0% | 55.0% | 1M | OpenRouter, GitHub | 20 RPM (OR), 10 RPM (GitHub) |
| **Llama 3.3-70B Instruct** | ~1193 | 82.0% | 81.0% | 58.0% | 51.0% | 128K | OpenRouter, Groq, Cerebras, GitHub, Cloudflare, OVHcloud, Nebius, Nscale | Varies |
| **Llama 4 Scout (17B-16E)** | ~1162 | 80.5% | 79.0% | 54.0% | 48.0% | 10M | Groq, Cerebras, GitHub, Cloudflare, OpenRouter | Varies |
| **Llama 3.1-8B Instruct** | ~1057 | 68.4% | 62.2% | 28.4% | 33.0% | 128K | Cerebras, NVIDIA, Cloudflare, OVHcloud, HuggingFace, Ollama | Varies |
| **Llama 3.1-405B** | ~1260 [E] | ~87.5% [E] | ~86% [E] | ~68% [E] | ~55% [E] | 128K | NVIDIA NIM | ~40 RPM |

### Mistral Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **Mistral Large 3 (2411)** | ~1220 | 84.0% | 80.0% | 60.0% | 52.0% | 256K | Mistral API, NVIDIA, OpenRouter | ~1 RPS |
| **Mistral Medium 3 (2505)** | ~1183 [E] | ~82.5% [E] | ~78% [E] | ~55% [E] | ~48% [E] | 128K | Mistral API | ~1 RPS |
| **Mistral Small 4 (2603)** | ~1135 [E] | ~78% [E] | ~75% [E] | ~52% [E] | ~43% [E] | 256K | Mistral API | ~1 RPS |
| **Mistral Small 3.1 (24B)** | ~1110 | 75.0% | 72.0% | 48.0% | 40.0% | 128K | Mistral API, GitHub, Groq, Cloudflare | ~1 RPS |
| **Codestral (2501)** | ~1125 (coding) | ~72% | 82.0% | ~40% | ~35% | 256K | Mistral API | ~1 RPS |
| **Pixtral Large** | ~1160 | ~82% [E] | ~76% [E] | ~55% [E] | ~48% [E] | 128K | Mistral API | ~1 RPS |
| **Devstral (2512)** | ~1115 [E] | ~74% [E] | ~80% [E] | ~46% [E] | ~38% [E] | 256K | OpenRouter | 20 RPM, 200 RPD |
| **Mistral Nemo (12B)** | ~1049 | 68.0% | 55.0% | 30.0% | 32.0% | 128K | Mistral API, OVHcloud | ~1 RPS |
| **Dolphin-Mistral-24B** | ~1060 [E] | ~73% [E] | ~68% [E] | ~42% [E] | ~36% [E] | 32K | OpenRouter | 20 RPM, 200 RPD |
| **Mixtral-8x7B** | ~1036 | 70.6% | 40.0% | 28.4% | 34.0% | 32K | HuggingFace, OVHcloud | ~1,000 RPD |
| **Mistral-7B-Instruct-v0.3** | ~993 | 63.6% | 40.0% | 20.0% | ~28% | 32K | HuggingFace | ~1,000 RPD |

### NVIDIA Nemotron Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **Nemotron Ultra 253B** | ~1275-1280 | ~89.0% | ~88% | ~70% | ~58% | 128K | NVIDIA NIM | ~40 RPM |
| **Nemotron Super 120B** | ~1210 | ~85% | ~82% | ~60% | ~52% | 262K | OpenRouter, NVIDIA, Kilo | 20 RPM (OR) |
| **Nemotron Nano 30B** | ~1110 | ~78% | ~72% | ~45% | ~40% | 128K | OpenRouter, NVIDIA | 20 RPM (OR) |
| **Nemotron Nano 2 VL** | ~1020 [E] | ~68% [E] | ? | ? | ? | 128K | NVIDIA NIM | ~40 RPM |

### Cohere Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **Command A (111B)** | ~1208 | 85.4% | 80.0% | 60.0% | 50.0% | 256K | Cohere API | 20 RPM |
| **Command R+** | ~1115 | 75.7% | 70.0% | 40.0% | ~38% | 128K | Cohere API | 20 RPM |
| **Command R** | ~1032 | 68.0% | 55.0% | 28.0% | ~30% | 128K | Cohere API | 20 RPM |
| **Command R7B** | ~968 [E] | ~55% [E] | ~40% [E] | ~15% [E] | ~22% [E] | 128K | Cohere API | 20 RPM |

### Zhipu / GLM Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **GLM-4.7-Flash** | ~1100 [E] | ~78% [E] | ~72% [E] | ~50% [E] | ~40% [E] | 200K | Zhipu AI, Cerebras | 1 concurrent |
| **GLM-4.5-Air** | ~1170 [E] | ~82% [E] | ~78% [E] | ~56% [E] | ~47% [E] | 128K | OpenRouter | 20 RPM, 200 RPD |
| **GLM-4.5-Flash** | ~1090 [E] | ~75% [E] | ~68% [E] | ~44% [E] | ~36% [E] | 128K | Zhipu AI | 1 concurrent |
| **GLM-4.6V-Flash** | ~1080 [E] | ~74% [E] | ~66% [E] | ~42% [E] | ~34% [E] | 128K | Zhipu AI | 1 concurrent |
| **GLM-4-9B** | ~1067 | 72.5% | 60.0% | 35.0% | ~32% | 128K | SiliconFlow | 1,000 RPM, 50K TPM |
| **GLM-4.1V-9B-Thinking** | ~1090 [E] | ~74% [E] | ~62% [E] | ~42% [E] | ~35% [E] | 128K | SiliconFlow | 1,000 RPM, 50K TPM |

### Moonshot / Kimi Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **Kimi K2.5** | ~1268 [E] | ~87% [E] | ~86% [E] | ~72% [E] | ~58% [E] | 256K | Cloudflare | 10K neurons/day |
| **Kimi K2** | ~1225 [E] | ~84% [E] | ~82% [E] | ~66% [E] | ~54% [E] | 262K | Groq | 30 RPM |

### MiniMax Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **MiniMax-M2.7** | ~1215 [E] | ~84.5% [E] | ~82% [E] | ~64% [E] | ~52% [E] | 1M+ | NVIDIA NIM | ~40 RPM |
| **MiniMax-M2.5** | ~1168 [E] | ~81.5% [E] | ~78% [E] | ~58% [E] | ~48% [E] | 196K | OpenRouter, Kilo | 20 RPM (OR) |

### AI21 Labs Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **Jamba Large 1.7** | ~1220 [E] | ~85.5% [E] | ~82% [E] | ~62% [E] | ~52% [E] | 256K | AI21 Labs | 200 RPM, 10 RPS |
| **Jamba Mini 2** | ~1105 [E] | ~76% [E] | ~70% [E] | ~46% [E] | ~38% [E] | 256K | AI21 Labs | 200 RPM, 10 RPS |

### Microsoft / Phi Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **Phi-3.5-mini (3.8B)** | ~1035 | 67.0% | 55.0% | 38.0% | ~30% | 128K | HuggingFace | ~1,000 RPD |

### NousResearch Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **Hermes 3 Llama 3.1 405B** | ~1255 [E] | ~88% [E] | ~85% [E] | ~65% [E] | ~56% [E] | 128K | OpenRouter | 20 RPM, 200 RPD |

### ByteDance Seed Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **Dola Seed 2.0 Pro** | ~1240 [E] | ~86% [E] | ~84% [E] | ~68% [E] | ~55% [E] | ? | Kilo | ~200 req/hr |

### Arcee Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **Trinity Large Thinking** | ~1200 [E] | ~82% [E] | ~78% [E] | ~68% [E] | ~50% [E] | ? | Kilo | ~200 req/hr |

### InclusionAI Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **Ling 2.6 Flash** | ~1095 [E] | ~77% [E] | ~70% [E] | ~48% [E] | ~38% [E] | ? | OpenRouter | 20 RPM, 200 RPD |

### Aion Labs Family

| Model | Arena ELO | MMLU | HumanEval | MATH | GPQA (Diamond) | Context | Free via | Rate Limit |
|---|---|---|---|---|---|---|---|---|
| **Aion 2.0** | ~1210 [E] | ~84% [E] | ~80% [E] | ~60% [E] | ~50% [E] | 131K | Aion Labs | Daily token allowance |
| **Aion 1.0** | ~1150 [E] | ~79% [E] | ~74% [E] | ~50% [E] | ~42% [E] | 131K | Aion Labs | Daily token allowance |
| **Aion 1.0-mini** | ~1060 [E] | ~72% [E] | ~62% [E] | ~35% [E] | ~30% [E] | 131K | Aion Labs | Daily token allowance |

---

## Provider Speed Benchmarks

> Speed is measured in output tokens/sec. TTFT = Time To First Token.
> Speeds are approximate ranges, highly dependent on model size, load, region, and API tier.

### Tier 1 — Ultra-Fast (400+ tok/s)

| Provider | Key Models | Output Speed | TTFT | Architecture |
|---|---|---|---|---|
| **Cerebras** | llama-3.1-8b, llama-3.3-70b, qwen-3-32b, qwen-3-235b, zai-glm-4.7, gpt-oss-120b, llama-4-scout | **400–2,000+ tok/s** (8B: ~1500-2000, 70B: ~400-600) | **10–50ms** | CS-3 wafer-scale chip |
| **Groq** | llama-3.3-70b-versatile, llama-3.1-8b-instant, llama-4-scout, qwen3-32b, kimi-k2, deepseek-r1-distill-70b | **250–800 tok/s** (8B: ~600-800, 70B: ~250-350) | **30–80ms** | LPU (Language Processing Unit) |

### Tier 2 — Fast (80–200 tok/s)

| Provider | Key Models | Output Speed | TTFT | Notes |
|---|---|---|---|---|
| **Google Gemini API** | gemini-2.5-flash, flash-lite, 2.5-pro, 3-flash-preview | **40–150 tok/s** (Flash: 80-150, Pro: 40-60) | **100–500ms** | TPU infrastructure |
| **Mistral API** | mistral-small, medium, large, codestral, nemo | **40–130 tok/s** (Small: 80-130, Large: 40-60) | **150–400ms** | Self-hosted infrastructure |
| **NVIDIA NIM** | deepseek-r1, nemotron, llama-405b, qwen2.5-72b | **30–120 tok/s** (Small: 80-120, Large: 25-40) | **200–600ms** | H100/A100 GPUs |
| **xAI API** | grok-3-mini, grok-4.1-fast, grok-4.3 | **40–120 tok/s** (Mini: 80-120) | **150–400ms** | Memphis supercluster |
| **AI21 Labs** | jamba-large, jamba-mini | **30–120 tok/s** (Mini: 80-120) | **200–500ms** | Mamba/Transformer hybrid |

### Tier 3 — Moderate (30–80 tok/s)

| Provider | Key Models | Output Speed | TTFT | Notes |
|---|---|---|---|---|
| **Cloudflare Workers AI** | llama models, mistral-small, qwq-32b, gemma-4-26b, kimi-k2.5 | **30–80 tok/s** | **150–500ms** | Edge GPU inference |
| **GitHub Models** | gpt-4.1, gpt-4o, o3/o4-mini, llama-4-scout, deepseek-r1 | **20–100 tok/s** | **300–800ms** | Proxies to Azure/OpenAI |
| **SiliconFlow** | deepseek, qwen, llama models | **40–100 tok/s** | **150–400ms** | Chinese GPU cloud |
| **Nebius** | llama, qwen, deepseek models | **40–120 tok/s** | **150–400ms** | H100/L40 GPUs |
| **Cohere API** | command-a, command-r-plus, command-r, command-r7b | **30–100 tok/s** (R7b: 80-100) | **200–500ms** | Self-hosted |
| **OpenRouter (free)** | Various free models | **15–45 tok/s** | **400–1,200ms** | Routes to upstream; free tier may queue |
| **DeepSeek API** | deepseek-chat (V3), deepseek-reasoner (R1) | **30–60 tok/s** | **200–500ms** | Chat: 40-60, R1 thinking tokens separate |
| **Alibaba Cloud** | qwen-2.5/3 series | **30–80 tok/s** | **200–500ms** | Smaller variants: ~80-100 |
| **Zhipu AI** | glm-4, glm-4.5 | **30–60 tok/s** | **200–500ms** | Chinese provider |
| **Nscale** | llama, qwen, deepseek models | **30–80 tok/s** | **200–500ms** | European GPU cloud |

### Tier 4 — Slow/Variable (10–60 tok/s)

| Provider | Output Speed | TTFT | Notes |
|---|---|---|---|
| **HuggingFace Inference API** | **10–60 tok/s** | **300–1,000ms+** | Free serverless has cold starts |
| **OVHcloud AI Endpoints** | **20–60 tok/s** | **300–700ms** | European GPU cloud |
| **LLM7.io** | **20–60 tok/s** | **300–800ms** | Smaller provider |
| **Kilo** | **30–80 tok/s** | **200–500ms** | Emerging provider |
| **ModelScope** | **20–60 tok/s** | **300–600ms** | Alibaba model hub |
| **Aion Labs** | **20–50 tok/s** | **300–600ms** | Niche provider |
| **Ollama Cloud** | **20–50 tok/s** | **300–600ms** | Cloud-hosted Ollama |

---

## Unified Comparison Table

> Top models ranked by estimated Arena ELO. Only includes models with ELO > 1200.

| Rank | Model | Provider(s) | Arena ELO | MMLU | MATH | HumanEval | GPQA | Context | Free? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | GPT-5 | GitHub Models | ~1420-1460 | ~92-94% | ~92%+ | ~95%+ | ~68% | ~200K | Yes (GitHub) |
| 2 | Gemini 2.5 Pro | Google Gemini API | ~1380-1410 | 90.0% | 90.2% | 92.0% | 65.0% | 1M | Yes (free tier) |
| 3 | Grok-4.3 | xAI API | ~1380-1420 | ~90-92% | ~90%+ | ~92% | ~65% | 1M | Credit-based |
| 4 | o4-mini | GitHub Models | ~1370-1400 | ~88% | ~91% | ~92% | ~63% | 200K | Yes (GitHub) |
| 5 | DeepSeek-R1-0528 | OpenRouter, llm7.io | ~1370-1400 | ~91-92% | ~85%+ | ~93% | ~73% | 128K | Yes (OR/llm7) |
| 6 | GPT-4.1 | GitHub Models | ~1350-1370 | 90.2% | 84.0% | 92.0% | 60.0% | 1M | Yes (GitHub) |
| 7 | GPT-OSS-120B | OpenRouter, Cerebras, Groq | ~1320-1360 | ~88-90% | ~82% | ~90% | ~56% | ~128K | Yes (OR/Cerebras) |
| 8 | Qwen3-Max | Alibaba Cloud | ~1350-1380 | ~89-91% | ~86% | ~89% | ~58% | 128K | Tiered |
| 9 | Qwen3-Coder-480B | OpenRouter, NVIDIA | ~1350-1400 | ~88-90% | ~85% | ~93%+ | ~58% | 128K | Yes (OR) |
| 10 | Qwen3-235B-A22B | OpenRouter, Cerebras | ~1330-1370 | 89.5% | 88.0% | 89.0% | 60.0% | 131K | Yes (OR/Cerebras) |
| 11 | Gemini 2.5 Flash | Google Gemini API | ~1300-1340 | 88.0% | 83.0% | 89.0% | 57.0% | 1M | Yes (free tier) |
| 12 | DeepSeek-R1 | OpenRouter, Groq, NVIDIA | ~1350-1380 | 90.8% | 79.8% | 91.6% | 71.5% | 128K | Yes (OR/Groq) |
| 13 | Nemotron Ultra 253B | NVIDIA NIM | ~1275-1280 | ~89.0% | ~70% | ~88% | ~58% | 128K | Yes (NIM) |
| 14 | o3-mini | GitHub Models | ~1320-1340 | 86.5% | 87.3% | 91.0% | 59.0% | 200K | Yes (GitHub) |
| 15 | Kimi K2.5 | Cloudflare | ~1268 | ~87% | ~72% | ~86% | ~58% | 256K | Yes (CF) |
| 16 | Hermes 3 Llama 405B | OpenRouter | ~1255 | ~88% | ~65% | ~85% | ~56% | 128K | Yes (OR) |
| 17 | Grok-3-mini | xAI API | ~1270-1320 | ~85% | ~85% | ~86% | ~55% | 131K | Credit-based |
| 18 | Llama 4 Maverick | OpenRouter, GitHub | ~1237 | 84.4% | 62.0% | 85.0% | 55.0% | 1M | Yes (OR/GitHub) |
| 19 | Mistral Large 3 | Mistral API, NVIDIA | ~1220 | 84.0% | 60.0% | 80.0% | 52.0% | 256K | Yes (Mistral) |
| 20 | Command A (111B) | Cohere API | ~1208 | 85.4% | 60.0% | 80.0% | 50.0% | 256K | Yes (trial) |

---

## Data Confidence Levels

| Level | Models | Explanation |
|---|---|---|
| **HIGH** | GPT-4o, GPT-4o-mini, DeepSeek-R1, DeepSeek-V3, Qwen2.5-7B, Qwen2.5-72B, Gemma 3 series, Llama 3.1-8B, Llama 3.3-70B, Mistral-7B, Mixtral-8x7B | Well-documented public benchmarks from multiple sources |
| **MEDIUM** | GPT-4.1, GPT-4.1-mini, o3-mini, Gemini 2.5 Pro/Flash, Qwen3 series, Llama 4 Scout/Maverick, Mistral Small/Large, DeepSeek-R1-Distill-70B, Command A | Recently released but benchmarks available from official sources |
| **LOW** | o4-mini, GPT-5, GPT-OSS, Gemini 3 Flash Preview, Grok-4.x, Qwen3.5, Qwen3.6, DeepSeek-V3.2, Mistral Small 4, MiniMax, GLM-4.x, Kimi K2.5, Jamba, Aion, Dola Seed | Very recent, speculative, or limited English-language benchmarks available |

---

## Sources

| Source | URL | What it provides |
|---|---|---|
| LMSYS Chatbot Arena | https://lmarena.ai | Arena ELO ratings, head-to-head comparisons |
| Artificial Analysis | https://artificialanalysis.ai | Speed (tok/s), TTFT, cost comparisons |
| HuggingFace Open LLM Leaderboard | https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard | Open model benchmarks |
| OpenRouter Models | https://openrouter.ai/models | Model specs, pricing, context windows |
| Aider Leaderboard | https://aider.chat/docs/leaderboards/ | Coding benchmarks |
| Provider model cards | Various | Official benchmark claims |
| Community benchmarks | Reddit, HN, X/Twitter | Real-world speed and quality reports |

---

## Notes for Leaderboard Page Builder

1. **Deduplicate models**: Many models appear under multiple providers (e.g., Llama 3.3-70B is on OpenRouter, Groq, Cerebras, GitHub, Cloudflare, NVIDIA, etc.). The leaderboard should group by model and show provider options.

2. **Speed is provider-dependent**: The same model can be 10-50x faster on Cerebras/Groq vs. OpenRouter free tier. Show speed per provider.

3. **Mark estimated data clearly**: Values marked with `~` or `[E]` should be visually distinguished from verified scores.

4. **Rate limits matter for free tier**: Users care about RPM/RPD as much as quality. A fast model with 2 RPM isn't useful for most use cases.

5. **Chinese providers have limited EN benchmarks**: MiniMax, Zhipu, Moonshot, InclusionAI, ByteDance scores are largely estimated. Consider a "data quality" indicator.

6. **Models to highlight for free tier users**:
   - Best quality: GPT-4.1 (GitHub), Gemini 2.5 Flash (Google), DeepSeek-R1 (OpenRouter/llm7)
   - Best speed: llama-3.1-8b (Cerebras), llama-3.3-70b (Cerebras/Groq)
   - Best value: GPT-4.1-mini (GitHub), Qwen3-32B (Groq), DeepSeek-V3 (OpenRouter)
