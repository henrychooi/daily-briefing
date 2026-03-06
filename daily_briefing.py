"""
AI Intelligence Bot - Enhanced Daily Briefing
Goals: Stay ahead of research trends + Find models for production use
Focus: Broad AI — LLM, ASR, VAD, Vision, Multimodal, RL, Diffusion, Safety & more
"""

import os
import json
import time
import re
import requests
import feedparser
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
SEEN_IDS_FILE = "seen_ids.json"
USER_KEYWORDS_FILE = "user_keywords.json"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

NOW_UTC = datetime.now(timezone.utc)
CUTOFF_24H = NOW_UTC - timedelta(hours=24)
CUTOFF_48H = NOW_UTC - timedelta(hours=48)

# ── KEYWORD RADAR ─────────────────────────────────────────────────────────────
KEYWORD_RADAR = {
    # ── Speech & Audio ────────────────────────────────────────────────────────
    "ASR / Speech Architecture": [
        "CTC", "RNN-T", "Transducer", "AED", "attention encoder decoder",
        "conformer", "squeezeformer", "zipformer", "branchformer", "e-branchformer",
        "streaming ASR", "end-to-end ASR", "acoustic model", "speech encoder",
        "wav2vec", "HuBERT", "whisper", "seamless", "MMS"
    ],
    "VAD / Speaker Analysis": [
        "voice activity detection", "VAD", "speaker diarization", "speaker verification",
        "speaker identification", "speaker embedding", "x-vector", "d-vector",
        "ECAPA", "pyannote", "silero", "WebRTC VAD", "overlap detection",
        "turn detection", "speaker change"
    ],
    "TTS / Voice Synthesis": [
        "text-to-speech", "TTS", "zero-shot TTS", "voice cloning", "speech synthesis",
        "neural vocoder", "HiFi-GAN", "WaveNet", "VITS", "FastSpeech",
        "YourTTS", "Bark", "Tortoise", "voicebox", "natural speech",
        "prosody", "expressive TTS", "multilingual TTS"
    ],
    "Keyword Spotting / Biasing": [
        "keyword spotting", "wake word", "hotword", "KWS",
        "contextual biasing", "keyword boosting", "rare word", "shallow fusion",
        "deep biasing", "prefix tree", "biasing list", "domain adaptation"
    ],

    # ── LLM & NLP ─────────────────────────────────────────────────────────────
    "LLM Architecture": [
        "transformer", "attention mechanism", "mixture of experts", "MoE",
        "state space model", "Mamba", "RWKV", "sliding window attention",
        "grouped query attention", "GQA", "multi-head latent attention", "MLA",
        "flash attention", "rotary embedding", "RoPE", "sparse attention"
    ],
    "LLM Training / Alignment": [
        "LoRA", "QLoRA", "PEFT", "instruction tuning", "RLHF", "DPO", "PPO",
        "GRPO", "constitutional AI", "reward model", "preference learning",
        "supervised fine-tuning", "SFT", "continual pretraining",
        "chain-of-thought", "reasoning", "test-time compute"
    ],
    "RAG / Agents / Tools": [
        "RAG", "retrieval augmented", "agentic", "tool use", "function calling",
        "code interpreter", "long context", "context window", "memory",
        "knowledge graph", "vector database", "embedding search",
        "multi-agent", "autonomous agent", "computer use"
    ],

    # ── Vision & Multimodal ───────────────────────────────────────────────────
    "Computer Vision": [
        "vision transformer", "ViT", "object detection", "YOLO", "DETR",
        "image segmentation", "SAM", "depth estimation", "optical flow",
        "image classification", "contrastive learning", "CLIP", "DINO",
        "feature pyramid", "anchor-free detection"
    ],
    "Multimodal / Vision-Language": [
        "vision language model", "VLM", "multimodal LLM", "image captioning",
        "visual question answering", "VQA", "document understanding",
        "OCR", "chart understanding", "video understanding",
        "speech LLM", "audio LLM", "omni model", "any-to-any"
    ],
    "Image / Video Generation": [
        "diffusion model", "stable diffusion", "FLUX", "DiT", "U-Net",
        "latent diffusion", "flow matching", "consistency model",
        "image generation", "video generation", "text-to-image", "text-to-video",
        "ControlNet", "LoRA diffusion", "inpainting", "super resolution"
    ],

    # ── Reinforcement Learning ────────────────────────────────────────────────
    "Reinforcement Learning": [
        "reinforcement learning", "RL", "deep RL", "policy gradient",
        "proximal policy optimization", "PPO", "SAC", "actor-critic",
        "model-based RL", "offline RL", "multi-agent RL", "MARL",
        "reward shaping", "exploration", "sim-to-real", "robotics"
    ],

    # ── Safety & Alignment ────────────────────────────────────────────────────
    "Safety / Alignment": [
        "AI safety", "alignment", "hallucination", "factuality", "truthfulness",
        "jailbreak", "red teaming", "adversarial", "robustness",
        "interpretability", "mechanistic interpretability", "watermarking",
        "bias", "fairness", "toxicity", "refusal", "guardrails"
    ],

    # ── Production & Deployment ───────────────────────────────────────────────
    "Production / Deployment": [
        "GGUF", "ONNX", "quantization", "int4", "int8", "AWQ", "GPTQ", "EXL2",
        "TensorRT", "OpenVINO", "llama.cpp", "vLLM", "TGI", "Triton",
        "edge deploy", "mobile inference", "CPU inference", "batching",
        "speculative decoding", "KV cache", "continuous batching"
    ],

    # ── Benchmarks ────────────────────────────────────────────────────────────
    "Benchmarks / Evals": [
        "WER", "word error rate", "BLEU", "ROUGE", "MMLU", "HellaSwag",
        "HumanEval", "GSM8K", "MATH", "ARC", "TruthfulQA", "BIG-Bench",
        "LMSYS", "Chatbot Arena", "LibriSpeech", "CommonVoice", "SUPERB",
        "FLEURS", "GLUE", "SuperGLUE", "leaderboard", "SOTA", "state-of-the-art",
        "perplexity", "pass@k", "ELO"
    ],
}

# ── RSS FEEDS ─────────────────────────────────────────────────────────────────
RSS_FEEDS = [
    # Major labs
    ("Google DeepMind",   "https://deepmind.google/blog/rss.xml"),
    ("Google Research",   "https://research.google/blog/rss/"),
    ("Google Cloud AI",   "https://cloud.google.com/blog/products/ai-machine-learning/rss"),
    ("OpenAI",            "https://openai.com/news/rss/"),
    ("Anthropic",         "https://www.anthropic.com/rss.xml"),
    ("Meta AI",           "https://ai.meta.com/blog/rss/"),
    ("Mistral",           "https://mistral.ai/news/rss"),
    ("EleutherAI",        "https://blog.eleuther.ai/rss.xml"),
    # Vision & Diffusion
    ("Stability AI",      "https://stability.ai/news/rss"),
    ("Hugging Face Blog", "https://huggingface.co/blog/feed.xml"),
    # Infra & Deployment
    ("NVIDIA AI",         "https://blogs.nvidia.com/blog/category/deep-learning/feed/"),
    ("LangChain",         "https://blog.langchain.dev/rss/"),
]

# ── GitHub Trending Search Queries ───────────────────────────────────────────
GITHUB_SEARCH_QUERIES = [
    # ASR / Speech
    "automatic speech recognition",
    "voice activity detection",
    "text to speech neural",
    # LLM
    "large language model fine-tuning",
    "llm inference serving",
    "LLM quantization",
    # Vision & Multimodal
    "vision language model",
    "image generation diffusion",
    "object detection transformer",
    # RL & Agents
    "reinforcement learning agent",
    "AI agent framework",
]

# ── HuggingFace Model Domain Groups ──────────────────────────────────────────
# Each entry: (display_label, emoji, color_key, hf_pipeline_tags, discord_color)
MODEL_DOMAINS = [
    (
        "LLM / Text Generation",
        "🧠", "nlp",
        "text-generation,text2text-generation,summarization,translation,question-answering,text-classification,token-classification",
        0x2563EB,  # Blue
    ),
    (
        "ASR / Speech / Audio",
        "🎙️", "speech",
        "automatic-speech-recognition,text-to-speech,audio-classification,audio-to-audio,voice-activity-detection",
        0x059669,  # Green
    ),
    (
        "Vision / Image",
        "👁️", "vision",
        "image-classification,object-detection,image-segmentation,depth-estimation,image-to-image,zero-shot-image-classification",
        0xD97706,  # Amber
    ),
    (
        "Multimodal",
        "🌐", "multimodal",
        "image-to-text,visual-question-answering,document-question-answering,video-classification,image-text-to-text",
        0x7C3AED,  # Purple
    ),
    (
        "Image / Video Generation",
        "🎨", "generation",
        "text-to-image,text-to-video,image-to-video,unconditional-image-generation,inpainting",
        0xDB2777,  # Pink
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_seen_ids() -> dict:
    if os.path.exists(SEEN_IDS_FILE):
        with open(SEEN_IDS_FILE) as f:
            return json.load(f)
    return {"papers": [], "models": [], "announcements": [], "github": []}


def load_user_keywords() -> list[str]:
    """Load user-defined keywords from file (managed via bot.py)."""
    if os.path.exists(USER_KEYWORDS_FILE):
        with open(USER_KEYWORDS_FILE) as f:
            data = json.load(f)
            return [kw.lower().strip() for kw in data.get("keywords", [])]
    return []


def save_seen_ids(seen: dict):
    # Keep last 500 per category to avoid unbounded growth
    for k in seen:
        seen[k] = seen[k][-500:]
    with open(SEEN_IDS_FILE, "w") as f:
        json.dump(seen, f, indent=2)


def radar_scan(text: str) -> dict[str, list[str]]:
    """Scan text for keyword radar matches. Returns {category: [matched_keywords]}"""
    text_lower = text.lower()
    hits = {}
    for category, keywords in KEYWORD_RADAR.items():
        matched = [kw for kw in keywords if kw.lower() in text_lower]
        if matched:
            hits[category] = matched[:4]  # Cap at 4 per category
    return hits


def format_radar_hits(hits: dict) -> str:
    if not hits:
        return ""
    lines = []
    for cat, kws in hits.items():
        kw_str = ", ".join(f"`{k}`" for k in kws)
        lines.append(f"  ↳ **{cat}**: {kw_str}")
    return "\n".join(lines)


def iso_to_dt(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def fetch_readme(repo_id: str) -> str:
    """Fetch HuggingFace model README for radar scanning."""
    try:
        url = f"https://huggingface.co/{repo_id}/raw/main/README.md"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return r.text[:6000]  # First 6k chars is enough
    except Exception:
        pass
    return ""


def truncate(s: str, n: int = 100) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE A — HuggingFace Daily Papers (with Trend Velocity)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_papers(seen_ids: list) -> list[dict]:
    print("📄 Fetching HF Daily Papers...")
    papers = []
    try:
        r = requests.get("https://huggingface.co/api/daily_papers", timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ⚠ Papers API failed: {e}")
        return []

    for item in data:
        paper = item.get("paper", {})
        pid = paper.get("id", "")
        if not pid or pid in seen_ids:
            continue

        title = paper.get("title", "Untitled")
        upvotes = item.get("numComments", 0) + paper.get("upvotes", 0)
        summary = paper.get("summary", "")[:400]
        published = paper.get("publishedAt", "")
        pub_dt = iso_to_dt(published)

        # Only last 24h papers with at least 3 upvotes
        if pub_dt < CUTOFF_24H or upvotes < 3:
            continue

        # Check for code/weights
        links = []
        github_url = None
        for link in paper.get("links", []):
            url = link.get("url", "")
            if "github.com" in url:
                github_url = url
                links.append(f"[💻 Code]({url})")
            elif "huggingface.co" in url and "/datasets/" not in url:
                links.append(f"[🤗 Weights]({url})")

        # Radar scan title + abstract
        radar_hits = radar_scan(title + " " + summary)

        papers.append({
            "id": pid,
            "title": title,
            "upvotes": upvotes,
            "summary": summary,
            "published": pub_dt.strftime("%b %d %H:%M UTC"),
            "links": links,
            "github_url": github_url,
            "radar": radar_hits,
            "url": f"https://huggingface.co/papers/{pid}",
            "authors": paper.get("authors", [])[:3],
        })

    # Sort by upvotes descending, take top 5
    papers.sort(key=lambda x: x["upvotes"], reverse=True)
    return papers[:5]


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE B — HuggingFace Models (NLP + Speech, with README radar)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_models(task: str, label: str, seen_ids: list, limit: int = 4) -> list[dict]:
    print(f"🤖 Fetching HF Models [{label}]...")
    params = {
        "filter": task,
        "sort": "createdAt",
        "direction": -1,
        "limit": 40,
        "full": "true",
    }
    try:
        r = requests.get("https://huggingface.co/api/models", params=params,
                         headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ⚠ Models API failed for {task}: {e}")
        return []

    models = []
    for m in data:
        mid = m.get("id", "")
        if not mid or mid in seen_ids:
            continue

        created = iso_to_dt(m.get("createdAt", ""))
        likes = m.get("likes", 0)
        downloads = m.get("downloads", 0)

        if created < CUTOFF_48H or likes < 2:
            continue

        # Scan card data + fetch README for deeper radar hits
        card_data = json.dumps(m.get("cardData", {}))
        readme_text = fetch_readme(mid)
        full_text = card_data + " " + readme_text
        radar_hits = radar_scan(mid + " " + full_text)

        # Extract benchmark scores from card metadata
        benchmarks = extract_benchmarks(m.get("cardData", {}))

        # Architecture tags
        tags = m.get("tags", [])
        arch_tags = [t for t in tags if t in (
            "transformers", "pytorch", "jax", "onnx", "ctranslate2",
            "gguf", "awq", "gptq", "safetensors"
        )]

        models.append({
            "id": mid,
            "likes": likes,
            "downloads": downloads,
            "created": created.strftime("%b %d %H:%M UTC"),
            "pipeline_tag": m.get("pipeline_tag", task),
            "tags": arch_tags[:5],
            "benchmarks": benchmarks,
            "radar": radar_hits,
            "url": f"https://huggingface.co/{mid}",
            "author": mid.split("/")[0] if "/" in mid else "unknown",
            "model_name": mid.split("/")[-1] if "/" in mid else mid,
        })

    models.sort(key=lambda x: (x["likes"], x["downloads"]), reverse=True)
    return models[:limit]


def extract_benchmarks(card_data: dict) -> list[str]:
    """Pull structured benchmark results from HF card metadata."""
    results = []
    evals = card_data.get("model-index", [])
    if isinstance(evals, list):
        for entry in evals[:1]:
            for result in entry.get("results", [])[:3]:
                task = result.get("task", {}).get("name", "")
                dataset = result.get("dataset", {}).get("name", "")
                for metric in result.get("metrics", [])[:2]:
                    name = metric.get("name", "")
                    val = metric.get("value", "")
                    if val:
                        results.append(f"{task}/{dataset} {name}: **{val}**")
    return results[:3]


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE C — RSS Industry Announcements
# ══════════════════════════════════════════════════════════════════════════════

def fetch_announcements(seen_ids: list) -> list[dict]:
    print("📡 Fetching RSS Announcements...")
    announcements = []

    for source, url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                eid = entry.get("id", entry.get("link", ""))
                if not eid or eid in seen_ids:
                    continue

                pub = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub:
                    pub_dt = datetime(*pub[:6], tzinfo=timezone.utc)
                    if pub_dt < CUTOFF_24H:
                        continue
                else:
                    continue

                title = entry.get("title", "")
                summary = re.sub(r"<[^>]+>", "", entry.get("summary", ""))[:300]
                link = entry.get("link", "")
                radar_hits = radar_scan(title + " " + summary)

                # Detect high-priority: model launches, API updates
                is_priority = any(kw in title.lower() for kw in [
                    "launch", "release", "introducing", "announce",
                    "api", "model", "gpt", "gemini", "claude", "llama",
                    "whisper", "benchmark", "sota"
                ])

                announcements.append({
                    "id": eid,
                    "source": source,
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "pub": pub_dt.strftime("%b %d %H:%M UTC"),
                    "radar": radar_hits,
                    "is_priority": is_priority,
                })
        except Exception as e:
            print(f"  ⚠ RSS failed [{source}]: {e}")

    return announcements


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE D — GitHub Trending (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_github_trending(seen_ids: list) -> list[dict]:
    print("🐙 Fetching GitHub trending repos...")
    repos = []
    seen_repo_ids = set()

    for query in GITHUB_SEARCH_QUERIES:
        try:
            params = {
                "q": f"{query} created:>{CUTOFF_48H.strftime('%Y-%m-%d')}",
                "sort": "stars",
                "order": "desc",
                "per_page": 5,
            }
            r = requests.get(
                "https://api.github.com/search/repositories",
                params=params,
                headers={"Accept": "application/vnd.github+json"},
                timeout=10
            )
            if r.status_code == 403:
                print("  ⚠ GitHub rate limit hit")
                break
            r.raise_for_status()
            data = r.json()

            for repo in data.get("items", []):
                rid = str(repo["id"])
                if rid in seen_ids or rid in seen_repo_ids:
                    continue
                if repo.get("stargazers_count", 0) < 10:
                    continue

                seen_repo_ids.add(rid)
                desc = repo.get("description", "") or ""
                radar_hits = radar_scan(repo["name"] + " " + desc)

                repos.append({
                    "id": rid,
                    "name": repo["full_name"],
                    "stars": repo["stargazers_count"],
                    "description": truncate(desc, 120),
                    "url": repo["html_url"],
                    "language": repo.get("language", ""),
                    "topics": repo.get("topics", [])[:5],
                    "radar": radar_hits,
                    "created": repo.get("created_at", "")[:10],
                })

            time.sleep(1)  # Respect GitHub rate limits

        except Exception as e:
            print(f"  ⚠ GitHub search failed [{query}]: {e}")

    repos.sort(key=lambda x: x["stars"], reverse=True)
    # Deduplicate by name, keep top 4
    seen_names = set()
    unique = []
    for r in repos:
        if r["name"] not in seen_names:
            seen_names.add(r["name"])
            unique.append(r)
    return unique[:4]


# ══════════════════════════════════════════════════════════════════════════════
# DISCORD FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "papers":        0x7C3AED,   # Purple
    "nlp":           0x2563EB,   # Blue
    "speech":        0x059669,   # Green
    "vision":        0xD97706,   # Amber
    "multimodal":    0x7C3AED,   # Purple
    "generation":    0xDB2777,   # Pink
    "announcements": 0xDC2626,   # Red
    "github":        0x374151,   # Dark grey
    "header":        0xF59E0B,   # Amber
}


def build_paper_embed(paper: dict) -> dict:
    radar_str = format_radar_hits(paper["radar"])
    links_str = "  " + " · ".join(paper["links"]) if paper["links"] else ""
    author_str = ""
    if paper["authors"]:
        names = [a.get("name", "") for a in paper["authors"] if a.get("name")]
        author_str = f"\n👤 *{', '.join(names[:3])}{'...' if len(paper['authors']) > 3 else ''}*"

    desc = f"**[{truncate(paper['title'], 110)}]({paper['url']})**{links_str}\n"
    desc += f"⬆ **{paper['upvotes']} upvotes** · 🕐 {paper['published']}{author_str}\n\n"
    desc += f"> {truncate(paper['summary'], 250)}\n"
    if radar_str:
        desc += f"\n🔭 **Keyword Radar**\n{radar_str}"

    return {
        "description": desc,
        "color": COLORS["papers"],
    }


def build_model_embed(model: dict, color_key: str) -> dict:
    tech_tags = " · ".join(f"`{t}`" for t in model["tags"]) if model["tags"] else ""
    bench_str = "\n".join(f"  📊 {b}" for b in model["benchmarks"]) if model["benchmarks"] else ""
    radar_str = format_radar_hits(model["radar"])

    desc = f"**[{truncate(model['model_name'], 80)}]({model['url']})**\n"
    desc += f"👤 `{model['author']}` · ❤️ {model['likes']} likes · ⬇ {model['downloads']:,} dl · 🕐 {model['created']}\n"
    if tech_tags:
        desc += f"🔧 {tech_tags}\n"
    if bench_str:
        desc += f"\n{bench_str}\n"
    if radar_str:
        desc += f"\n🔭 **Keyword Radar**\n{radar_str}"

    return {
        "description": desc,
        "color": COLORS.get(color_key, 0x6B7280),  # fallback grey
    }


def build_announcement_embed(ann: dict) -> dict:
    priority_flag = "🚨 " if ann["is_priority"] else ""
    radar_str = format_radar_hits(ann["radar"])

    desc = f"{priority_flag}**[{truncate(ann['title'], 110)}]({ann['link']})**\n"
    desc += f"📡 `{ann['source']}` · 🕐 {ann['pub']}\n\n"
    if ann["summary"]:
        desc += f"> {truncate(ann['summary'], 220)}\n"
    if radar_str:
        desc += f"\n🔭 **Keyword Radar**\n{radar_str}"

    return {
        "description": desc,
        "color": COLORS["announcements"],
    }


def build_github_embed(repo: dict) -> dict:
    topics_str = " ".join(f"`{t}`" for t in repo["topics"]) if repo["topics"] else ""
    radar_str = format_radar_hits(repo["radar"])

    desc = f"**[{repo['name']}]({repo['url']})**\n"
    desc += f"⭐ **{repo['stars']}** stars · 🗓 {repo['created']}"
    if repo["language"]:
        desc += f" · 💻 `{repo['language']}`"
    desc += "\n"
    if repo["description"]:
        desc += f"> {repo['description']}\n"
    if topics_str:
        desc += f"🏷 {topics_str}\n"
    if radar_str:
        desc += f"\n🔭 **Keyword Radar**\n{radar_str}"

    return {
        "description": desc,
        "color": COLORS["github"],
    }


def post_to_discord(webhook_url: str, embeds: list[dict], content: str = ""):
    """Post embeds to Discord. Splits into chunks of 10 (Discord max)."""
    if not webhook_url:
        return
    chunks = [embeds[i:i+10] for i in range(0, len(embeds), 10)]
    for i, chunk in enumerate(chunks):
        payload = {"embeds": chunk}
        if i == 0 and content:
            payload["content"] = content
        try:
            r = requests.post(webhook_url, json=payload, timeout=15)
            r.raise_for_status()
            time.sleep(0.5)
        except Exception as e:
            print(f"  ⚠ Discord post failed: {e}")


def build_section_header(emoji: str, title: str, subtitle: str, color: int) -> dict:
    return {
        "description": f"## {emoji} {title}\n*{subtitle}*",
        "color": color,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n🤖 AI Intelligence Bot starting at {NOW_UTC.strftime('%Y-%m-%d %H:%M UTC')}\n")

    seen = load_seen_ids()
    user_keywords = load_user_keywords()
    if user_keywords:
        print(f"🔑 User keywords active: {', '.join(user_keywords)}\n")

    # ── Fetch all pipelines ──────────────────────────────────────────────────
    papers        = fetch_papers(seen["papers"])
    announcements = fetch_announcements(seen["announcements"])
    github_repos  = fetch_github_trending(seen["github"])

    # Fetch models for every domain dynamically
    domain_models = {}  # label -> list[dict]
    for (label, emoji, color_key, hf_tags, color) in MODEL_DOMAINS:
        domain_models[label] = fetch_models(hf_tags, label, seen["models"])

    all_models_flat = [m for ml in domain_models.values() for m in ml]

    # ── Silent failure ───────────────────────────────────────────────────────
    total = len(papers) + len(all_models_flat) + len(announcements) + len(github_repos)
    if total == 0:
        print("ℹ️  No new content found today. Staying silent.")
        return

    # ── Build main digest ────────────────────────────────────────────────────
    date_str = NOW_UTC.strftime("%A, %B %d %Y")
    all_embeds = []

    # ── Header embed ─────────────────────────────────────────────────────────
    radar_total = sum(len(p["radar"]) for p in papers + all_models_flat)
    domain_counts = " · ".join(
        f"{emoji} `{len(domain_models[label])}` {label}"
        for (label, emoji, *_) in MODEL_DOMAINS
        if domain_models[label]
    )
    kw_line = f"\n🔑 Tracking **{len(user_keywords)}** custom keywords" if user_keywords else ""
    all_embeds.append({
        "title": f"🤖 AI Intelligence Briefing — {date_str}",
        "description": (
            f"**{total} new items** tracked across 4 pipelines\n\n"
            f"📄 `{len(papers)}` papers · "
            f"📡 `{len(announcements)}` announcements · "
            f"🐙 `{len(github_repos)}` GitHub repos\n"
            f"{domain_counts}\n"
            f"🔭 `{radar_total}` keyword radar hits"
            f"{kw_line}"
        ),
        "color": COLORS["header"],
        "footer": {"text": "AI Intelligence Bot • Broad AI Coverage • Use /search and /keyword in Discord"},
        "timestamp": NOW_UTC.isoformat(),
    })

    # ── Pipeline A: Papers ────────────────────────────────────────────────────
    if papers:
        all_embeds.append(build_section_header(
            "📄", "CURATED RESEARCH PAPERS",
            "Top community-upvoted · Last 24h · HuggingFace Daily Papers",
            COLORS["papers"]
        ))
        for p in papers:
            embed = build_paper_embed(p)
            # Flag if matches user keywords
            user_hits = [kw for kw in user_keywords if kw in (p["title"] + p["summary"]).lower()]
            if user_hits:
                embed["description"] = f"🔑 *Matches your keywords: {', '.join(f'`{k}`' for k in user_hits[:4])}*\n" + embed["description"]
            all_embeds.append(embed)

    # ── Pipeline B: Models (all domains) ─────────────────────────────────────
    for (label, emoji, color_key, hf_tags, color) in MODEL_DOMAINS:
        models = domain_models[label]
        if models:
            all_embeds.append(build_section_header(
                emoji, f"NEW {label.upper()} MODELS",
                "Created <48h · ≥2 likes · Sorted by likes + downloads",
                color,
            ))
            for m in models:
                embed = build_model_embed(m, color_key)
                embed["color"] = color
                user_hits = [kw for kw in user_keywords if kw in m["model_name"].lower()]
                if user_hits:
                    embed["description"] = f"🔑 *Matches your keywords: {', '.join(f'`{k}`' for k in user_hits[:4])}*\n" + embed["description"]
                all_embeds.append(embed)

    # ── Pipeline C: Announcements ─────────────────────────────────────────────
    if announcements:
        priority_count = sum(1 for a in announcements if a["is_priority"])
        all_embeds.append(build_section_header(
            "📡", "INDUSTRY ANNOUNCEMENTS",
            f"Last 24h · {priority_count} priority · Google · OpenAI · Anthropic · Meta · Mistral",
            COLORS["announcements"]
        ))
        sorted_ann = sorted(announcements, key=lambda x: x["is_priority"], reverse=True)
        for ann in sorted_ann[:6]:
            embed = build_announcement_embed(ann)
            user_hits = [kw for kw in user_keywords if kw in (ann["title"] + ann["summary"]).lower()]
            if user_hits:
                embed["description"] = f"🔑 *Matches your keywords: {', '.join(f'`{k}`' for k in user_hits[:4])}*\n" + embed["description"]
            all_embeds.append(embed)

    # ── Pipeline D: GitHub ────────────────────────────────────────────────────
    if github_repos:
        all_embeds.append(build_section_header(
            "🐙", "GITHUB TRENDING",
            "New repos <48h · ≥10 stars · Broad AI focus",
            COLORS["github"]
        ))
        for repo in github_repos:
            all_embeds.append(build_github_embed(repo))

    # ── Post to Discord ───────────────────────────────────────────────────────
    print(f"\n📨 Posting {len(all_embeds)} embeds to Discord...")
    post_to_discord(DISCORD_WEBHOOK_URL, all_embeds)
    print("✅ Main digest posted.\n")

    # ── Update seen IDs ───────────────────────────────────────────────────────
    seen["papers"]        += [p["id"] for p in papers]
    seen["models"]        += [m["id"] for m in all_models_flat]
    seen["announcements"] += [a["id"] for a in announcements]
    seen["github"]        += [r["id"] for r in github_repos]
    save_seen_ids(seen)
    print("💾 seen_ids.json updated.")


if __name__ == "__main__":
    main()