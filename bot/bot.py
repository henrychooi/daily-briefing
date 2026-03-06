"""
AI Intelligence Bot — Interactive Discord Bot
Slash commands for keyword management and on-demand article search.

Commands:
  /keyword add <word>      — Add a keyword to track
  /keyword remove <word>   — Remove a tracked keyword
  /keyword list            — Show all tracked keywords
  /keyword clear           — Remove all keywords
  /search <query>          — Search papers, models & announcements right now
  /search today            — Re-run today's full digest on demand
"""

import os
import json
import asyncio
import re
import requests
import feedparser
from datetime import datetime, timezone, timedelta

import discord
from discord import app_commands
from discord.ext import commands

# ── CONFIG ────────────────────────────────────────────────────────────────────
DISCORD_BOT_TOKEN    = os.environ["DISCORD_BOT_TOKEN"]
DISCORD_WEBHOOK_URL  = os.environ["DISCORD_WEBHOOK_URL"]
USER_KEYWORDS_FILE   = "user_keywords.json"
HF_TOKEN             = os.environ.get("HF_TOKEN", "")
HEADERS              = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

NOW_UTC   = lambda: datetime.now(timezone.utc)
CUTOFF_7D = lambda: NOW_UTC() - timedelta(days=7)

# ── COLOR PALETTE ─────────────────────────────────────────────────────────────
COLOR_PAPER  = 0x7C3AED
COLOR_MODEL  = 0x2563EB
COLOR_NEWS   = 0xDC2626
COLOR_KW     = 0xF59E0B
COLOR_OK     = 0x059669
COLOR_ERR    = 0xEF4444

# ── KEYWORD STORE ─────────────────────────────────────────────────────────────

def load_keywords() -> list[str]:
    if os.path.exists(USER_KEYWORDS_FILE):
        with open(USER_KEYWORDS_FILE) as f:
            return json.load(f).get("keywords", [])
    return []


def save_keywords(keywords: list[str]):
    with open(USER_KEYWORDS_FILE, "w") as f:
        json.dump({"keywords": keywords}, f, indent=2)


# ── SEARCH HELPERS ────────────────────────────────────────────────────────────

def truncate(s: str, n: int = 100) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def search_hf_papers(query: str, max_results: int = 5) -> list[dict]:
    """Search HuggingFace papers by keyword against last 7 days of daily papers."""
    results = []
    query_lower = query.lower()
    try:
        r = requests.get("https://huggingface.co/api/daily_papers", timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    for item in data:
        paper = item.get("paper", {})
        title = paper.get("title", "")
        summary = paper.get("summary", "")
        if query_lower in title.lower() or query_lower in summary.lower():
            upvotes = item.get("numComments", 0) + paper.get("upvotes", 0)
            pid = paper.get("id", "")
            results.append({
                "title": title,
                "upvotes": upvotes,
                "summary": summary[:300],
                "url": f"https://huggingface.co/papers/{pid}",
                "published": paper.get("publishedAt", "")[:10],
            })

    results.sort(key=lambda x: x["upvotes"], reverse=True)
    return results[:max_results]


def search_hf_models(query: str, max_results: int = 5) -> list[dict]:
    """Search HuggingFace models by keyword."""
    try:
        r = requests.get(
            "https://huggingface.co/api/models",
            params={"search": query, "sort": "likes", "direction": -1, "limit": max_results, "full": "true"},
            headers=HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    models = []
    for m in data:
        mid = m.get("id", "")
        models.append({
            "id": mid,
            "model_name": mid.split("/")[-1] if "/" in mid else mid,
            "author": mid.split("/")[0] if "/" in mid else "unknown",
            "likes": m.get("likes", 0),
            "downloads": m.get("downloads", 0),
            "pipeline_tag": m.get("pipeline_tag", ""),
            "url": f"https://huggingface.co/{mid}",
        })
    return models


def search_rss_announcements(query: str, max_results: int = 5) -> list[dict]:
    """Search recent RSS posts from all lab feeds for a keyword."""
    RSS_FEEDS = [
        ("Google DeepMind",   "https://deepmind.google/blog/rss.xml"),
        ("Google Research",   "https://research.google/blog/rss/"),
        ("OpenAI",            "https://openai.com/news/rss/"),
        ("Anthropic",         "https://www.anthropic.com/rss.xml"),
        ("Meta AI",           "https://ai.meta.com/blog/rss/"),
        ("Mistral",           "https://mistral.ai/news/rss"),
        ("Hugging Face Blog", "https://huggingface.co/blog/feed.xml"),
    ]

    query_lower = query.lower()
    cutoff = CUTOFF_7D()
    results = []

    for source, url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                title = entry.get("title", "")
                summary = re.sub(r"<[^>]+>", "", entry.get("summary", ""))
                if query_lower not in title.lower() and query_lower not in summary.lower():
                    continue
                pub = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub:
                    pub_dt = datetime(*pub[:6], tzinfo=timezone.utc)
                    if pub_dt < cutoff:
                        continue
                results.append({
                    "source": source,
                    "title": title,
                    "summary": summary[:250],
                    "link": entry.get("link", ""),
                    "pub": pub_dt.strftime("%b %d") if pub else "?",
                })
        except Exception:
            continue

    return results[:max_results]


# ── DISCORD EMBED BUILDERS ────────────────────────────────────────────────────

def paper_embed(p: dict) -> discord.Embed:
    e = discord.Embed(color=COLOR_PAPER)
    e.description = (
        f"**[{truncate(p['title'], 110)}]({p['url']})**\n"
        f"⬆ **{p['upvotes']} upvotes** · 🗓 {p['published']}\n\n"
        f"> {truncate(p['summary'], 280)}"
    )
    return e


def model_embed(m: dict) -> discord.Embed:
    e = discord.Embed(color=COLOR_MODEL)
    tag = f" · `{m['pipeline_tag']}`" if m["pipeline_tag"] else ""
    e.description = (
        f"**[{truncate(m['model_name'], 80)}]({m['url']})**\n"
        f"👤 `{m['author']}`{tag} · ❤️ {m['likes']} likes · ⬇ {m['downloads']:,} dl"
    )
    return e


def news_embed(n: dict) -> discord.Embed:
    e = discord.Embed(color=COLOR_NEWS)
    e.description = (
        f"**[{truncate(n['title'], 110)}]({n['link']})**\n"
        f"📡 `{n['source']}` · 🗓 {n['pub']}\n\n"
        f"> {truncate(n['summary'], 240)}"
    )
    return e


def section_embed(emoji: str, title: str, subtitle: str, color: int) -> discord.Embed:
    e = discord.Embed(color=color)
    e.description = f"## {emoji} {title}\n*{subtitle}*"
    return e


# ── BOT SETUP ─────────────────────────────────────────────────────────────────

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"✅ Bot online as {bot.user} — slash commands synced")


# ══════════════════════════════════════════════════════════════════════════════
# /keyword  commands
# ══════════════════════════════════════════════════════════════════════════════

keyword_group = app_commands.Group(name="keyword", description="Manage your tracked keywords")


@keyword_group.command(name="add", description="Add a keyword to track in daily digests")
@app_commands.describe(word="Keyword or phrase to track (e.g. 'LoRA', 'conformer', 'RLHF')")
async def keyword_add(interaction: discord.Interaction, word: str):
    word = word.strip().lower()
    if not word:
        await interaction.response.send_message("❌ Keyword cannot be empty.", ephemeral=True)
        return

    keywords = load_keywords()
    if word in [k.lower() for k in keywords]:
        await interaction.response.send_message(
            embed=discord.Embed(
                description=f"⚠️ `{word}` is already being tracked.",
                color=COLOR_KW
            ), ephemeral=True
        )
        return

    keywords.append(word)
    save_keywords(keywords)

    e = discord.Embed(color=COLOR_OK)
    e.description = (
        f"✅ **`{word}`** added to your keyword radar.\n\n"
        f"You're now tracking **{len(keywords)}** keyword(s). "
        f"Items matching your keywords will be flagged 🔑 in tomorrow's digest.\n\n"
        f"Use `/search {word}` to search right now."
    )
    await interaction.response.send_message(embed=e)


@keyword_group.command(name="remove", description="Remove a tracked keyword")
@app_commands.describe(word="Keyword to remove")
async def keyword_remove(interaction: discord.Interaction, word: str):
    word = word.strip().lower()
    keywords = load_keywords()
    original = keywords[:]
    keywords = [k for k in keywords if k.lower() != word]

    if len(keywords) == len(original):
        await interaction.response.send_message(
            embed=discord.Embed(description=f"❌ `{word}` was not found in your keywords.", color=COLOR_ERR),
            ephemeral=True
        )
        return

    save_keywords(keywords)
    e = discord.Embed(color=COLOR_OK)
    e.description = f"🗑️ **`{word}`** removed. You now have **{len(keywords)}** keyword(s) tracked."
    await interaction.response.send_message(embed=e)


@keyword_group.command(name="list", description="Show all currently tracked keywords")
async def keyword_list(interaction: discord.Interaction):
    keywords = load_keywords()
    e = discord.Embed(title="🔑 Tracked Keywords", color=COLOR_KW)
    if not keywords:
        e.description = (
            "No keywords tracked yet.\n\n"
            "Add some with `/keyword add <word>` — e.g.:\n"
            "`/keyword add LoRA`\n`/keyword add conformer`\n`/keyword add RLHF`"
        )
    else:
        kw_list = "\n".join(f"• `{k}`" for k in sorted(keywords))
        e.description = (
            f"Tracking **{len(keywords)}** keyword(s):\n\n{kw_list}\n\n"
            f"Matching items in daily digests are flagged with 🔑"
        )
    await interaction.response.send_message(embed=e)


@keyword_group.command(name="clear", description="Remove all tracked keywords")
async def keyword_clear(interaction: discord.Interaction):
    save_keywords([])
    e = discord.Embed(color=COLOR_OK)
    e.description = "🧹 All keywords cleared."
    await interaction.response.send_message(embed=e)


bot.tree.add_command(keyword_group)


# ══════════════════════════════════════════════════════════════════════════════
# /search  command
# ══════════════════════════════════════════════════════════════════════════════

@bot.tree.command(name="search", description="Search papers, models and news right now")
@app_commands.describe(
    query="What to search for (e.g. 'conformer ASR', 'LoRA fine-tuning', 'GPT-4o')",
    scope="Where to search"
)
@app_commands.choices(scope=[
    app_commands.Choice(name="Everything (papers + models + news)", value="all"),
    app_commands.Choice(name="Papers only",                         value="papers"),
    app_commands.Choice(name="Models only",                         value="models"),
    app_commands.Choice(name="News / announcements only",           value="news"),
])
async def search(interaction: discord.Interaction, query: str, scope: str = "all"):
    await interaction.response.defer(thinking=True)

    embeds = []
    query = query.strip()

    # ── Papers ────────────────────────────────────────────────────────────────
    if scope in ("all", "papers"):
        papers = await asyncio.to_thread(search_hf_papers, query)
        if papers:
            embeds.append(section_embed("📄", f"PAPERS — \"{query}\"",
                          f"{len(papers)} result(s) · HuggingFace Daily Papers · last 30 days",
                          COLOR_PAPER))
            for p in papers:
                embeds.append(paper_embed(p))

    # ── Models ────────────────────────────────────────────────────────────────
    if scope in ("all", "models"):
        models = await asyncio.to_thread(search_hf_models, query)
        if models:
            embeds.append(section_embed("🤖", f"MODELS — \"{query}\"",
                          f"{len(models)} result(s) · HuggingFace Hub · sorted by likes",
                          COLOR_MODEL))
            for m in models:
                embeds.append(model_embed(m))

    # ── News ──────────────────────────────────────────────────────────────────
    if scope in ("all", "news"):
        news = await asyncio.to_thread(search_rss_announcements, query)
        if news:
            embeds.append(section_embed("📡", f"NEWS — \"{query}\"",
                          f"{len(news)} result(s) · Lab blogs & RSS · last 7 days",
                          COLOR_NEWS))
            for n in news:
                embeds.append(news_embed(n))

    # ── No results ────────────────────────────────────────────────────────────
    if not embeds:
        e = discord.Embed(color=COLOR_ERR)
        e.description = (
            f"😔 No results found for **`{query}`** in the last 7 days.\n\n"
            "Try a broader term, or check your spelling. "
            "Use `/keyword add` to get it flagged automatically in future digests."
        )
        await interaction.followup.send(embed=e)
        return

    # Discord allows max 10 embeds per message — chunk if needed
    chunks = [embeds[i:i+10] for i in range(0, len(embeds), 10)]
    for chunk in chunks:
        await interaction.followup.send(embeds=chunk)


# ══════════════════════════════════════════════════════════════════════════════
# /digest  command  — trigger today's full briefing on demand
# ══════════════════════════════════════════════════════════════════════════════

@bot.tree.command(name="digest", description="Trigger today's full AI briefing right now")
async def digest(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)

    # Run daily_briefing.py as a subprocess to avoid duplicating all the logic
    import subprocess
    env = {**os.environ}
    proc = await asyncio.create_subprocess_exec(
        "python", "daily_briefing.py",
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

    if proc.returncode == 0:
        e = discord.Embed(color=COLOR_OK)
        e.description = "✅ **Today's AI briefing has been posted!** Check the digest channel."
        await interaction.followup.send(embed=e)
    else:
        e = discord.Embed(color=COLOR_ERR)
        e.description = (
            f"❌ Briefing failed with exit code `{proc.returncode}`\n"
            f"```{stderr.decode()[-800:]}```"
        )
        await interaction.followup.send(embed=e)


# ══════════════════════════════════════════════════════════════════════════════
# /help  command
# ══════════════════════════════════════════════════════════════════════════════

@bot.tree.command(name="help", description="Show all available commands")
async def help_cmd(interaction: discord.Interaction):
    e = discord.Embed(title="🤖 AI Intelligence Bot — Commands", color=COLOR_KW)
    e.description = (
        "**Keyword Management**\n"
        "`/keyword add <word>` — Track a keyword in daily digests\n"
        "`/keyword remove <word>` — Stop tracking a keyword\n"
        "`/keyword list` — See all your tracked keywords\n"
        "`/keyword clear` — Remove all keywords\n\n"
        "**Search**\n"
        "`/search <query>` — Search papers, models & news right now\n"
        "`/search <query> scope:Papers only` — Narrow search scope\n\n"
        "**Digest**\n"
        "`/digest` — Trigger today's full briefing on demand\n\n"
        "**Tips**\n"
        "• Keywords are case-insensitive and matched against titles, abstracts, and READMEs\n"
        "• Matching items are flagged with 🔑 in the daily digest\n"
        "• `/search` scans the last 7 days of content\n"
        "• Use `scope:` to narrow `/search` to just papers, models, or news"
    )
    await interaction.response.send_message(embed=e)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)