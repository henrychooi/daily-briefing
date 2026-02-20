import os
import json
import requests
import feedparser
from datetime import datetime, timedelta
from huggingface_hub import HfApi

# Configuration
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
HISTORY_FILE = "seen_ids.json"

# --- DATA SOURCES ---

def get_huggingface_daily_papers():
    """Fetches top 5 papers from HF Daily Papers"""
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://huggingface.co/api/daily_papers?date={today}"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return []
            
        data = response.json()
        papers = []
        
        # Take top 5 to avoid overflow
        for paper in data[:5]:
            p = paper["paper"]
            papers.append({
                "id": p["id"],
                "title": p["title"],
                "link": f"https://huggingface.co/papers/{p['id']}",
                "type": "Paper"
            })
        return papers
    except Exception as e:
        print(f"Error fetching papers: {e}")
        return []

def get_trending_models():
    """Fetches top 5 trending LLM/Speech models from HF"""
    api = HfApi()
    models = api.list_models(
        filter=["text-generation", "automatic-speech-recognition"],
        sort="likes",
        direction=-1,
        limit=20,
        full=True
    )
    
    new_models = []
    for m in models:
        # Check if created/updated in last 48 hours
        if m.lastModified and (datetime.now().astimezone() - m.lastModified).days < 2:
            new_models.append({
                "id": m.modelId,
                "title": m.modelId,
                "link": f"https://huggingface.co/{m.modelId}",
                "likes": m.likes,
                "type": "Model"
            })
            if len(new_models) >= 5: break # Hard limit
            
    return new_models

def get_lab_news():
    """Fetches latest blog posts from major AI labs"""
    feeds = [
        ("OpenAI", "https://openai.com/index.xml"),
        ("Anthropic", "https://www.anthropic.com/feed"),
        ("DeepMind", "https://deepmind.google/blog/rss.xml"),
        ("Hugging Face", "https://huggingface.co/blog/feed.xml")
    ]
    
    news = []
    for source, url in feeds:
        try:
            feed = feedparser.parse(url)
            if not feed.entries: continue
            
            entry = feed.entries[0] # Just check the very latest one
            published = datetime(*entry.published_parsed[:6])
            
            if (datetime.now() - published).days < 1:
                news.append({
                    "id": entry.link,
                    "title": f"{source}: {entry.title}",
                    "link": entry.link,
                    "type": "News"
                })
        except:
            continue
    return news

# --- CORE LOGIC ---

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_history(new_ids, old_history):
    # Combine and keep last 1000
    updated = list(old_history.union(new_ids))[-1000:]
    with open(HISTORY_FILE, "w") as f:
        json.dump(updated, f)

def send_discord_digest(items):
    if not items: return

    # 1. Group items by type
    papers = [i for i in items if i['type'] == 'Paper']
    models = [i for i in items if i['type'] == 'Model']
    news = [i for i in items if i['type'] == 'News']

    # 2. Build the Embed Fields
    fields = []
    
    if news:
        text = "\n".join([f"• **[{n['title']}]({n['link']})**" for n in news])
        fields.append({"name": "📢 Industry News", "value": text, "inline": False})

    if models:
        # Format: Name (Likes)
        text = "\n".join([f"• **[{m['title']}]({m['link']})** (⭐ {m['likes']})" for m in models])
        fields.append({"name": "🚀 Trending Models", "value": text, "inline": False})

    if papers:
        text = "\n".join([f"• **[{p['title']}]({p['link']})**" for p in papers])
        fields.append({"name": "📚 Daily Papers", "value": text, "inline": False})

    # 3. Construct Payload
    payload = {
        "embeds": [{
            "title": f"Daily AI Briefing • {datetime.now().strftime('%b %d')}",
            "description": "Here are the top trending updates for today.",
            "color": 3447003, # Blue
            "fields": fields,
            "footer": {"text": "Sources: Hugging Face & Corporate Blogs"}
        }]
    }

    requests.post(WEBHOOK_URL, json=payload)

def main():
    if not WEBHOOK_URL:
        print("No Webhook URL found.")
        return

    history = load_history()
    
    # Fetch all candidates
    candidates = get_huggingface_daily_papers() + get_trending_models() + get_lab_news()
    
    # Filter: Only keep what we haven't seen
    new_items = [item for item in candidates if item['id'] not in history]
    
    if new_items:
        print(f"Found {len(new_items)} new items. Sending digest...")
        send_discord_digest(new_items)
        
        # Update history
        new_ids = {item['id'] for item in new_items}
        save_history(new_ids, history)
    else:
        print("No new items to report.")

if __name__ == "__main__":
    main()
