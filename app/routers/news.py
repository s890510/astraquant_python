from typing import List

import feedparser
from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(prefix="/news", tags=["news"])


@router.get("/latest")
async def latest_news() -> List[dict]:
    items: List[dict] = []
    sources = settings.RSS_SOURCES or [
        "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    ]
    for url in sources:
        try:
            feed = feedparser.parse(url)
            for entry in (feed.entries or [])[:10]:
                items.append({
                    "title": getattr(entry, "title", None),
                    "link": getattr(entry, "link", None),
                    "published": getattr(entry, "published", None),
                    "source": url,
                })
        except Exception:
            # skip faulty source
            continue
    return items
