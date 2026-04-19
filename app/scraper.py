import logging
from typing import Optional

import requests
import httpx
from bs4 import BeautifulSoup

from app.decorators import retry

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 10         
MIN_PARAGRAPH_LEN = 50
MAX_PARAGRAPHS = 15

def _validate_url(url: str) -> None:
    """Lève ValueError si l'URL n'est pas valide."""
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"URL invalide (doit commencer par http/https) : {url}")

def _parse_html(html: str, url: str) -> dict:
    
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else "Titre introuvable"
    paragraphs = [
        p.get_text(strip=True)
        for p in soup.find_all("p")
        if len(p.get_text(strip=True)) >= MIN_PARAGRAPH_LEN
    ]
    body = " ".join(paragraphs[:MAX_PARAGRAPHS])

    if not body:
        raise ValueError("Aucun contenu textuel trouvé dans la page.")

    logger.info("[SCRAPER] '%s' → %d paragraphes extraits", url[:60], len(paragraphs))

    return {"url": url, "title": title, "text": body}

@retry(max_attempts=3, delay=1.5, exceptions=(requests.RequestException,))
def scrape_article(url: str) -> dict:
    _validate_url(url)
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()       
    return _parse_html(response.text, url)


async def async_scrape_article(url: str) -> dict:
    _validate_url(url)

    async with httpx.AsyncClient(headers=HEADERS, timeout=TIMEOUT, follow_redirects=True) as client:
        for attempt in range(1, 4):
            try:
                response = await client.get(url)
                response.raise_for_status()
                return _parse_html(response.text, url)
            except httpx.HTTPError as exc:
                logger.warning("[ASYNC SCRAPER] tentative %d/3 — %s", attempt, exc)
                if attempt == 3:
                    raise