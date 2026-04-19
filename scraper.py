import logging
import random
import socket

import certifi
import httpx
import requests
import urllib3
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

TIMEOUT = 15
MIN_PARAGRAPH_LEN = 40
MAX_PARAGRAPHS = 15

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]


def _get_headers() -> dict:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Cache-Control": "max-age=0",
    }


def _validate_url(url: str) -> None:
    if not url.startswith(("http://", "https://")):
        raise ValueError("L'URL doit commencer par http:// ou https://")


def _parse_html(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    else:
        og = soup.find("meta", property="og:title")
        title = og["content"] if og else (soup.title.string.strip() if soup.title else "Titre introuvable")

    paragraphs = [
        p.get_text(strip=True)
        for p in soup.find_all("p")
        if len(p.get_text(strip=True)) >= MIN_PARAGRAPH_LEN
    ]
    body = " ".join(paragraphs[:MAX_PARAGRAPHS])

    if not body:
        body = soup.get_text(separator=" ", strip=True)[:3000]

    if not body.strip():
        raise ValueError("Aucun contenu textuel trouvé dans cette page.")

    logger.info("[SCRAPER] '%s' → %d paragraphes", url[:60], len(paragraphs))
    return {"url": url, "title": title, "text": body}


def _friendly_error(exc: Exception, url: str) -> str:
    """
    Traduit n'importe quelle exception réseau en message lisible pour l'utilisateur.
    Aucune stack trace technique n'est exposée côté client.
    """
    domain = url.split("/")[2] if "/" in url else url
    msg = str(exc).lower()

    if "getaddrinfo" in msg or "nodename nor servname" in msg or "name or service not known" in msg:
        return (
            f"Impossible de joindre '{domain}' — vérifiez votre connexion internet "
            "ou l'URL saisie (nom de domaine introuvable)."
        )
    if "timed out" in msg or "timeout" in msg:
        return f"Le site '{domain}' met trop de temps à répondre. Réessayez dans quelques secondes."
    if "ssl" in msg or "certificate" in msg:
        return f"Problème de certificat SSL avec '{domain}'. Essayez un autre site."
    if "403" in msg or "forbidden" in msg:
        return (
            f"'{domain}' bloque les accès automatiques (403 Forbidden). "
            "Essayez : bbc.com, reuters.com, lemonde.fr, aljazeera.com, theguardian.com"
        )
    if "404" in msg or "not found" in msg:
        return f"Page introuvable sur '{domain}' (404). Vérifiez que l'URL est correcte."
    if "500" in msg or "502" in msg or "503" in msg:
        return f"Le serveur de '{domain}' est momentanément indisponible. Réessayez plus tard."
    if "connexion" in msg or "connection" in msg:
        return f"Connexion refusée par '{domain}'. Le site est peut-être hors ligne."

    logger.error("[SCRAPER] Erreur non classifiée pour '%s' : %s", domain, exc)
    return f"Impossible d'analyser cet article depuis '{domain}'. Essayez avec un autre lien."

async def async_scrape_article(url: str) -> dict:
    _validate_url(url)
    headers = _get_headers()

    async def _fetch(verify_ssl) -> dict:
        async with httpx.AsyncClient(
            headers=headers,
            timeout=TIMEOUT,
            follow_redirects=True,
            verify=verify_ssl,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            return _parse_html(response.text, url)

    try:
        try:
            return await _fetch(verify_ssl=certifi.where())
        except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            if "ssl" in str(exc).lower() or "certificate" in str(exc).lower():
                logger.warning("[SCRAPER] SSL échoué → retry verify=False pour '%s'", url[:60])
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                return await _fetch(verify_ssl=False)
            raise

    except Exception as exc:
        raise ValueError(_friendly_error(exc, url)) from None
    
def scrape_article(url: str) -> dict:
    """Version synchrone avec requests — même gestion d'erreurs."""
    _validate_url(url)
    session = requests.Session()
    session.headers.update(_get_headers())

    try:
        try:
            response = session.get(url, timeout=TIMEOUT, verify=certifi.where())
            response.raise_for_status()
        except requests.exceptions.SSLError:
            logger.warning("[SCRAPER] SSL échoué → retry verify=False pour '%s'", url[:60])
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = session.get(url, timeout=TIMEOUT, verify=False)
            response.raise_for_status()

        return _parse_html(response.text, url)

    except Exception as exc:
        raise ValueError(_friendly_error(exc, url)) from None