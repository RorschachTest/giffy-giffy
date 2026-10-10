"""Short share links that chat apps unfurl as the video itself.

    GET /c/{share_id}          the link people paste: video page + preview tags
    GET /c/{share_id}/embed    bare player, for apps that embed one (Slack, X)
    GET /oembed?url=...        oEmbed JSON, for apps that ask for it

The video never leaves the server. A chat app that sees the link fetches the
page, reads its Open Graph / Twitter card / oEmbed tags and draws the preview:
iMessage, Discord and Telegram play `og:video` inline, Slack plays the
`twitter:player` embed, WhatsApp shows the thumbnail and opens the page.

share_id is the first 10 hex characters of the file's SHA-256: stable, needs no
extra column, and the same clip always gets the same link.

Those apps can only fetch a public https address, so set PUBLIC_BASE_URL to it
(see the Cloudflare Tunnel note in README.md). Without it links use the address
the request came in on, which is right for localhost testing.
"""
from __future__ import annotations

import html
from urllib.parse import urlencode, urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from . import config, db

router = APIRouter()

SHARE_ID_LEN = 10
THUMB_HEIGHT = 360   # media.make_thumb height


def public_base(request: Request | None) -> str:
    if config.PUBLIC_BASE_URL:
        return config.PUBLIC_BASE_URL.rstrip("/")
    return str(request.base_url).rstrip("/") if request else ""


def share_url(base: str, share_id: str) -> str:
    return f"{base}/c/{share_id}"


def _clip(share_id: str) -> dict:
    share_id = share_id.lower()
    if len(share_id) != SHARE_ID_LEN or any(ch not in "0123456789abcdef" for ch in share_id):
        raise HTTPException(404, "no such clip")
    with db.session() as conn:
        row = conn.execute(
            """SELECT id, left(sha256, 10) AS share_id, file, thumb, duration, width, height,
                      title, folk_names, transcript_roman, use_when, reactions, people, source_urls
                 FROM clips WHERE left(sha256, 10) = %s""",
            (share_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "no such clip")
    return row


def _facts(c: dict, base: str) -> dict:
    w, h = c["width"] or 640, c["height"] or 360
    return {
        "name": (c["folk_names"] or [c["title"] or "meme"])[0],
        "said": c["transcript_roman"] or "",
        "about": ", ".join([*c["reactions"][:2], *c["use_when"][:1]]),
        "page": share_url(base, c["share_id"]),
        "embed": f"{share_url(base, c['share_id'])}/embed",
        "video": f"{base}/media/{c['file']}",
        "thumb": f"{base}/media/{c['thumb']}" if c["thumb"] else "",
        "width": w, "height": h,
        "thumb_width": round(THUMB_HEIGHT * w / h) if h else THUMB_HEIGHT, "thumb_height": THUMB_HEIGHT,
        "source": next((u for u in c.get("source_urls") or [] if urlparse(u).scheme in ("http", "https")), ""),
    }


def _meta_tags(f: dict, base: str) -> str:
    description = f"“{f['said']}”" if f["said"] else (f["about"] or "A meme clip on memeclip")
    tags = [
        ("property", "og:site_name", "memeclip"),
        ("property", "og:type", "video.other"),
        ("property", "og:title", f["name"]),
        ("property", "og:description", description),
        ("property", "og:url", f["page"]),
        ("property", "og:image", f["thumb"]),
        ("property", "og:image:width", f["thumb_width"]),
        ("property", "og:image:height", f["thumb_height"]),
        ("property", "og:video", f["video"]),
        ("property", "og:video:url", f["video"]),
        ("property", "og:video:secure_url", f["video"]),
        ("property", "og:video:type", "video/mp4"),
        ("property", "og:video:width", f["width"]),
        ("property", "og:video:height", f["height"]),
        ("name", "twitter:card", "player"),
        ("name", "twitter:title", f["name"]),
        ("name", "twitter:description", description),
        ("name", "twitter:image", f["thumb"]),
        ("name", "twitter:player", f["embed"]),
        ("name", "twitter:player:width", f["width"]),
        ("name", "twitter:player:height", f["height"]),
        ("name", "twitter:player:stream", f["video"]),
        ("name", "twitter:player:stream:content_type", "video/mp4"),
    ]
    out = [f'<meta {k}="{v}" content="{html.escape(str(c), quote=True)}">' for k, v, c in tags if c != ""]
    oembed = f"{base}/oembed?" + urlencode({"url": f["page"], "format": "json"})
    out.append(f'<link rel="alternate" type="application/json+oembed" href="{html.escape(oembed, quote=True)}">')
    out.append(f'<link rel="canonical" href="{html.escape(f["page"], quote=True)}">')
    return "\n".join(out)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title}</title>
<link rel="icon" href="/static/icon.svg" type="image/svg+xml">
{meta}
<style>
  :root {{ color-scheme: dark; }}
  html, body {{ margin: 0; height: 100%; background: #0b0b0c; color: #f2f0eb;
               font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ min-height: 100%; display: grid; grid-template-rows: 1fr auto; }}
  .stage {{ display: grid; place-items: center; min-height: 60vh; }}
  video {{ width: 100%; max-height: 82vh; background: #000; display: block; }}
  .unmute {{ position: fixed; top: max(14px, env(safe-area-inset-top)); right: 14px; border: 0;
            border-radius: 999px; padding: 8px 14px; background: rgba(255,255,255,.16); color: inherit;
            font: inherit; cursor: pointer; backdrop-filter: blur(6px); }}
  .unmute[hidden] {{ display: none; }}
  footer {{ padding: 14px 18px max(18px, env(safe-area-inset-bottom)); display: grid; gap: 4px; }}
  h1 {{ margin: 0; font-size: 18px; }}
  .said {{ color: #b9b5ac; font-style: italic; }}
  a {{ color: #ff8a5c; }}
  .links {{ display: flex; flex-wrap: wrap; gap: 4px 16px; }}
</style>
</head>
<body>
<main>
  <div class="stage">
    <video id="v" src="{video}" poster="{thumb}" autoplay muted loop playsinline controls preload="auto"></video>
  </div>
  <footer>
    <h1>{title}</h1>
    {said}
    <div class="links">{source}<a href="{home}">Find another meme</a><a href="{legal}">Removal requests</a></div>
  </footer>
</main>
<button class="unmute" id="u" type="button">Tap for sound</button>
<script>
  const v = document.getElementById("v"), u = document.getElementById("u");
  u.onclick = () => {{ v.muted = false; v.play(); u.hidden = true; }};
  v.onvolumechange = () => {{ if (!v.muted) u.hidden = true; }};
</script>
</body>
</html>
"""

EMBED = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>html,body{{margin:0;height:100%;background:#000}}video{{width:100%;height:100%;object-fit:contain;display:block}}</style>
</head><body>
<video src="{video}" poster="{thumb}" autoplay muted loop playsinline controls></video>
</body></html>
"""


@router.get("/c/{share_id}", response_class=HTMLResponse)
def share_page(share_id: str, request: Request) -> HTMLResponse:
    base = public_base(request)
    f = _facts(_clip(share_id), base)
    e = lambda s: html.escape(str(s), quote=True)  # noqa: E731
    return HTMLResponse(PAGE.format(
        title=e(f["name"]), meta=_meta_tags(f, base), video=e(f["video"]), thumb=e(f["thumb"]),
        said=f'<div class="said">“{e(f["said"])}”</div>' if f["said"] else "", home=e(base + "/"),
        source=f'<a href="{e(f["source"])}" rel="noopener nofollow">Source</a>' if f["source"] else "",
        legal=e(base + "/legal"),
    ))


@router.get("/c/{share_id}/embed", response_class=HTMLResponse)
def share_embed(share_id: str, request: Request) -> HTMLResponse:
    f = _facts(_clip(share_id), public_base(request))
    e = lambda s: html.escape(str(s), quote=True)  # noqa: E731
    return HTMLResponse(EMBED.format(title=e(f["name"]), video=e(f["video"]), thumb=e(f["thumb"])))


@router.get("/oembed")
def oembed(url: str, request: Request, maxwidth: int | None = None) -> dict:
    path = urlparse(url).path.rstrip("/").split("/")
    if len(path) < 3 or path[-2] != "c":
        raise HTTPException(404, "not a memeclip link")
    base = public_base(request)
    f = _facts(_clip(path[-1]), base)
    w, h = f["width"], f["height"]
    if maxwidth and maxwidth < w:
        w, h = maxwidth, round(h * maxwidth / w)
    return {
        "version": "1.0", "type": "video", "provider_name": "memeclip", "provider_url": base + "/",
        "title": f["name"], "width": w, "height": h,
        "thumbnail_url": f["thumb"], "thumbnail_width": f["thumb_width"], "thumbnail_height": f["thumb_height"],
        "html": (f'<iframe src="{html.escape(f["embed"], quote=True)}" width="{w}" height="{h}" '
                 'frameborder="0" allow="autoplay; fullscreen" allowfullscreen></iframe>'),
    }
