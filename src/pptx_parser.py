"""Extract text from .pptx without heavy deps — a pptx is a zip of XML.

Native text comes from shape/table runs (<a:t>) joined per paragraph (<a:p>),
plus speaker notes. When an OCR gateway is configured (src.ocr_client), embedded
slide images are OCR'd too — important for decks where data lives in infographics.
"""

import html
import logging
import re
import zipfile

from src.config import config
from src.ocr_client import ocr_enabled, ocr_image_bytes

logger = logging.getLogger(__name__)

_PARA = re.compile(r"<a:p>(.*?)</a:p>", re.S)
_RUN = re.compile(r"<a:t>(.*?)</a:t>", re.S)
_SLIDE = re.compile(r"ppt/slides/slide(\d+)\.xml$")
_NOTES = re.compile(r"ppt/notesSlides/notesSlide(\d+)\.xml$")
_MEDIA = re.compile(r"ppt/media/.+\.(?:png|jpe?g|tiff?|bmp|webp)$", re.I)


def _paragraphs(xml: str) -> list[str]:
    """Reconstruct readable lines: join runs within a paragraph, drop empties."""
    lines = []
    for para in _PARA.findall(xml):
        text = "".join(html.unescape(t) for t in _RUN.findall(para)).strip()
        if text:
            lines.append(text)
    return lines


def parse_pptx(filepath: str) -> str:
    """Return all extractable text from a .pptx (native + optional image OCR)."""
    with zipfile.ZipFile(filepath) as z:
        names = z.namelist()
        slides = sorted(
            (n for n in names if _SLIDE.search(n)),
            key=lambda n: int(_SLIDE.search(n).group(1)),
        )
        notes = {int(m.group(1)): n for n in names if (m := _NOTES.search(n))}

        blocks = []
        for name in slides:
            idx = int(_SLIDE.search(name).group(1))
            lines = _paragraphs(z.read(name).decode("utf-8", "ignore"))
            if idx in notes:
                lines += [
                    ln
                    for ln in _paragraphs(z.read(notes[idx]).decode("utf-8", "ignore"))
                    if not ln.isdigit()  # notes slides carry the slide number alone
                ]
            if lines:
                blocks.append("\n".join(lines))

        if ocr_enabled():
            blocks.extend(_ocr_media(z, names))

        return "\n\n".join(blocks)


def _ocr_media(z: zipfile.ZipFile, names: list[str]) -> list[str]:
    """OCR each embedded image above the size floor. Best-effort per image."""
    out = []
    for name in names:
        if not _MEDIA.search(name):
            continue
        data = z.read(name)
        # ponytail: byte-size floor skips icons/logos; lower OCR_MIN_IMAGE_BYTES
        # if real text in small images is being missed.
        if len(data) < config.OCR_MIN_IMAGE_BYTES:
            continue
        text = ocr_image_bytes(data, name.rsplit("/", 1)[-1]).strip()
        if text:
            out.append(text)
    logger.info("OCR extracted text from %d slide image(s)", len(out))
    return out
