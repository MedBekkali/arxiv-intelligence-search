"""Helpers for building canonical arXiv URLs from paper IDs."""

ARXIV_ABS  = "https://arxiv.org/abs/{id}"
ARXIV_PDF  = "https://arxiv.org/pdf/{id}"


def abs_url(arxiv_id: str) -> str:
    """Return the abstract page URL for *arxiv_id*."""
    return ARXIV_ABS.format(id=arxiv_id.strip())


def pdf_url(arxiv_id: str) -> str:
    """Return the PDF URL for *arxiv_id*."""
    return ARXIV_PDF.format(id=arxiv_id.strip())


def paper_links(arxiv_id: str) -> dict[str, str]:
    """Return a dict with both ``abs_url`` and ``pdf_url`` keys."""
    _id = arxiv_id.strip()
    return {"abs_url": abs_url(_id), "pdf_url": pdf_url(_id)}