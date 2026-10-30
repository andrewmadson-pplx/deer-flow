"""Perplexity community web search tool.

Calls the Perplexity Search API (``POST https://api.perplexity.ai/search``),
which returns ranked web results with a title, URL, extracted snippet and
publication date. An API key is required: the tool's ``api_key`` config or
``PERPLEXITY_API_KEY``. Get one at https://www.perplexity.ai/account/api.

The module is self-contained (httpx only, no new dependency). The tool is a
coroutine that sends its request through ``httpx.AsyncClient`` and resolves the
tool config on an executor thread, like the You.com and Webz providers.
"""

import asyncio
import json
import logging
import os

import httpx
from langchain.tools import tool

from deerflow.community.search_max_results import coerce_max_results
from deerflow.community.search_time_range import SearchTimeRange
from deerflow.community.url_safety import safe_public_result_url
from deerflow.config import get_app_config

logger = logging.getLogger(__name__)

_PERPLEXITY_SEARCH_ENDPOINT = "https://api.perplexity.ai/search"
_DEFAULT_MAX_RESULTS = 5
# The Search API accepts 1-20 results for web search.
_MAX_RESULTS_CAP = 20
_TIMEOUT_S = 30.0
_STATUS_HINTS = {
    401: "invalid or expired PERPLEXITY_API_KEY",
    429: "rate limit reached",
}


def _resolve_tool_settings(tool_name: str) -> tuple[str | None, object]:
    """Resolve the API key and the configured ``max_results`` in one config read."""
    config = get_app_config().get_tool_config(tool_name)
    max_results = None
    api_key = None
    if config is not None:
        extra = config.model_extra or {}
        max_results = extra.get("max_results")
        candidate = extra.get("api_key")
        if isinstance(candidate, str) and candidate.strip():
            api_key = candidate.strip()
    if api_key is None:
        env_key = os.getenv("PERPLEXITY_API_KEY")
        if isinstance(env_key, str) and env_key.strip():
            api_key = env_key.strip()
    return api_key, max_results


def _count_for_request(value: object) -> int:
    """Coerce ``max_results`` via the shared owner, then apply the 20-result cap."""
    return min(
        coerce_max_results(value, provider="Perplexity", logger=logger, default=_DEFAULT_MAX_RESULTS),
        _MAX_RESULTS_CAP,
    )


@tool("web_search", parse_docstring=True)
async def web_search_tool(query: str, max_results: int | None = None, time_range: SearchTimeRange | None = None) -> str:
    """Search the web for information using Perplexity.

    Args:
        query: Search keywords describing what you want to find. Be specific for better results.
        max_results: Maximum number of search results to return. If omitted, uses the configured value (default 5). Clamped to 1-20.
        time_range: Optional relative publication/update window. Use only when the request requires recent results.
    """
    # Config resolution can read files; keep it off the event loop.
    api_key, configured_max_results = await asyncio.to_thread(_resolve_tool_settings, "web_search")
    if not api_key:
        return json.dumps(
            {"error": "Perplexity API key is missing; configure api_key or PERPLEXITY_API_KEY", "query": query},
            ensure_ascii=False,
        )
    if max_results is None:
        max_results = configured_max_results
    count = _DEFAULT_MAX_RESULTS if max_results is None else _count_for_request(max_results)

    payload: dict = {"query": query, "max_results": count}
    if time_range is not None:
        payload["search_recency_filter"] = time_range
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "X-Pplx-Integration": "deer-flow",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S, follow_redirects=False) as client:
            response = await client.post(_PERPLEXITY_SEARCH_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        logger.error("Perplexity API returned HTTP %s", status)
        hint = _STATUS_HINTS.get(status)
        message = f"Perplexity API error: HTTP {status}"
        if hint:
            message = f"{message} ({hint})"
        return json.dumps({"error": message, "query": query}, ensure_ascii=False)
    except Exception as e:
        logger.error("Perplexity search failed: %s", type(e).__name__)
        return json.dumps({"error": f"Perplexity search failed: {type(e).__name__}", "query": query}, ensure_ascii=False)

    if not isinstance(data, dict):
        logger.error("Perplexity returned an unexpected payload type: %s", type(data).__name__)
        return json.dumps({"error": "Perplexity returned an unexpected response format", "query": query}, ensure_ascii=False)
    results = data.get("results")
    if results is not None and not isinstance(results, list):
        logger.error("Perplexity returned an unexpected results type: %s", type(results).__name__)
        return json.dumps({"error": "Perplexity returned an unexpected response format", "query": query}, ensure_ascii=False)
    if not results:
        return json.dumps({"error": "No results found", "query": query}, ensure_ascii=False)

    entries = [r for r in results if isinstance(r, dict)]
    if not entries:
        logger.error("Perplexity returned %d results with no usable objects", len(results))
        return json.dumps({"error": "Perplexity returned an unexpected response format", "query": query}, ensure_ascii=False)

    normalized_results = [
        {
            "title": r.get("title") if isinstance(r.get("title"), str) else "",
            "url": safe_public_result_url(r.get("url")),
            "snippet": r.get("snippet") if isinstance(r.get("snippet"), str) else "",
            "date": r.get("date") if isinstance(r.get("date"), str) else "",
        }
        for r in entries[:count]
    ]
    return json.dumps(normalized_results, indent=2, ensure_ascii=False)
