"""Unit tests for the Perplexity community web search tool."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

ENDPOINT = "https://api.perplexity.ai/search"

pytestmark = pytest.mark.anyio


@pytest.fixture
def mock_config_with_key():
    with patch("deerflow.community.perplexity.tools.get_app_config") as mock:
        tool_config = MagicMock()
        tool_config.model_extra = {"api_key": "test-pplx-key", "max_results": 5}
        mock.return_value.get_tool_config.return_value = tool_config
        yield mock


def _make_response(payload: object) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _patch_post(mock_resp: MagicMock):
    """Patch httpx.AsyncClient so the context-managed .post returns mock_resp."""
    patcher = patch("deerflow.community.perplexity.tools.httpx.AsyncClient")
    mock_client_cls = patcher.start()
    mock_client_cls.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_resp)
    return patcher, mock_client_cls


def _post_call(mock_client_cls):
    return mock_client_cls.return_value.__aenter__.return_value.post.call_args


def _result(**overrides) -> dict:
    result = {"title": "A", "url": "https://ex.com/a", "snippet": "s1", "date": "2026-10-27", "last_updated": "2026-10-28"}
    result.update(overrides)
    return result


async def _invoke(args: dict, payload: object = None, mock_resp: MagicMock | None = None):
    patcher, mock_client_cls = _patch_post(mock_resp or _make_response(payload if payload is not None else {"results": [_result()]}))
    try:
        from deerflow.community.perplexity.tools import web_search_tool

        output = await web_search_tool.ainvoke(args)
    finally:
        patcher.stop()
    return json.loads(output), mock_client_cls


class TestResolveToolSettings:
    def test_returns_config_key_when_present(self):
        with patch("deerflow.community.perplexity.tools.get_app_config") as mock:
            tool_config = MagicMock()
            tool_config.model_extra = {"api_key": "from-config"}
            mock.return_value.get_tool_config.return_value = tool_config

            from deerflow.community.perplexity.tools import _resolve_tool_settings

            assert _resolve_tool_settings("web_search")[0] == "from-config"

    def test_falls_back_to_env_when_config_key_blank(self):
        with patch("deerflow.community.perplexity.tools.get_app_config") as mock:
            tool_config = MagicMock()
            tool_config.model_extra = {"api_key": "   "}
            mock.return_value.get_tool_config.return_value = tool_config
            with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "env-key"}, clear=True):
                from deerflow.community.perplexity.tools import _resolve_tool_settings

                assert _resolve_tool_settings("web_search")[0] == "env-key"

    async def test_missing_key_returns_error_without_request(self):
        with patch("deerflow.community.perplexity.tools.get_app_config") as mock:
            mock.return_value.get_tool_config.return_value = None
            with patch.dict("os.environ", {}, clear=True):
                parsed, mock_client_cls = await _invoke({"query": "test"})

        assert "PERPLEXITY_API_KEY" in parsed["error"]
        mock_client_cls.assert_not_called()


class TestRequest:
    async def test_posts_query_with_bearer_key_and_integration_header(self, mock_config_with_key):
        _, mock_client_cls = await _invoke({"query": "test"})
        call = _post_call(mock_client_cls)

        assert call.args[0] == ENDPOINT
        assert call.kwargs["headers"]["Authorization"] == "Bearer test-pplx-key"
        assert call.kwargs["headers"]["X-Pplx-Integration"] == "deer-flow"
        assert call.kwargs["json"] == {"query": "test", "max_results": 5}

    @pytest.mark.parametrize("time_range", ["day", "week", "month", "year"])
    async def test_time_range_maps_to_search_recency_filter(self, time_range, mock_config_with_key):
        _, mock_client_cls = await _invoke({"query": "test", "time_range": time_range})

        assert _post_call(mock_client_cls).kwargs["json"]["search_recency_filter"] == time_range

    async def test_agent_max_results_is_honored_over_config(self, mock_config_with_key):
        _, mock_client_cls = await _invoke({"query": "test", "max_results": 2})

        assert _post_call(mock_client_cls).kwargs["json"]["max_results"] == 2

    @pytest.mark.parametrize(("max_results", "expected"), [(0, 5), (-2, 5), (1, 1), (20, 20), ("500", 20), (float("inf"), 5), (True, 5), (3.5, 5)])
    async def test_configured_max_results_coerced_and_capped(self, max_results, expected):
        with patch("deerflow.community.perplexity.tools.get_app_config") as mock:
            tool_config = MagicMock()
            tool_config.model_extra = {"api_key": "k", "max_results": max_results}
            mock.return_value.get_tool_config.return_value = tool_config
            _, mock_client_cls = await _invoke({"query": "test"})

        assert _post_call(mock_client_cls).kwargs["json"]["max_results"] == expected

    async def test_invalid_max_results_warns_under_provider_label(self, caplog):
        with patch("deerflow.community.perplexity.tools.get_app_config") as mock:
            tool_config = MagicMock()
            tool_config.model_extra = {"api_key": "k", "max_results": "not-a-number"}
            mock.return_value.get_tool_config.return_value = tool_config
            with caplog.at_level(logging.WARNING, logger="deerflow.community.perplexity.tools"):
                await _invoke({"query": "test"})

        assert any("Invalid Perplexity max_results" in record.message for record in caplog.records)


class TestResponse:
    async def test_returns_normalized_results(self, mock_config_with_key):
        parsed, _ = await _invoke({"query": "test"})

        assert parsed == [{"title": "A", "url": "https://ex.com/a", "snippet": "s1", "date": "2026-10-27"}]

    async def test_skips_non_object_entries_and_defaults_bad_fields(self, mock_config_with_key):
        parsed, _ = await _invoke({"query": "test"}, {"results": ["bad", {"title": 7}]})

        assert parsed == [{"title": "", "url": "", "snippet": "", "date": ""}]

    @pytest.mark.parametrize("payload", [{"results": []}, {"id": "x"}, {"results": None}])
    async def test_empty_or_missing_results_returns_no_results(self, payload, mock_config_with_key):
        parsed, _ = await _invoke({"query": "nothing"}, payload)

        assert parsed == {"error": "No results found", "query": "nothing"}

    @pytest.mark.parametrize("payload", [["not", "a", "dict"], {"results": {"web": []}}, {"results": ["bad", 1]}])
    async def test_malformed_payload_returns_format_error(self, payload, mock_config_with_key):
        parsed, _ = await _invoke({"query": "test"}, payload)

        assert parsed["error"] == "Perplexity returned an unexpected response format"

    @pytest.mark.parametrize(("status", "hint"), [(401, "invalid or expired PERPLEXITY_API_KEY"), (429, "rate limit reached"), (500, None)])
    async def test_http_error_returns_structured_error(self, status, hint, mock_config_with_key):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(str(status), request=MagicMock(), response=MagicMock(status_code=status, text="boom"))
        parsed, _ = await _invoke({"query": "test"}, mock_resp=mock_resp)

        assert parsed["error"] == f"Perplexity API error: HTTP {status}" + (f" ({hint})" if hint else "")

    async def test_network_exception_returns_error_without_details(self, mock_config_with_key):
        patcher, mock_client_cls = _patch_post(MagicMock())
        mock_client_cls.return_value.__aenter__.return_value.post.side_effect = httpx.ConnectError("test-pplx-key leaked")
        try:
            from deerflow.community.perplexity.tools import web_search_tool

            parsed = json.loads(await web_search_tool.ainvoke({"query": "test"}))
        finally:
            patcher.stop()

        assert parsed["error"] == "Perplexity search failed: ConnectError"


def test_package_exports_web_search_tool():
    from deerflow.community.perplexity import web_search_tool
    from deerflow.community.perplexity.tools import web_search_tool as direct_web_search_tool

    assert web_search_tool is direct_web_search_tool
