"""Shared fixtures for route tests.

Injects an ``httpx.MockTransport`` in place of the real upstream client so route
handlers can be exercised without a live LiteLLM/vLLM backend.
"""

import httpx
import pytest

from sanitizer import routes_messages, routes_passthrough


@pytest.fixture
def upstream(monkeypatch):
    """Patch both routers' ``_make_client`` to a MockTransport-backed client.

    Returns a small controller: call ``upstream.set_handler(fn)`` with a function
    ``fn(request) -> httpx.Response``; inspect ``upstream.requests`` afterwards.
    """

    class Controller:
        def __init__(self):
            self.requests = []
            self._handler = lambda request: httpx.Response(200, json={})

        def set_handler(self, handler):
            self._handler = handler

        def _factory(self, timeout):
            def _handle(request):
                self.requests.append(request)
                return self._handler(request)

            return httpx.AsyncClient(transport=httpx.MockTransport(_handle))

    controller = Controller()
    monkeypatch.setattr(routes_messages, "_make_client", controller._factory)
    monkeypatch.setattr(routes_passthrough, "_make_client", controller._factory)
    return controller
