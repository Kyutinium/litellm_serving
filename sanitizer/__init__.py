"""LiteLLM Sanitizer — a thin Anthropic-facing reverse proxy in front of LiteLLM.

Exposes an Anthropic ``/v1/messages`` endpoint that repairs LiteLLM's malformed
Anthropic SSE and (optionally) bridges to the upstream OpenAI chat route,
working around confirmed LiteLLM adapter bugs. Everything else is passed through
byte-for-byte. Communicates with LiteLLM over HTTP only — never imports it.
"""

__version__ = "0.1.0"
