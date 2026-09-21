"""The production image publishes ``effort_levels`` only in a topology where they
mean what the gateway takes them to mean.

The levels are learned on ``/v1/chat/completions`` (startup probe + the model's
own 400s). oh-my-gateway reads them off the relayed ``/v1/models`` and turns them
into the strong ``reasoning_effort: true`` capability — "the requested effort
reaches the model". That holds for a ``/v1/messages`` request only when the
bridge carries its ``output_config.effort`` onto that same ``reasoning_effort``
field and path (``anthropic_request_to_openai_body`` → clamp → retry-on-400);
with the bridge off the request is relayed raw to LiteLLM's Anthropic adapter,
where none of that applies. So the production image and compose must run the
sanitizer in front of LiteLLM **with the bridge on**, exactly like the dev
topology. This pins the pairing so it cannot drift apart silently.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env_from_dockerfile(text: str) -> dict:
    out = {}
    for m in re.finditer(r"^ENV\s+([A-Z0-9_]+)=(\S+)", text, re.MULTILINE):
        out[m.group(1)] = m.group(2)
    return out


def _env_from_compose(text: str) -> dict:
    out = {}
    for m in re.finditer(r"^\s*-\s*([A-Z0-9_]+)=(\S+)", text, re.MULTILINE):
        out[m.group(1)] = m.group(2)
    return out


def test_production_dockerfile_runs_the_sanitizer_in_front_with_the_bridge_on():
    text = (ROOT / "Dockerfile").read_text()
    env = _env_from_dockerfile(text)
    assert 'ENTRYPOINT ["/app/entrypoint.sh"]' in text, "LiteLLM alone publishes no effort_levels"
    assert "COPY sanitizer /app/sanitizer" in text
    assert env.get("SANITIZER_USE_OPENAI_BRIDGE", "").lower() == "true", (
        "bridge off = /v1/messages relayed raw to LiteLLM's Anthropic adapter; the "
        "published effort_levels would not describe the path clients use"
    )
    assert env.get("UPSTREAM_BASE_URL") == f"http://localhost:{env.get('LITELLM_PORT', '3999')}"
    assert re.search(r"^EXPOSE\s+.*\b3996\b", text, re.MULTILINE), "clients must be able to reach the sanitizer"


def test_production_compose_matches_the_dev_topology():
    prod = _env_from_compose((ROOT / "docker-compose.yml").read_text())
    dev = _env_from_compose((ROOT / "docker-compose-dev.yml").read_text())
    assert prod.get("SANITIZER_USE_OPENAI_BRIDGE", "").lower() == "true"
    assert dev.get("SANITIZER_USE_OPENAI_BRIDGE", "").lower() == "true", "dev is the E2E-verified reference"
    assert prod.get("UPSTREAM_BASE_URL") == f"http://localhost:{prod.get('LITELLM_PORT', '3999')}"
    # The probe authenticates against LiteLLM with this key; without it every
    # probe request is a 401 and nothing is learned at startup.
    assert "LITELLM_MASTER_KEY" in prod
