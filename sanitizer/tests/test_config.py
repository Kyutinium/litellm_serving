"""Unit tests for sanitizer.config env-var resolution and fallbacks."""

from sanitizer import config


def test_upstream_url_default(monkeypatch):
    monkeypatch.delenv("UPSTREAM_BASE_URL", raising=False)
    assert config.get_upstream_url() == "http://localhost:3999"


def test_upstream_url_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("UPSTREAM_BASE_URL", "http://host:1234/")
    assert config.get_upstream_url() == "http://host:1234"


def test_port_default_and_invalid(monkeypatch):
    monkeypatch.delenv("SANITIZER_PORT", raising=False)
    assert config.get_port() == 3996
    monkeypatch.setenv("SANITIZER_PORT", "not-a-number")
    assert config.get_port() == 3996
    monkeypatch.setenv("SANITIZER_PORT", "5501")
    assert config.get_port() == 5501


def test_tls_verify_truthy_values(monkeypatch):
    for value in ("true", "True", "1", "yes", "on"):
        monkeypatch.setenv("SANITIZER_TLS_VERIFY", value)
        assert config.get_tls_verify() is True


def test_tls_verify_falsy_values(monkeypatch):
    for value in ("false", "False", "0", "no", "off"):
        monkeypatch.setenv("SANITIZER_TLS_VERIFY", value)
        assert config.get_tls_verify() is False


def test_tls_verify_ca_path(monkeypatch):
    monkeypatch.setenv("SANITIZER_TLS_VERIFY", "/etc/ssl/certs/ca.pem")
    assert config.get_tls_verify() == "/etc/ssl/certs/ca.pem"


def test_tls_verify_default_true(monkeypatch):
    monkeypatch.delenv("SANITIZER_TLS_VERIFY", raising=False)
    assert config.get_tls_verify() is True


def test_request_timeout_zero_is_none(monkeypatch):
    monkeypatch.setenv("SANITIZER_REQUEST_TIMEOUT", "0")
    assert config.get_request_timeout_seconds() is None


def test_request_timeout_default_none(monkeypatch):
    monkeypatch.delenv("SANITIZER_REQUEST_TIMEOUT", raising=False)
    assert config.get_request_timeout_seconds() is None


def test_request_timeout_negative_and_invalid_none(monkeypatch):
    monkeypatch.setenv("SANITIZER_REQUEST_TIMEOUT", "-5")
    assert config.get_request_timeout_seconds() is None
    monkeypatch.setenv("SANITIZER_REQUEST_TIMEOUT", "abc")
    assert config.get_request_timeout_seconds() is None


def test_request_timeout_positive(monkeypatch):
    monkeypatch.setenv("SANITIZER_REQUEST_TIMEOUT", "30")
    assert config.get_request_timeout_seconds() == 30.0


def test_openai_bridge_flag(monkeypatch):
    monkeypatch.delenv("SANITIZER_USE_OPENAI_BRIDGE", raising=False)
    assert config.is_openai_bridge_enabled() is False
    for value in ("true", "1", "yes", "on"):
        monkeypatch.setenv("SANITIZER_USE_OPENAI_BRIDGE", value)
        assert config.is_openai_bridge_enabled() is True
    monkeypatch.setenv("SANITIZER_USE_OPENAI_BRIDGE", "false")
    assert config.is_openai_bridge_enabled() is False


def test_think_output_mode_valid_and_invalid(monkeypatch):
    for value in ("default", "none", "text", "think_tag", "bridge"):
        monkeypatch.setenv("THINK_OUTPUT_MODE", value)
        assert config.get_think_output_mode() == value
    monkeypatch.setenv("THINK_OUTPUT_MODE", "bogus")
    assert config.get_think_output_mode() == "default"
    monkeypatch.delenv("THINK_OUTPUT_MODE", raising=False)
    assert config.get_think_output_mode() == "default"
