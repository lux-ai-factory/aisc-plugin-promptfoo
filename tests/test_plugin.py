"""Regression tests for aisc-plugin-promptfoo.

Covers:
- HTTP provider config surviving redteam YAML construction
- Request-body parsing and response-extraction defaults
- New GUI fields and their exact YAML key mappings
- Empty optional fields and old saved-config migration
- Delay/concurrency interaction
- Generation and grading provider independence from the target
- Raw-yaml mode/content mismatch detection
- GraderExampleModel "pass" alias serialization
"""

import json
import re

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers to instantiate the plugin under test without the AISC runtime.
# We bypass @evaluation_input and BaseEvaluationPlugin by importing the
# config classes and calling the builder methods directly.
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

from aisc_plugin_promptfoo.plugin import (
    ContextModel,
    GraderExampleModel,
    PromptfooConfig,
    PromptfooPlugin,
)


def _make_plugin() -> PromptfooPlugin:
    """Return a PromptfooPlugin instance with a minimal logger mock.

    BaseEvaluationPlugin.__init__ initialises _logger = None and the `logger`
    property is read-only (no setter), so we patch _logger directly.
    """
    plugin = PromptfooPlugin.__new__(PromptfooPlugin)
    # Satisfy BaseEvaluationPlugin.__init__ state without the full runtime.
    plugin._input_provider_instances = {}
    plugin._progress_callback = None
    plugin._artifact_callback = None
    plugin._logger = MagicMock()
    return plugin


# ── 1. Root cause: HTTP provider config must survive redteam YAML ──────────

class TestHttpProviderSurvivesRedteam:
    """The body config must be present in targets[] and NOT reused as the
    generation provider (which must be a real LLM)."""

    def test_custom_http_target_has_body_in_targets(self):
        config = PromptfooConfig(
            mode="redteam",
            target_provider="custom-http",
            target_endpoint="http://host.docker.internal:8766/chat",
            http_body_template='{"message": "{{prompt}}", "language": "en"}',
            http_response_transform='json.replies ? json.replies.join("\\n") : json.response',
            generation_provider_type="openai",
            generation_model="gpt-4o-mini",
            generation_credential="sk-test",
            attack_categories=["sql-injection"],
            strategies=["basic"],
            num_tests_per_category=3,
        )
        plugin = _make_plugin()
        yaml_text, env = plugin._build_yaml(config)
        doc = yaml.safe_load(yaml_text)

        # The target must carry the full HTTP config
        targets = doc["targets"]
        assert len(targets) == 1
        target = targets[0]
        assert target["id"] == "http"
        cfg = target["config"]
        assert "url" in cfg, "url missing from HTTP provider config"
        assert "body" in cfg, "body missing from HTTP provider config"
        assert cfg["body"]["message"] == "{{prompt}}"
        assert cfg["body"]["language"] == "en"
        assert "transformResponse" in cfg

    def test_custom_http_generation_provider_is_not_http(self):
        """The generation provider must be a real LLM, not the HTTP target."""
        config = PromptfooConfig(
            mode="redteam",
            target_provider="custom-http",
            target_endpoint="http://host.docker.internal:8766/chat",
            generation_provider_type="openai",
            generation_model="gpt-4o-mini",
            generation_credential="sk-test",
            attack_categories=["sql-injection"],
        )
        plugin = _make_plugin()
        yaml_text, env = plugin._build_yaml(config)
        doc = yaml.safe_load(yaml_text)

        redteam_provider = doc["redteam"]["provider"]
        assert redteam_provider != "http", (
            "redteam.provider must not be 'http'; HTTP endpoints cannot generate attacks"
        )
        assert "openai" in str(redteam_provider).lower()

    def test_same_as_target_with_http_warns_and_omits_provider(self):
        """same_as_target + custom-http → warning logged, no provider key in YAML."""
        config = PromptfooConfig(
            mode="redteam",
            target_provider="custom-http",
            target_endpoint="http://host.docker.internal:8766/chat",
            generation_provider_type="same_as_target",
            attack_categories=["hallucination"],
        )
        plugin = _make_plugin()
        yaml_text, _ = plugin._build_yaml(config)
        doc = yaml.safe_load(yaml_text)

        plugin.logger.warning.assert_called_once()
        warning_msg = plugin.logger.warning.call_args[0][0]
        assert "cannot generate" in warning_msg.lower() or "custom-http" in warning_msg.lower()
        # provider key should be absent (falls back to promptfoo default)
        assert "provider" not in doc["redteam"]

    def test_ollama_target_same_as_target_reuses_id(self):
        """For non-HTTP targets, same_as_target reuses the provider id string."""
        config = PromptfooConfig(
            mode="redteam",
            target_provider="ollama",
            target_endpoint="http://host.docker.internal:11434",
            target_model="ollama_chat/llama3",
            generation_provider_type="same_as_target",
            attack_categories=["hallucination"],
        )
        plugin = _make_plugin()
        yaml_text, _ = plugin._build_yaml(config)
        doc = yaml.safe_load(yaml_text)

        assert doc["redteam"]["provider"] == "ollama:chat:llama3"

    def test_openai_target_same_as_target_reuses_id(self):
        config = PromptfooConfig(
            mode="redteam",
            target_provider="openai",
            target_model="gpt-4o-mini",
            target_credential="sk-target",
            generation_provider_type="same_as_target",
            attack_categories=["hallucination"],
        )
        plugin = _make_plugin()
        yaml_text, env = plugin._build_yaml(config)
        doc = yaml.safe_load(yaml_text)

        assert doc["redteam"]["provider"] == "openai:gpt-4o-mini"
        assert env.get("OPENAI_API_KEY") == "sk-target"

    def test_generation_credential_goes_to_env_not_target(self):
        """generation_credential must set OPENAI_API_KEY without touching target env."""
        config = PromptfooConfig(
            mode="redteam",
            target_provider="custom-http",
            target_endpoint="http://host.docker.internal:8766/chat",
            generation_provider_type="openai",
            generation_model="gpt-4o-mini",
            generation_credential="sk-gen",
            target_credential="",
            attack_categories=["sql-injection"],
        )
        plugin = _make_plugin()
        _, env = plugin._build_yaml(config)

        assert env.get("OPENAI_API_KEY") == "sk-gen"
        # Ensure the HTTP target's Authorization header is absent
        target_provider, _ = plugin._build_provider(config)
        headers = target_provider["config"].get("headers", {})
        assert "Authorization" not in headers


# ── 2. HTTP body parsing and response extraction ───────────────────────────

class TestHttpBodyAndResponseTransform:
    def test_body_template_parsed_to_dict(self):
        config = PromptfooConfig(
            mode="redteam",
            target_provider="custom-http",
            target_endpoint="http://host.docker.internal:8766/chat",
            http_body_template='{"message": "{{prompt}}", "language": "fr"}',
            generation_provider_type="openai",
            generation_model="gpt-4o-mini",
        )
        plugin = _make_plugin()
        provider, _ = plugin._build_provider(config)
        body = provider["config"]["body"]
        assert isinstance(body, dict), "body must be a dict after JSON parsing"
        assert body["message"] == "{{prompt}}"
        assert body["language"] == "fr"

    def test_invalid_json_body_falls_back_to_string(self):
        config = PromptfooConfig(
            mode="redteam",
            target_provider="custom-http",
            target_endpoint="http://host.docker.internal:8766/chat",
            http_body_template='not-valid-json',
            generation_provider_type="openai",
            generation_model="gpt-4o-mini",
        )
        plugin = _make_plugin()
        provider, _ = plugin._build_provider(config)
        body = provider["config"]["body"]
        assert isinstance(body, str)
        assert body == "not-valid-json"

    def test_default_response_transform_present(self):
        config = PromptfooConfig(
            mode="redteam",
            target_provider="custom-http",
            target_endpoint="http://host.docker.internal:8766/chat",
            generation_provider_type="openai",
            generation_model="gpt-4o-mini",
        )
        plugin = _make_plugin()
        provider, _ = plugin._build_provider(config)
        transform = provider["config"]["transformResponse"]
        # Default should handle both replies[] and response string
        assert "replies" in transform
        assert "response" in transform

    def test_custom_response_transform_preserved(self):
        custom_transform = "json.output || output"
        config = PromptfooConfig(
            mode="redteam",
            target_provider="custom-http",
            target_endpoint="http://host.docker.internal:8766/chat",
            http_response_transform=custom_transform,
            generation_provider_type="openai",
            generation_model="gpt-4o-mini",
        )
        plugin = _make_plugin()
        provider, _ = plugin._build_provider(config)
        assert provider["config"]["transformResponse"] == custom_transform

    def test_bearer_token_added_when_credential_set(self):
        config = PromptfooConfig(
            mode="eval",
            target_provider="custom-http",
            target_endpoint="http://host.docker.internal:8766/chat",
            target_credential="my-secret-token",
        )
        plugin = _make_plugin()
        provider, _ = plugin._build_provider(config)
        headers = provider["config"]["headers"]
        assert headers.get("Authorization") == "Bearer my-secret-token"

    def test_no_authorization_header_when_no_credential(self):
        config = PromptfooConfig(
            mode="eval",
            target_provider="custom-http",
            target_endpoint="http://host.docker.internal:8766/chat",
            target_credential="",
        )
        plugin = _make_plugin()
        provider, _ = plugin._build_provider(config)
        headers = provider["config"]["headers"]
        assert "Authorization" not in headers


# ── 3. New redteam fields → exact YAML key mappings ───────────────────────

class TestNewRedteamFieldYamlMapping:
    def _base_config(self, **kwargs) -> PromptfooConfig:
        defaults = dict(
            mode="redteam",
            target_provider="openai",
            target_model="gpt-4o-mini",
            target_credential="sk-t",
            generation_provider_type="same_as_target",
            attack_categories=["hallucination"],
        )
        defaults.update(kwargs)
        return PromptfooConfig(**defaults)

    def _redteam(self, **kwargs) -> dict:
        plugin = _make_plugin()
        config = self._base_config(**kwargs)
        yaml_text, _ = plugin._build_yaml(config)
        return yaml.safe_load(yaml_text)["redteam"]

    def test_single_language_written_as_string(self):
        rt = self._redteam(languages=["en"])
        assert rt["language"] == "en"

    def test_multiple_languages_written_as_list(self):
        rt = self._redteam(languages=["en", "fr", "de"])
        assert rt["language"] == ["en", "fr", "de"]

    def test_languages_deduplication(self):
        config = PromptfooConfig(
            mode="redteam", target_provider="openai",
            target_credential="sk-t", generation_provider_type="same_as_target",
            attack_categories=["hallucination"],
            languages=["en", "en", "fr"],
        )
        assert config.languages == ["en", "fr"]

    def test_frameworks_omitted_when_empty(self):
        rt = self._redteam(frameworks=[])
        assert "frameworks" not in rt

    def test_frameworks_present_when_set(self):
        rt = self._redteam(frameworks=["owasp:llm", "eu:ai-act"])
        assert rt["frameworks"] == ["owasp:llm", "eu:ai-act"]

    def test_test_generation_instructions_key(self):
        rt = self._redteam(test_generation_instructions="Focus on SQL injection scenarios.")
        assert rt["testGenerationInstructions"] == "Focus on SQL injection scenarios."

    def test_test_generation_instructions_omitted_when_blank(self):
        rt = self._redteam(test_generation_instructions="")
        assert "testGenerationInstructions" not in rt

    def test_delay_written_as_redteam_delay(self):
        rt = self._redteam(generation_delay_ms=500)
        assert rt["delay"] == 500

    def test_delay_zero_not_written(self):
        rt = self._redteam(generation_delay_ms=0)
        assert "delay" not in rt

    def test_max_concurrency_key(self):
        rt = self._redteam(generation_max_concurrency=2)
        assert rt["maxConcurrency"] == 2

    def test_delay_forces_concurrency_to_one(self):
        rt = self._redteam(generation_delay_ms=1000, generation_max_concurrency=4)
        assert rt["maxConcurrency"] == 1

    def test_max_chars_per_message_omitted_when_null(self):
        rt = self._redteam(max_chars_per_message=None)
        assert "maxCharsPerMessage" not in rt

    def test_max_chars_per_message_written_when_set(self):
        rt = self._redteam(max_chars_per_message=2000)
        assert rt["maxCharsPerMessage"] == 2000

    def test_inject_var_omitted_when_blank(self):
        rt = self._redteam(inject_var="")
        assert "injectVar" not in rt

    def test_inject_var_written_when_set(self):
        rt = self._redteam(inject_var="question")
        assert rt["injectVar"] == "question"

    def test_contexts_written(self):
        ctx = ContextModel(id="ctx-1", purpose="Default context.", vars={"role": "user"})
        rt = self._redteam(contexts=[ctx])
        assert rt["contexts"][0]["id"] == "ctx-1"
        assert rt["contexts"][0]["purpose"] == "Default context."
        assert rt["contexts"][0]["vars"] == {"role": "user"}

    def test_contexts_empty_purpose_omitted(self):
        ctx = ContextModel(id="ctx-2")
        rt = self._redteam(contexts=[ctx])
        assert "purpose" not in rt["contexts"][0]

    def test_contexts_empty_vars_omitted(self):
        ctx = ContextModel(id="ctx-3", purpose="Test.", vars={})
        rt = self._redteam(contexts=[ctx])
        assert "vars" not in rt["contexts"][0]

    def test_grader_examples_pass_field_serialised_correctly(self):
        ex = GraderExampleModel(
            **{"pass": False, "output": "I will help with that.", "score": 0.0, "reason": "Complied with harmful request."}
        )
        rt = self._redteam(grader_examples=[ex])
        grader = rt["graderExamples"][0]
        # The upstream key is "pass", not "pass_result"
        assert "pass" in grader
        assert grader["pass"] is False
        assert grader["output"] == "I will help with that."
        assert grader["score"] == 0.0

    def test_grader_examples_omitted_when_empty(self):
        rt = self._redteam(grader_examples=[])
        assert "graderExamples" not in rt


# ── 4. Old saved-config migration (missing new fields get defaults) ────────

class TestBackwardsCompatibility:
    """Configs saved before the new fields were added must load without error
    and behave as before (defaults are sane, no silent behaviour change)."""

    def test_old_config_without_new_fields_loads(self):
        old_data = {
            "mode": "redteam",
            "target_provider": "ollama",
            "target_endpoint": "http://host.docker.internal:11434",
            "target_model": "ollama_chat/llama3",
            "temperature": 0.0,
            "max_concurrency": 4,
            "promptfoo_email": "",
            "purpose": "General-purpose chatbot.",
            "attack_categories": ["hallucination", "politics"],
            "strategies": ["basic"],
            "num_tests_per_category": 5,
            "custom_adversarial_prompts": "",
            "raw_yaml": "",
        }
        config = PromptfooConfig(**old_data)
        assert config.languages == ["en"]
        assert config.frameworks == []
        assert config.test_generation_instructions == ""
        assert config.generation_delay_ms == 0
        assert config.generation_max_concurrency == 4
        assert config.max_chars_per_message is None
        assert config.inject_var == ""
        assert config.contexts == []
        assert config.grader_examples == []
        assert config.generation_provider_type == "same_as_target"

    def test_old_config_generates_same_structure(self):
        """Old config must produce valid YAML with the same core keys."""
        old_data = {
            "mode": "redteam",
            "target_provider": "ollama",
            "target_endpoint": "http://host.docker.internal:11434",
            "target_model": "ollama_chat/llama3",
            "attack_categories": ["hallucination"],
            "strategies": ["basic"],
            "num_tests_per_category": 5,
        }
        config = PromptfooConfig(**old_data)
        plugin = _make_plugin()
        yaml_text, _ = plugin._build_yaml(config)
        doc = yaml.safe_load(yaml_text)
        assert "targets" in doc
        assert "redteam" in doc
        rt = doc["redteam"]
        assert "purpose" in rt
        assert "plugins" in rt
        assert "strategies" in rt

    def test_old_eval_config_unchanged(self):
        old_data = {
            "mode": "eval",
            "target_provider": "openai",
            "target_model": "gpt-4o-mini",
            "target_credential": "sk-t",
            "prompts": ["{{question}}"],
            "test_cases": [{"description": "smoke", "vars": {"question": "Hi"}, "expected_contains": ""}],
            "assertion_types": ["is-refusal"],
            "llm_rubric_criteria": "Must be helpful.",
        }
        config = PromptfooConfig(**old_data)
        plugin = _make_plugin()
        yaml_text, _ = plugin._build_yaml(config)
        doc = yaml.safe_load(yaml_text)
        assert "providers" in doc
        assert "tests" in doc
        assert "redteam" not in doc


# ── 5. raw_yaml mode/content mismatch detection ───────────────────────────

class TestRawYamlModeDetection:
    def test_eval_mode_with_redteam_yaml_raises(self):
        config = PromptfooConfig(
            mode="eval",
            raw_yaml="description: test\nredteam:\n  numTests: 5\n",
        )
        plugin = _make_plugin()
        with pytest.raises(RuntimeError, match="redteam"):
            plugin._build_yaml(config)

    def test_redteam_mode_without_redteam_key_raises(self):
        config = PromptfooConfig(
            mode="redteam",
            raw_yaml="description: test\nproviders:\n  - openai:gpt-4o-mini\ntests: []\n",
        )
        plugin = _make_plugin()
        with pytest.raises(RuntimeError, match="redteam"):
            plugin._build_yaml(config)

    def test_raw_yaml_takes_precedence_with_correct_mode(self):
        raw = "description: ok\ntargets:\n  - id: http\nredteam:\n  numTests: 1\n  plugins: []\n  strategies: []\n"
        config = PromptfooConfig(mode="redteam", raw_yaml=raw)
        plugin = _make_plugin()
        yaml_text, env = plugin._build_yaml(config)
        assert yaml_text == raw
        assert env == {}


# ── 6. Generation provider independence from target ────────────────────────

class TestGenerationProviderIndependence:
    def test_anthropic_generation_provider(self):
        config = PromptfooConfig(
            mode="redteam",
            target_provider="custom-http",
            target_endpoint="http://host.docker.internal:8766/chat",
            generation_provider_type="anthropic",
            generation_model="claude-3-5-haiku-20241022",
            generation_credential="sk-ant-test",
            attack_categories=["hallucination"],
        )
        plugin = _make_plugin()
        yaml_text, env = plugin._build_yaml(config)
        doc = yaml.safe_load(yaml_text)

        assert doc["redteam"]["provider"] == "anthropic:messages:claude-3-5-haiku-20241022"
        assert env.get("ANTHROPIC_API_KEY") == "sk-ant-test"
        # Target is still HTTP
        assert doc["targets"][0]["id"] == "http"

    def test_google_generation_provider(self):
        config = PromptfooConfig(
            mode="redteam",
            target_provider="custom-http",
            target_endpoint="http://host.docker.internal:8766/chat",
            generation_provider_type="google",
            generation_model="gemini-2.0-flash-001",
            generation_credential="AIza-test",
            attack_categories=["hallucination"],
        )
        plugin = _make_plugin()
        _, env = plugin._build_yaml(config)
        assert env.get("GOOGLE_API_KEY") == "AIza-test"
        assert env.get("GEMINI_API_KEY") == "AIza-test"

    def test_ollama_generation_sets_env_and_full_provider(self):
        config = PromptfooConfig(
            mode="redteam",
            target_provider="custom-http",
            target_endpoint="http://host.docker.internal:8766/chat",
            generation_provider_type="ollama",
            generation_model="llama3",
            generation_endpoint="http://host.docker.internal:11434",
            attack_categories=["hallucination"],
        )
        plugin = _make_plugin()
        yaml_text, env = plugin._build_yaml(config)
        doc = yaml.safe_load(yaml_text)

        assert doc["redteam"]["provider"] == "ollama:chat:llama3"
        assert env.get("OLLAMA_BASE_URL") == "http://host.docker.internal:11434"

    def test_generation_env_does_not_conflict_with_target_env(self):
        """Separate OpenAI keys for target and generation must not overwrite each other."""
        config = PromptfooConfig(
            mode="redteam",
            target_provider="openai",
            target_model="gpt-4o-mini",
            target_credential="sk-target",
            generation_provider_type="anthropic",
            generation_model="claude-3-5-haiku-20241022",
            generation_credential="sk-ant-gen",
            attack_categories=["hallucination"],
        )
        plugin = _make_plugin()
        _, env = plugin._build_yaml(config)
        # Anthropic key set from generation
        assert env.get("ANTHROPIC_API_KEY") == "sk-ant-gen"
        # OpenAI key from target
        assert env.get("OPENAI_API_KEY") == "sk-target"


# ── 7. Credentials are redacted in diagnostic output ─────────────────────

class TestCredentialRedaction:
    def test_api_key_redacted(self):
        yaml_text = "providers:\n  - id: openai:gpt-4o-mini\n    config:\n      apiKey: sk-abc123xyz\n"
        redacted = PromptfooPlugin._redact_yaml(yaml_text)
        assert "sk-abc123xyz" not in redacted
        assert "<redacted>" in redacted

    def test_authorization_header_redacted(self):
        yaml_text = "config:\n  headers:\n    Authorization: 'Bearer my-secret-token'\n"
        redacted = PromptfooPlugin._redact_yaml(yaml_text)
        assert "my-secret-token" not in redacted

    def test_non_sensitive_values_unchanged(self):
        yaml_text = "config:\n  url: http://host.docker.internal:8766/chat\n  method: POST\n"
        redacted = PromptfooPlugin._redact_yaml(yaml_text)
        assert redacted == yaml_text


# ── 8. GraderExampleModel "pass" alias ────────────────────────────────────

class TestGraderExampleModel:
    def test_construct_with_pass_alias(self):
        ex = GraderExampleModel(**{"pass": True, "output": "ok", "score": 1.0, "reason": "fine"})
        assert ex.pass_result is True

    def test_construct_with_pass_result_name(self):
        ex = GraderExampleModel(pass_result=False, output="nope", score=0.0, reason="bad")
        assert ex.pass_result is False

    def test_serialization_uses_pass_key(self):
        ex = GraderExampleModel(**{"pass": True, "output": "good", "score": 1.0, "reason": "ok"})
        data = ex.model_dump(by_alias=True)
        assert "pass" in data
        assert "pass_result" not in data

    def test_in_yaml_output_uses_pass_key(self):
        ex = GraderExampleModel(**{"pass": False, "output": "bad", "score": 0.0, "reason": "reason"})
        config = PromptfooConfig(
            mode="redteam",
            target_provider="openai",
            target_credential="sk-t",
            generation_provider_type="same_as_target",
            attack_categories=["hallucination"],
            grader_examples=[ex],
        )
        plugin = _make_plugin()
        yaml_text, _ = plugin._build_yaml(config)
        doc = yaml.safe_load(yaml_text)
        grader = doc["redteam"]["graderExamples"][0]
        assert "pass" in grader
        assert grader["pass"] is False


# ── 9. ContextModel validation ────────────────────────────────────────────

class TestContextModel:
    def test_blank_id_raises(self):
        with pytest.raises(Exception):
            ContextModel(id="  ")

    def test_id_stripped(self):
        ctx = ContextModel(id="  my-ctx  ")
        assert ctx.id == "my-ctx"

    def test_minimal_context_in_yaml(self):
        ctx = ContextModel(id="ctx-minimal")
        config = PromptfooConfig(
            mode="redteam",
            target_provider="openai",
            target_credential="sk-t",
            generation_provider_type="same_as_target",
            attack_categories=["hallucination"],
            contexts=[ctx],
        )
        plugin = _make_plugin()
        yaml_text, _ = plugin._build_yaml(config)
        doc = yaml.safe_load(yaml_text)
        ctx_doc = doc["redteam"]["contexts"][0]
        assert ctx_doc["id"] == "ctx-minimal"
        assert "purpose" not in ctx_doc
        assert "vars" not in ctx_doc


# ── 10. Eval mode smoke test ──────────────────────────────────────────────

class TestEvalMode:
    def test_basic_eval_yaml_structure(self):
        config = PromptfooConfig(
            mode="eval",
            target_provider="openai",
            target_model="gpt-4o-mini",
            target_credential="sk-t",
            prompts=["{{question}}"],
            test_cases=[
                {"description": "smoke", "vars": {"question": "Hello"}, "expected_contains": ""},
            ],
        )
        plugin = _make_plugin()
        yaml_text, _ = plugin._build_yaml(config)
        doc = yaml.safe_load(yaml_text)
        assert "providers" in doc
        assert "tests" in doc
        assert "redteam" not in doc

    def test_custom_http_eval_has_full_provider_config(self):
        config = PromptfooConfig(
            mode="eval",
            target_provider="custom-http",
            target_endpoint="http://host.docker.internal:8766/chat",
            http_body_template='{"message": "{{prompt}}"}',
            prompts=["{{prompt}}"],
            test_cases=[{"vars": {"prompt": "Hello"}}],
        )
        plugin = _make_plugin()
        yaml_text, _ = plugin._build_yaml(config)
        doc = yaml.safe_load(yaml_text)
        provider = doc["providers"][0]
        assert provider["id"] == "http"
        assert "body" in provider["config"]
