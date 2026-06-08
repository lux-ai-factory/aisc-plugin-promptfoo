import json
import os
import shutil
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from aisc_plugin_interface import (
    BaseEvaluationPlugin,
    ChartType,
    InputType,
    Measure,
    MetricVisualization,
    PluginFeatureFlags,
    TaskProgress,
    evaluation_input,
    metric,
)

from .input_providers import RawBytesProvider


Mode = Literal["eval", "redteam"]

# Curated target-model dropdown (real ids; non-OpenAI prefixed so the string is
# self-describing). Promptfoo's CLI wants a provider:model id built from
# target_provider + a bare model name, so _build_provider() strips these
# prefixes before composing the promptfoo provider id.
MODELS = Literal[
    "gpt-4o-mini",
    "gpt-4o",
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-5-haiku-20241022",
    "mistral/mistral-small-latest",
    "mistral/mistral-large-latest",
    "gemini/gemini-2.0-flash-001",
    "ollama_chat/qwen2.5:7b",
    "ollama_chat/llama3",
    "ollama_chat/mistral",
]


def _bare_model(model: str) -> str:
    """Strip a known LiteLLM provider prefix so promptfoo gets a bare model name.

    Promptfoo composes its own provider id (e.g. ``ollama:chat:<m>``,
    ``anthropic:messages:<m>``) from target_provider, so it needs the model
    WITHOUT the ``anthropic/``/``mistral/``/``gemini/``/``ollama_chat/`` prefix
    carried by the curated dropdown values.
    """
    for pfx in ("anthropic/", "mistral/", "gemini/", "openai/",
                "ollama_chat/", "ollama/"):
        if model.startswith(pfx):
            return model[len(pfx):]
    return model

TARGET_PROVIDER = Literal[
    "ollama",
    "openai",
    "anthropic",
    "google",
    "custom-http",
]

ASSERTION_TYPE = Literal[
    "llm-rubric",
    "contains",
    "not-contains",
    "icontains",
    "regex",
    "is-refusal",
    "is-json",
    "similar",
]

ATTACK_CATEGORY = Literal[
    # Security
    "prompt-extraction",
    "indirect-prompt-injection",
    "sql-injection",
    "ssrf",
    "shell-injection",
    "bola",
    "bfla",
    "rbac",
    # Privacy
    "pii:direct",
    "pii:session",
    "pii:social",
    "cross-session-leak",
    # Harm
    "harmful:violent-crime",
    "harmful:hate",
    "harmful:sexual-content",
    "harmful:self-harm",
    "harmful:indiscriminate-weapons",
    "harmful:cybercrime",
    "harmful:misinformation-disinformation",
    # Bias (promptfoo BIAS_PLUGINS = age/disability/gender/race; no `religion` plugin exists)
    "bias:gender",
    "bias:race",
    "bias:age",
    "bias:disability",
    # Safety
    "excessive-agency",
    "hallucination",
    "overreliance",
    "politics",
    # Pre-curated datasets
    "harmbench",
    "donotanswer",
    "xstest",
    "beavertails",
    "toxic-chat",
]

STRATEGY = Literal[
    "basic",
    "base64",
    "leetspeak",
    "multilingual",
    # jailbreak maps to jailbreak:meta internally and requires remote generation
    # (promptfoo cloud or PROMPTFOO_REMOTE_GENERATION_URL). Excluded from default.
    "jailbreak",
]


class TestCase(BaseModel):
    description: str = Field(default="")
    vars: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Values for the prompt template's placeholders, for example "
            "{question: 'What is 2+2?'} fills in {{question}}."
        ),
    )
    expected_contains: str = Field(
        default="",
        description="Optional. The response must contain this substring for the test to pass.",
    )


class PromptfooConfig(BaseModel):
    mode: Mode = Field(
        default="redteam",
        description=(
            "What kind of run this is. `eval` runs your own functional tests against the "
            "model (you supply the prompts and assertions). `redteam` auto-generates "
            "adversarial prompts and scans the model for safety and security issues."
        ),
    )

    # ─── Target system ─────────────────────────────────────────────────────
    target_provider: TARGET_PROVIDER = Field(
        default="ollama",
        title="Target provider",
        description=(
            "Which provider family the model under test belongs to. Promptfoo uses this to "
            "build its provider:model id (for example ollama:chat:..., anthropic:messages:..., "
            "openai:...). Pick the family that matches the target model below. `custom-http` "
            "lets you point at any HTTP chat endpoint."
        ),
    )
    target_model: MODELS = Field(
        default="ollama_chat/llama3",
        title="Target model (under test)",
        description=(
            "The model being tested. Pick one whose family matches the provider above "
            "(gpt-* with openai, anthropic/* with anthropic, gemini/* with google, "
            "ollama_chat/* with ollama). The provider prefix is just a hint for you and is "
            "stripped before the model name is handed to promptfoo. Note: the mistral/* and "
            "gemini/* entries only work if the matching provider exists, and a few of the "
            "hosted ids may be dated, so prefer a current model id for your provider."
        ),
    )
    target_endpoint: str = Field(
        default="http://host.docker.internal:11434",
        description=(
            "Endpoint URL for the model. Needed for `ollama` (for example "
            "http://host.docker.internal:11434) and for `custom-http` (your full chat "
            "endpoint). Hosted providers use their own API endpoint, so this is ignored there."
        ),
    )
    target_credential: str = Field(
        default="",
        description=(
            "API key for the target. Used by openai, anthropic, and google (it maps to "
            "OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY). Leave it blank for "
            "`ollama` and `custom-http`. Secret, never shown in reports."
        ),
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_concurrency: int = Field(default=4, ge=1, le=32)

    # ─── Promptfoo account (redteam only) ─────────────────────────────────
    promptfoo_email: str = Field(
        default="",
        description=(
            "Optional, redteam only. The promptfoo CLI asks for an email before running a "
            "redteam. Put yours here to clear that prompt without any interaction. Leave it "
            "blank and a placeholder (vera@localhost) is used. Not needed for eval mode."
        ),
    )

    # ─── Eval mode ────────────────────────────────────────────────────────
    prompts: list[str] = Field(
        default_factory=lambda: ["{{question}}"],
        description=(
            "Prompt templates sent to the model. Use `{{var}}` placeholders that match your "
            "test variables (or CSV column headers), for example `{{question}}`."
        ),
    )
    test_cases: list[TestCase] = Field(
        default_factory=lambda: [
            TestCase(
                description="Self-introduction smoke test",
                vars={"question": "Briefly introduce yourself in one sentence."},
                expected_contains="",
            ),
        ],
        description=(
            "Inline test cases for eval mode. Each one sets the prompt variables plus an "
            "optional expected substring. Eval mode needs at least one source of tests, so "
            "either keep a test case here, or upload an eval-tests CSV instead (the CSV wins "
            "if both are present)."
        ),
    )
    assertion_types: list[ASSERTION_TYPE] = Field(
        default_factory=lambda: ["llm-rubric", "is-refusal"],
        description=(
            "Assertions applied to every eval test, on top of each test's own "
            "expected_contains. Only llm-rubric, is-refusal, and is-json are applied at this "
            "shared level; contains, not-contains, icontains, regex, and similar are valid "
            "promptfoo types but only make sense per-test, so they aren't written here."
        ),
    )
    llm_rubric_criteria: str = Field(
        default="Response is accurate, helpful, and non-evasive.",
        description=(
            "The rubric an LLM judge grades against when llm-rubric is in your assertions. "
            "Describe what a good answer looks like."
        ),
    )

    # ─── Redteam mode ─────────────────────────────────────────────────────
    purpose: str = Field(
        default="General-purpose chatbot.",
        description=(
            "A short description of what the model under test is meant to do. Promptfoo uses "
            "this to make the generated attacks more relevant, so be specific (for example "
            "'customer support bot for a bank')."
        ),
    )
    attack_categories: list[ATTACK_CATEGORY] = Field(
        default_factory=lambda: [
            "hallucination",
            "politics",
            "excessive-agency",
            "donotanswer",
            "harmbench",
        ],
        description=(
            "The adversarial categories to probe. Promptfoo generates the attack prompts for "
            "each one. Heads up: many categories (most harmful:*, pii:*, bias:*, and the "
            "security ones) rely on promptfoo's remote generation service, so they need cloud "
            "access. Categories that run fully locally include hallucination, politics, "
            "excessive-agency, overreliance, donotanswer, harmbench, beavertails, xstest, and "
            "toxic-chat. Note beavertails is gated behind Hugging Face access."
        ),
    )
    strategies: list[STRATEGY] = Field(
        default_factory=lambda: ["basic"],
        description=(
            "How the attack prompts are delivered. basic, base64, and leetspeak run locally. "
            "multilingual still works but is deprecated upstream. jailbreak needs promptfoo "
            "cloud or a self-hosted PROMPTFOO_REMOTE_GENERATION_URL, so it won't run fully offline."
        ),
    )
    num_tests_per_category: int = Field(
        default=5,
        ge=1,
        le=50,
        description="How many generated tests to run per attack category.",
    )
    custom_adversarial_prompts: str = Field(
        default="",
        description=(
            "Optional, redteam only. One seed prompt per line. Each seed is fed to promptfoo's "
            "intent plugin, which turns it into a test case alongside the built-in attacks. "
            "Use this to steer the redteam toward your own scenarios."
        ),
    )

    # ─── Escape hatch ─────────────────────────────────────────────────────
    raw_yaml: str = Field(
        default="",
        description=(
            "Power-user escape hatch: paste a complete promptfoo YAML and it's used verbatim. "
            "This overrides everything else, including an uploaded eval-tests CSV and all the "
            "fields above."
        ),
    )


@evaluation_input(
    name="eval-tests-file",
    label="Eval tests CSV (eval mode only)",
    input_provider_class=RawBytesProvider,
    input_type=InputType.DATASET,
    required=False,
)
class PromptfooPlugin(BaseEvaluationPlugin[PromptfooConfig]):
    plugin_name = "Promptfoo"
    ui_icon = "policy"

    form_ui_schema = {
        "target_model": {
            "ui:help": (
                "Pick a model whose family matches the provider. ollama_chat/* models are "
                "local, so set the endpoint above. The provider prefix is just a hint and is "
                "stripped automatically."
            ),
        },
        "raw_yaml": {"ui:widget": "textarea", "ui:options": {"rows": 16}},
        "custom_adversarial_prompts": {"ui:widget": "textarea", "ui:options": {"rows": 6}},
        "prompts": {"ui:options": {"orderable": True}},
        "test_cases": {
            "ui:options": {"orderable": True},
            "ui:help": (
                "Eval mode only. Skipped if you upload an eval-tests CSV (the CSV takes "
                "precedence)."
            ),
        },
        "target_credential": {
            "ui:widget": "password",
            "ui:help": "API key for openai/anthropic/google. Leave blank for ollama/custom-http.",
        },
    }

    @property
    def feature_flags(self) -> PluginFeatureFlags:
        return PluginFeatureFlags(can_parse_config_from_dataset=False)

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        # One table of the aggregate KPIs (one row per KPI). The per-test detail
        # is the "Per-run results" CSV, so we don't render long-format per-test rows.
        return [
            MetricVisualization(
                chart_type=ChartType.TABLE,
                metrics=["pass_rate", "fail_rate", "refusal_rate",
                         "mean_latency_ms", "total_cost", "n_tests"],
            )
        ]

    # Fields used only when mode=eval
    EVAL_ONLY_FIELDS = (
        "prompts",
        "test_cases",
        "assertion_types",
        "llm_rubric_criteria",
    )
    # Fields used only when mode=redteam
    REDTEAM_ONLY_FIELDS = (
        "purpose",
        "attack_categories",
        "strategies",
        "num_tests_per_category",
        "custom_adversarial_prompts",
        "promptfoo_email",
    )

    @staticmethod
    def _drop_field(form_data, field: str) -> None:
        """Delete a hidden field's value from form_data so stale input isn't
        persisted once the field is no longer shown. Works for both a plain
        dict and a pydantic-style object."""
        if isinstance(form_data, dict):
            form_data.pop(field, None)
        elif form_data is not None and hasattr(form_data, field):
            try:
                delattr(form_data, field)
            except (AttributeError, ValueError):
                pass

    def on_config_change(self, form_data):
        schema, ui_schema = self.get_full_schema()

        selected_mode = "redteam"
        if isinstance(form_data, dict):
            selected_mode = form_data.get("mode") or "redteam"
        elif form_data is not None:
            selected_mode = getattr(form_data, "mode", "redteam")

        props = schema.get("properties", {})
        required = schema.get("required", [])
        hide = self.EVAL_ONLY_FIELDS if selected_mode == "redteam" else self.REDTEAM_ONLY_FIELDS

        for field in hide:
            props.pop(field, None)
            if field in required:
                required.remove(field)
            # Also hide via ui:widget if for any reason the field reappears.
            ui_schema.setdefault(field, {})["ui:widget"] = "hidden"
            # Drop the stale value so a hidden field can't carry old input forward.
            self._drop_field(form_data, field)

        # Handle provider-specific fields
        selected_provider = "ollama"
        if isinstance(form_data, dict):
            selected_provider = form_data.get("target_provider") or "ollama"
        elif form_data is not None:
            selected_provider = getattr(form_data, "target_provider", "ollama")

        if selected_provider in ("ollama", "custom-http"):
            # Hide the credential (these targets don't use an API key).
            if "target_credential" in required:
                required.remove("target_credential")
            ui_schema.setdefault("target_credential", {})["ui:widget"] = "hidden"
            self._drop_field(form_data, "target_credential")
        else:
            # Hide the endpoint (hosted providers have a fixed API endpoint).
            if "target_endpoint" in required:
                required.remove("target_endpoint")
            ui_schema.setdefault("target_endpoint", {})["ui:widget"] = "hidden"
            self._drop_field(form_data, "target_endpoint")

        return form_data, schema, ui_schema

    # ── Provider building ────────────────────────────────────────────────

    def _build_provider(self, config: PromptfooConfig) -> tuple[dict[str, Any], dict[str, str]]:
        """Return (provider_dict, extra_env_vars) for the chosen target."""
        env: dict[str, str] = {}
        provider_id: str
        provider_cfg: dict[str, Any] = {"temperature": config.temperature}

        # Curated dropdown values are provider-prefixed; promptfoo wants a bare
        # model name in its own provider:model id, so strip the prefix here.
        model = _bare_model(config.target_model)

        if config.target_provider == "ollama":
            provider_id = f"ollama:chat:{model or 'llama3:latest'}"
            host = (config.target_endpoint or "http://host.docker.internal:11434").rstrip("/")
            # Strip trailing /v1 (promptfoo's ollama provider points at the base API).
            if host.endswith("/v1"):
                host = host[:-3].rstrip("/")
            env["OLLAMA_BASE_URL"] = host
        elif config.target_provider == "openai":
            provider_id = f"openai:{model or 'gpt-4o-mini'}"
            if config.target_credential:
                env["OPENAI_API_KEY"] = config.target_credential
            if config.target_endpoint:
                env["OPENAI_BASE_URL"] = config.target_endpoint.rstrip("/")
        elif config.target_provider == "anthropic":
            provider_id = f"anthropic:messages:{model or 'claude-3-5-sonnet-20241022'}"
            if config.target_credential:
                env["ANTHROPIC_API_KEY"] = config.target_credential
        elif config.target_provider == "google":
            provider_id = f"google:{model or 'gemini-1.5-flash'}"
            if config.target_credential:
                env["GOOGLE_API_KEY"] = config.target_credential
                env.setdefault("GEMINI_API_KEY", config.target_credential)
        elif config.target_provider == "custom-http":
            provider_id = "http"
            provider_cfg = {
                "url": config.target_endpoint,
                "method": "POST",
                "headers": (
                    {"Authorization": f"Bearer {config.target_credential}"}
                    if config.target_credential
                    else {}
                ),
                "body": {"prompt": "{{prompt}}"},
                "transformResponse": "json.response || json.output || output",
            }
        else:
            raise ValueError(f"Unknown target_provider: {config.target_provider}")

        return {"id": provider_id, "config": provider_cfg}, env

    # ── YAML composition ────────────────────────────────────────────────

    def _build_eval_yaml(
        self,
        config: PromptfooConfig,
        provider: dict[str, Any],
        tests_csv_name: str | None = None,
    ) -> dict[str, Any]:
        default_assertions: list[dict[str, Any]] = []
        for atype in config.assertion_types:
            if atype == "llm-rubric":
                default_assertions.append({"type": "llm-rubric", "value": config.llm_rubric_criteria})
            elif atype == "is-refusal":
                default_assertions.append({"type": "is-refusal"})
            elif atype == "is-json":
                default_assertions.append({"type": "is-json"})
            # text-substring/regex assertions only make sense per test; skip at default level

        doc: dict[str, Any] = {
            "description": "AISC Promptfoo eval",
            "providers": [provider],
            "prompts": config.prompts or ["{{prompt}}"],
        }
        if default_assertions:
            doc["defaultTest"] = {"assert": default_assertions}

        # Precedence (layered): raw_yaml > uploaded eval-tests CSV > inline test_cases.
        # raw_yaml is handled upstream in _build_yaml (it never reaches here). If a CSV
        # was uploaded, reference it and skip the inline test_cases loop entirely —
        # promptfoo parses the CSV natively (rows = tests, headers = {{vars}},
        # __expected*/__metadata:* etc.).
        if tests_csv_name:
            doc["tests"] = tests_csv_name
            return doc

        tests_doc: list[dict[str, Any]] = []
        for t in config.test_cases:
            entry: dict[str, Any] = {
                "description": t.description or "",
                "vars": t.vars,
            }
            per_asserts: list[dict[str, Any]] = []
            if t.expected_contains:
                per_asserts.append({"type": "contains", "value": t.expected_contains})
            if per_asserts:
                entry["assert"] = per_asserts
            tests_doc.append(entry)

        if tests_doc:
            doc["tests"] = tests_doc
        return doc

    def _build_redteam_yaml(self, config: PromptfooConfig, provider: dict[str, Any]) -> dict[str, Any]:
        plugin_list: list[Any] = list(config.attack_categories)
        custom_intents = [
            line.strip()
            for line in (config.custom_adversarial_prompts or "").splitlines()
            if line.strip()
        ]
        if custom_intents:
            plugin_list.append({"id": "intent", "config": {"intent": custom_intents}})

        return {
            "description": "AISC Promptfoo redteam",
            "targets": [provider],
            "redteam": {
                "purpose": config.purpose,
                "numTests": config.num_tests_per_category,
                # Force the attack generator to use the user's target provider
                # so everything stays local (avoids promptfoo cloud + email
                # verification gate).
                "provider": provider["id"],
                "plugins": plugin_list,
                "strategies": list(config.strategies),
            },
        }

    def _build_yaml(
        self, config: PromptfooConfig, tests_csv_name: str | None = None
    ) -> tuple[str, dict[str, str]]:
        if config.raw_yaml.strip():
            return config.raw_yaml, {}

        provider, env = self._build_provider(config)
        if config.mode == "redteam":
            doc = self._build_redteam_yaml(config, provider)
        else:
            doc = self._build_eval_yaml(config, provider, tests_csv_name=tests_csv_name)
        return yaml.safe_dump(doc, sort_keys=False), env

    # ── CLI invocation ──────────────────────────────────────────────────

    @staticmethod
    def _bundled_node_bin() -> Path | None:
        """Directory of the Node runtime bundled with this plugin via
        nodejs-wheel-binaries (node/npm/npx live here). Returns None if the
        wheel is unavailable. Keeps promptfoo self-contained — no Node on the
        host/image required."""
        try:
            from nodejs_wheel.executable import ROOT_DIR
        except Exception:
            return None
        bin_dir = Path(ROOT_DIR) / "bin"
        return bin_dir if bin_dir.is_dir() else None

    def _resolve_promptfoo(self) -> list[str]:
        # Prefer anything already on PATH (lets a host/image override win).
        binary = shutil.which("promptfoo")
        if binary:
            return [binary]
        npx = shutil.which("npx")
        if npx:
            return [npx, "-y", "promptfoo@latest"]
        # Fall back to the Node bundled with the plugin (self-contained).
        # Invoke npm's npx-cli.js with the bundled node directly: the bin/npx
        # shim resolves its modules relative to its own path and breaks when
        # called by absolute path, whereas `node .../npm/bin/npx-cli.js` works.
        bundled = self._bundled_node_bin()
        if bundled:
            node = bundled / "node"
            npx_cli = bundled.parent / "lib" / "node_modules" / "npm" / "bin" / "npx-cli.js"
            if node.exists() and npx_cli.exists():
                return [str(node), str(npx_cli), "-y", "promptfoo@latest"]
        raise RuntimeError(
            "No Node runtime available: `promptfoo`/`npx` not on PATH and the "
            "bundled nodejs-wheel-binaries Node was not found. Reinstall the "
            "plugin (it depends on nodejs-wheel-binaries)."
        )

    # ── Result aggregation ─────────────────────────────────────────────

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return float(int(value)) if isinstance(value, bool) else None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _aggregate(self, results_payload: dict) -> dict[str, Any]:
        results_array = results_payload.get("results", [])
        if isinstance(results_array, dict):
            results_array = results_array.get("results", [])

        scores: list[float] = []
        successes: list[float] = []
        latencies: list[float] = []
        costs: list[float] = []
        refusal_count = 0
        per_test: list[dict[str, Any]] = []
        by_category: dict[str, dict[str, int]] = {}

        for r in results_array:
            score = self._to_float(r.get("score"))
            success = r.get("success")
            success_f = 1.0 if success is True else 0.0 if success is False else None
            latency = self._to_float(r.get("latencyMs"))
            cost = self._to_float(r.get("cost"))

            if score is not None:
                scores.append(score)
            if success_f is not None:
                successes.append(success_f)
            if latency is not None:
                latencies.append(latency)
            if cost is not None:
                costs.append(cost)

            grading = r.get("gradingResult") or {}
            grading_reason = (grading.get("reason") or "")
            if "refus" in grading_reason.lower():
                refusal_count += 1

            metadata = r.get("metadata") or {}
            category = metadata.get("pluginId") or metadata.get("plugin") or r.get("vars", {}).get("plugin")
            strategy = metadata.get("strategyId") or metadata.get("strategy") or r.get("vars", {}).get("strategy")
            if category:
                bucket = by_category.setdefault(category, {"tested": 0, "failed": 0})
                bucket["tested"] += 1
                if success is False:
                    bucket["failed"] += 1

            per_test.append({
                "category": category,
                "strategy": strategy,
                "prompt": (r.get("prompt") or {}).get("raw") if isinstance(r.get("prompt"), dict) else r.get("prompt"),
                "vars": r.get("vars") or {},
                "output": (r.get("response") or {}).get("output"),
                "score": score,
                "success": success,
                "latencyMs": latency,
                "cost": cost,
                "reason": grading_reason[:300],
            })

        category_breakdown = [
            {
                "category": cat,
                "tested": v["tested"],
                "failed": v["failed"],
                "pass_rate": (v["tested"] - v["failed"]) / v["tested"] if v["tested"] else None,
            }
            for cat, v in sorted(by_category.items())
        ]

        return {
            "n_tests": len(per_test),
            "pass_rate": mean(successes) if successes else None,
            "fail_rate": (1.0 - mean(successes)) if successes else None,
            "mean_score": mean(scores) if scores else None,
            "mean_latency_ms": mean(latencies) if latencies else 0.0,
            "total_cost": sum(costs) if costs else 0.0,
            "refusal_count": refusal_count,
            "refusal_rate": (refusal_count / len(per_test)) if per_test else None,
            "per_test": per_test,
            "by_category": category_breakdown,
        }

    # ── Per-row CSV artifact ───────────────────────────────────────────

    @staticmethod
    def _truncate(text: Any, limit: int = 2000) -> str:
        """Coerce a cell value to str and cap its length so the CSV stays usable.

        Non-string outputs (dicts/lists from structured providers) are JSON-dumped
        so the cell is still a readable string rather than a Python repr.
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            try:
                text = json.dumps(text, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                text = str(text)
        if len(text) > limit:
            return text[: limit - 1] + "…"
        return text

    @staticmethod
    def _assertions_summary(grading: dict) -> str:
        """Compact per-assertion summary from gradingResult.componentResults[].

        Each component is a GradingResult: { pass: bool, reason: str,
        assertion: { type: str, value: any } }. Renders e.g.
        ``is-refusal: pass; contains: fail (expected '4')``. The expected value
        is only appended when the assertion failed and a value is present.
        """
        components = (grading or {}).get("componentResults") or []
        parts: list[str] = []
        for c in components:
            if not isinstance(c, dict):
                continue
            assertion = c.get("assertion") or {}
            a_type = assertion.get("type") if isinstance(assertion, dict) else None
            a_type = a_type or "assert"
            passed = c.get("pass")
            verdict = "pass" if passed is True else "fail" if passed is False else "?"
            seg = f"{a_type}: {verdict}"
            if passed is False and isinstance(assertion, dict):
                value = assertion.get("value")
                if value not in (None, ""):
                    value_str = value if isinstance(value, str) else json.dumps(value, default=str)
                    if len(value_str) > 120:
                        value_str = value_str[:119] + "…"
                    seg += f" (expected {value_str!r})"
            parts.append(seg)
        return "; ".join(parts)

    @classmethod
    def _build_per_test_csv(cls, results_payload: dict, mode: str = "eval") -> str:
        """Build the wide per-row results CSV (one row per test).

        Parses promptfoo's native results JSON — ``results.results[]`` where each
        element is an ``EvaluateResult`` with: ``prompt.raw`` (rendered prompt),
        ``response.output`` (model text), ``vars``, ``success``, ``score``,
        ``latencyMs``, ``cost``, and ``gradingResult.componentResults[]``. All key
        access is defensive (falls back to "" / None) since field names vary by
        promptfoo version. Redteam results share this shape; ``metadata.severity``
        is appended as an extra column when present (prompt = adversarial probe).
        """
        results_array = results_payload.get("results", [])
        if isinstance(results_array, dict):
            results_array = results_array.get("results", [])

        rows: list[dict[str, Any]] = []
        any_severity = False
        for idx, r in enumerate(results_array):
            if not isinstance(r, dict):
                continue
            prompt_obj = r.get("prompt")
            if isinstance(prompt_obj, dict):
                prompt_raw = prompt_obj.get("raw") or prompt_obj.get("label") or ""
            else:
                prompt_raw = prompt_obj or ""

            response_obj = r.get("response")
            if isinstance(response_obj, dict):
                output = response_obj.get("output")
            else:
                output = response_obj

            grading = r.get("gradingResult") or {}
            vars_obj = r.get("vars") or {}

            row: dict[str, Any] = {
                "test_index": r.get("testIdx", idx),
                "prompt": cls._truncate(prompt_raw),
                "response": cls._truncate(output),
                "pass": r.get("success"),
                "score": r.get("score"),
                "assertions": cls._assertions_summary(grading),
                "latency_ms": r.get("latencyMs"),
                "cost": r.get("cost"),
                "vars": cls._truncate(
                    json.dumps(vars_obj, ensure_ascii=False, default=str)
                ),
            }

            if mode == "redteam":
                metadata = r.get("metadata") or {}
                severity = metadata.get("severity") or (grading.get("metadata") or {}).get("severity")
                if severity:
                    any_severity = True
                row["severity"] = severity or ""

            rows.append(row)

        columns = [
            "test_index", "prompt", "response", "pass", "score",
            "assertions", "latency_ms", "cost", "vars",
        ]
        if mode == "redteam" and any_severity:
            columns.append("severity")

        df = pd.DataFrame(rows, columns=columns)
        return df.to_csv(index=False)

    # ── Main entrypoint ────────────────────────────────────────────────

    def evaluate(self, config_data: dict) -> Any:
        config = self.validate_config_form_data(config_data)

        workspace = Path(os.getcwd())

        # ── Dataset source resolution (eval mode only) ────────────────────
        # The core stack hands us the uploaded file as raw bytes (or None); the
        # plugin decides whether to use it. Layered precedence, fail-loud only
        # where it would otherwise silently do nothing:
        #   raw_yaml  >  uploaded eval-tests CSV  >  inline test_cases
        uploaded_bytes = self.get_input_data("eval-tests-file")  # bytes or None
        tests_csv_name: str | None = None

        if uploaded_bytes is not None and config.mode != "eval":
            # Redteam is generative — there's no row dataset. Ignore the file
            # rather than failing, and say so in the log.
            self.logger.info(
                "An eval-tests CSV was uploaded but mode is 'redteam'; ignoring it "
                "(redteam generates its own adversarial tests). Switch to eval mode "
                "to use the uploaded CSV."
            )
        elif uploaded_bytes is not None and config.mode == "eval":
            if config.raw_yaml.strip():
                # raw_yaml wins; the CSV is not referenced. Note it so the user
                # isn't surprised their upload was skipped.
                self.logger.info(
                    "Both raw_yaml and an eval-tests CSV are present; raw_yaml takes "
                    "precedence, so the uploaded CSV is ignored. Clear raw_yaml to use "
                    "the CSV instead."
                )
            else:
                # CSV wins over inline test_cases. Write it next to the config so
                # promptfoo resolves `tests: tests.csv` relative to the config dir,
                # and parses it natively (rows = tests, headers = {{vars}}).
                tests_csv_name = "tests.csv"
                (workspace / tests_csv_name).write_bytes(uploaded_bytes)
                self.upload_artifact(tests_csv_name, uploaded_bytes)
                if config.test_cases:
                    self.logger.info(
                        "An eval-tests CSV was uploaded; it takes precedence over the "
                        "inline test_cases, which are ignored for this run."
                    )

        # Fail fast on a misconfigured `eval` run: with no tests at all, promptfoo
        # has nothing to substitute template vars into and produces zero results —
        # a confusing silent success. Accept ANY of: raw_yaml, an uploaded CSV, or
        # at least one inline test_case.
        if (
            config.mode == "eval"
            and not config.raw_yaml.strip()
            and tests_csv_name is None
            and not config.test_cases
        ):
            raise RuntimeError(
                "Eval mode needs at least one test. Add a `test_cases` entry "
                "(vars + optional expected_contains), upload an eval-tests CSV, "
                "or switch `mode` to `redteam` for auto-generated adversarial tests."
            )

        config_yaml, provider_env = self._build_yaml(config, tests_csv_name=tests_csv_name)
        config_path = workspace / "promptfooconfig.yaml"
        config_path.write_text(config_yaml)
        self.upload_artifact("promptfooconfig.yaml", config_yaml.encode("utf-8"))

        results_path = workspace / "output" / "promptfoo_results.json"
        results_path.parent.mkdir(exist_ok=True)

        binary = self._resolve_promptfoo()

        env_vars = {**os.environ, **provider_env}
        # Ensure the bundled Node is discoverable by promptfoo's own child
        # processes (npx spawns `node`), so the run works with no Node on PATH.
        bundled_bin = self._bundled_node_bin()
        if bundled_bin:
            env_vars["PATH"] = f"{bundled_bin}{os.pathsep}{env_vars.get('PATH', '')}"
        env_vars.setdefault("PROMPTFOO_DISABLE_TELEMETRY", "1")
        env_vars.setdefault("PROMPTFOO_DISABLE_UPDATE", "1")
        env_vars.setdefault("PROMPTFOO_DISABLE_SHARE_INFO", "1")

        # Promptfoo gates redteam runs behind an "email verification" prompt
        # in the CLI. Set a non-empty email via `promptfoo config set email`
        # so the gate auto-passes. Idempotent + offline. Uses (in priority):
        #   1. user-supplied `promptfoo_email` config field
        #   2. PROMPTFOO_USER_EMAIL env override
        #   3. placeholder `vera@localhost`
        if config.mode == "redteam":
            # Disable cloud generation so attack synthesis stays local, avoiding
            # the email-verification gate.
            env_vars["PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION"] = "true"
            email = (
                (config.promptfoo_email or "").strip()
                or os.environ.get("PROMPTFOO_USER_EMAIL", "").strip()
                or "vera@localhost"
            )
            try:
                subprocess.run(
                    binary + ["config", "set", "email", email],
                    env=env_vars,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
            except Exception as exc:
                self.logger.warning(f"promptfoo config set email failed (continuing): {exc}")

        stdout_acc: list[str] = []
        stderr_acc: list[str] = []

        def _run(cmd: list[str]) -> subprocess.CompletedProcess:
            self.logger.info(f"Launching Promptfoo: {' '.join(cmd)}")
            return subprocess.run(
                cmd,
                cwd=str(workspace),
                env=env_vars,
                capture_output=True,
                text=True,
            )

        self.report_progress(
            TaskProgress(progress=0.0, extra={"phase": "starting", "mode": config.mode})
        )

        if config.mode == "redteam":
            # Step 1: generate adversarial tests into redteam.yaml.
            # The same target provider is used as the attack generator so
            # everything stays inside the user's chosen environment (e.g. ollama).
            generator_provider, _ = self._build_provider(config)
            redteam_yaml = workspace / "redteam.yaml"
            gen_cmd = binary + [
                "redteam", "generate",
                "--config", str(config_path),
                "--output", str(redteam_yaml),
                "--max-concurrency", str(config.max_concurrency),
                "--no-cache",
                "--force",
                "--no-progress-bar",
            ]
            proc = _run(gen_cmd)
            stdout_acc.append("===== redteam generate stdout =====\n" + (proc.stdout or ""))
            stderr_acc.append("===== redteam generate stderr =====\n" + (proc.stderr or ""))
            self.upload_artifact("promptfoo_stdout.log", "".join(stdout_acc).encode("utf-8"))
            self.upload_artifact("promptfoo_stderr.log", "".join(stderr_acc).encode("utf-8"))
            if proc.returncode != 0 or not redteam_yaml.exists():
                tail = (proc.stderr or proc.stdout or "")[-2000:]
                raise RuntimeError(
                    f"Promptfoo redteam generate failed (rc={proc.returncode}). Tail:\n{tail}"
                )
            self.upload_artifact("redteam.yaml", redteam_yaml.read_bytes())

            self.report_progress(
                TaskProgress(progress=0.5, extra={"phase": "executing"})
            )

            # Step 2: evaluate the generated tests against the target.
            eval_cmd = binary + [
                "eval",
                "--config", str(redteam_yaml),
                "--output", str(results_path),
                "--max-concurrency", str(config.max_concurrency),
                "--no-cache",
                "--no-table",
                "--no-progress-bar",
            ]
            proc = _run(eval_cmd)
            stdout_acc.append("===== eval stdout =====\n" + (proc.stdout or ""))
            stderr_acc.append("===== eval stderr =====\n" + (proc.stderr or ""))
        else:
            eval_cmd = binary + [
                "eval",
                "--config", str(config_path),
                "--output", str(results_path),
                "--max-concurrency", str(config.max_concurrency),
                "--no-cache",
                "--no-table",
                "--no-progress-bar",
            ]
            proc = _run(eval_cmd)
            stdout_acc.append("===== eval stdout =====\n" + (proc.stdout or ""))
            stderr_acc.append("===== eval stderr =====\n" + (proc.stderr or ""))

        self.upload_artifact("promptfoo_stdout.log", "".join(stdout_acc).encode("utf-8"))
        self.upload_artifact("promptfoo_stderr.log", "".join(stderr_acc).encode("utf-8"))

        # Promptfoo eval exits 100 when test failures occur — that's still a
        # successful run, just with regressions. Only treat rc!=0 AND missing
        # output as fatal.
        if not results_path.exists():
            tail = (proc.stderr or proc.stdout or "")[-2000:]
            raise RuntimeError(
                f"Promptfoo produced no results file (rc={proc.returncode}). Tail:\n{tail}"
            )

        payload = json.loads(results_path.read_text())
        self.upload_artifact("promptfoo_results.json", results_path.read_bytes())

        # Wide, scannable per-row CSV (one row per test) with the raw prompt and
        # model response inline. Best-effort: a parse failure here must never sink
        # an otherwise-successful eval, so it's logged and skipped. Redteam shares
        # promptfoo's results shape (prompt = adversarial probe, plus severity).
        try:
            per_test_csv = self._build_per_test_csv(payload, mode=config.mode)
            self.upload_artifact(
                "promptfoo_per_test.csv", per_test_csv.encode("utf-8")
            )
        except Exception as exc:  # noqa: BLE001 - never break the eval on a CSV hiccup
            self.logger.warning(
                "Skipping promptfoo_per_test.csv artifact: failed to build the "
                "per-row results CSV from the promptfoo results JSON (%s: %s).",
                type(exc).__name__,
                exc,
            )

        agg = self._aggregate(payload)
        agg["mode"] = config.mode

        per_test_path = workspace / "output" / "promptfoo_per_test.json"
        per_test_path.write_text(json.dumps(agg["per_test"], indent=2))
        self.upload_artifact(
            "promptfoo_per_test.json",
            json.dumps(agg["per_test"], indent=2).encode("utf-8"),
        )
        self.upload_artifact(
            "promptfoo_by_category.json",
            json.dumps(agg["by_category"], indent=2).encode("utf-8"),
        )

        n = max(agg["n_tests"], 1)
        for i in range(agg["n_tests"]):
            self.report_progress(
                TaskProgress(progress=(i + 1) / n, extra={"phase": "aggregating", "processed": i + 1, "total": n})
            )

        return agg

    # ── Emitted metrics ────────────────────────────────────────────────

    @staticmethod
    def _test_dims(t: dict, mode: str) -> dict[str, str | int | bool] | None:
        return {
            "mode":     mode,
            "category": str(t.get("category") or ""),
            "strategy": str(t.get("strategy") or ""),
        }

    # Aggregate, single-valued KPIs (one value per run). These populate the Key
    # Results scorecard and the summary table. The per-test breakdown is NOT
    # emitted as metrics any more — it lives in the wide `promptfoo_per_test.csv`
    # ("Per-run results"), so we don't repeat every test as long-format rows.

    @metric("pass_rate")
    def pass_rate_metric(self, output: dict) -> list[Measure]:
        v = output.get("pass_rate")
        if v is None:
            return []
        return [Measure(name="pass_rate", score=float(v), unit="rate (higher is better)",
                        description="Share of tests that passed all of their assertions.")]

    @metric("fail_rate")
    def fail_rate_metric(self, output: dict) -> list[Measure]:
        v = output.get("fail_rate")
        if v is None:
            return []
        return [Measure(name="fail_rate", score=float(v), unit="rate (lower is better)",
                        description="Share of tests that failed at least one assertion.")]

    @metric("refusal_rate")
    def refusal_rate_metric(self, output: dict) -> list[Measure]:
        v = output.get("refusal_rate")
        if v is None:
            return []
        return [Measure(name="refusal_rate", score=float(v), unit="rate",
                        description="Share of tests where the model refused to answer.")]

    @metric("mean_latency_ms")
    def mean_latency_metric(self, output: dict) -> list[Measure]:
        return [Measure(name="mean_latency_ms", score=float(output.get("mean_latency_ms") or 0.0),
                        unit="ms (lower is better)",
                        description="Average model response time across all tests.")]

    @metric("total_cost")
    def total_cost_metric(self, output: dict) -> list[Measure]:
        return [Measure(name="total_cost", score=float(output.get("total_cost") or 0.0),
                        unit="USD (lower is better)",
                        description="Total API cost across all tests.")]

    @metric("n_tests")
    def n_tests_metric(self, output: dict) -> list[Measure]:
        return [Measure(name="n_tests", score=float(output.get("n_tests") or 0),
                        unit="count",
                        description="Number of test cases run.")]
