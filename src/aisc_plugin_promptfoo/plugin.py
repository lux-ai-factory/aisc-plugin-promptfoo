import json
import os
import shutil
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

GENERATION_PROVIDER_TYPE = Literal[
    "same_as_target",
    "openai",
    "anthropic",
    "google",
    "ollama",
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


class ContextModel(BaseModel):
    """A named execution context for redteam — prompts are generated for each context."""
    id: str = Field(
        description="Unique, nonempty identifier for this context."
    )
    purpose: str = Field(
        default="",
        description="Optional context-specific system description, overrides the top-level purpose.",
    )
    vars: dict[str, str] = Field(
        default_factory=dict,
        description="Variables injected into target requests for this context.",
    )

    @field_validator("id")
    @classmethod
    def id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Context id must not be blank.")
        return v.strip()


class GraderExampleModel(BaseModel):
    """A grading example that calibrates the LLM judge used to score redteam results."""
    model_config = ConfigDict(populate_by_name=True)

    output: str = Field(description="Example model response text.")
    # "pass" is a Python reserved word; serialise as the upstream key.
    pass_result: bool = Field(
        alias="pass",
        serialization_alias="pass",
        default=True,
        description="Whether this response should be graded as passing.",
    )
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str = Field(default="", description="Explanation of the grading decision.")


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
            "hosted ids may be dated, so prefer a current model id for your provider. "
            "Hidden for custom-http (the endpoint controls its own model)."
        ),
    )
    target_endpoint: str = Field(
        default="http://host.docker.internal:11434",
        description=(
            "Endpoint URL for the model. Needed for `ollama` (for example "
            "http://host.docker.internal:11434) and for `custom-http` (your full chat "
            "endpoint URL, for example http://host.docker.internal:8766/chat). "
            "Hosted providers use their own API endpoint, so this is ignored there."
        ),
    )
    target_credential: str = Field(
        default="",
        description=(
            "API key for the target. Used by openai, anthropic, and google (it maps to "
            "OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY). For custom-http, it is "
            "sent as a Bearer token in the Authorization header when non-empty. "
            "Leave it blank for `ollama` and unauthenticated `custom-http`. Secret, never shown in reports."
        ),
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM providers. Hidden for custom-http.",
    )
    max_concurrency: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum concurrent requests during evaluation (promptfoo eval --max-concurrency).",
    )

    # ─── Custom-HTTP request/response mapping ────────────────────────────────
    http_body_template: str = Field(
        default='{"message": "{{prompt}}", "language": "en"}',
        title="HTTP request body (JSON)",
        description=(
            "JSON object sent as the request body for custom-http targets. Use `{{prompt}}` "
            "as the placeholder where the adversarial/eval prompt is injected. "
            "Example for the AISC leaderboard: "
            '{"message": "{{prompt}}", "language": "en"} — '
            "the `language` field here controls what language the request is sent in; "
            "it is separate from the redteam `languages` setting, which controls the "
            "language the attack prompts are generated in. Set both consistently."
        ),
    )
    http_response_transform: str = Field(
        default='json.replies ? json.replies.join("\\n") : (json.response || output)',
        title="HTTP response extraction (JS expression)",
        description=(
            "A JavaScript expression evaluated by promptfoo to extract the model's reply "
            "from the HTTP response. Available variables: `json` (parsed response body), "
            "`output` (raw response string), `context` (headers, status). "
            "AISC leaderboard default extracts the first string in `replies[]` or falls back "
            "to a top-level `response` field: "
            'json.replies ? json.replies.join("\\n") : (json.response || output). '
            "For a plain `{response: '...'}` shape, use: json.response || output."
        ),
    )

    # ─── Promptfoo account (redteam only) ─────────────────────────────────
    promptfoo_email: str = Field(
        default="",
        description=(
            "Optional, redteam only. The promptfoo CLI asks for an email before running a "
            "redteam. Put yours here to clear that prompt without any interaction. Leave it "
            "blank and a placeholder (vera@localhost) is used. Not needed for eval mode."
        ),
    )

    # ─── Attack-generation / grading provider (redteam only) ─────────────
    generation_provider_type: GENERATION_PROVIDER_TYPE = Field(
        default="same_as_target",
        title="Attack-generation provider",
        description=(
            "The LLM used to generate adversarial prompts and grade results. "
            "Distinct from the target being tested. "
            "`same_as_target` reuses the target provider and is suitable for openai/anthropic/"
            "google/ollama targets. For `custom-http` targets the HTTP endpoint cannot generate "
            "attacks, so choose a real LLM here (e.g. openai or anthropic) and supply an API key below."
        ),
    )
    generation_model: str = Field(
        default="gpt-4o-mini",
        title="Generation model",
        description=(
            "Model id for the attack-generation provider. Examples: gpt-4o-mini (openai), "
            "claude-3-5-haiku-20241022 (anthropic), gemini-2.0-flash-001 (google), "
            "llama3 (ollama). Ignored when generation_provider_type is same_as_target."
        ),
    )
    generation_credential: str = Field(
        default="",
        title="Generation provider API key",
        description=(
            "API key for the attack-generation provider (openai → OPENAI_API_KEY, "
            "anthropic → ANTHROPIC_API_KEY, google → GOOGLE_API_KEY). "
            "Ignored when generation_provider_type is same_as_target. Secret, never shown in reports."
        ),
    )
    generation_endpoint: str = Field(
        default="http://host.docker.internal:11434",
        title="Generation provider endpoint (ollama only)",
        description=(
            "Base URL for the ollama generation provider, e.g. http://host.docker.internal:11434. "
            "Only used when generation_provider_type is ollama."
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

    # ─── Redteam advanced options ──────────────────────────────────────────
    languages: list[str] = Field(
        default_factory=lambda: ["en"],
        title="Attack languages",
        description=(
            "Languages used to generate adversarial test messages (IETF language tags such as "
            "en, fr, de, lb). Promptfoo maps this to redteam.language in the YAML. "
            "A single entry is written as a string; multiple entries as a list. "
            "Note: this controls the language of the *generated attack prompts*. For "
            "custom-http targets you must also update the http_body_template `language` field "
            "to control the language of the actual HTTP request payload."
        ),
    )
    frameworks: list[str] = Field(
        default_factory=list,
        title="Compliance frameworks",
        description=(
            "Framework ids to include in promptfoo reports (e.g. owasp:llm, owasp:agentic, "
            "eu:ai-act). Omitted when empty, which includes all frameworks supported by the "
            "installed promptfoo version. This is a display/filter setting only — it does not "
            "automatically select attack categories or establish compliance."
        ),
    )
    test_generation_instructions: str = Field(
        default="",
        title="Test-generation instructions",
        description=(
            "Free-text guidance added to promptfoo's attack-generation system prompt "
            "(redteam.testGenerationInstructions). Use this to focus attacks on your domain, "
            "e.g. 'Focus on a database-backed model leaderboard. Use realistic requests "
            "involving rankings, comparisons, and filters.' Omitted when blank."
        ),
    )
    generation_delay_ms: int = Field(
        default=0,
        ge=0,
        title="Generation delay (milliseconds)",
        description=(
            "Milliseconds to wait between attack-generation requests "
            "(redteam.delay). Affects only the generation phase, not evaluation. "
            "When greater than zero, generation effectively runs at concurrency 1 "
            "(generation_max_concurrency is forced to 1 in the CLI invocation)."
        ),
    )
    generation_max_concurrency: int = Field(
        default=4,
        ge=1,
        le=32,
        title="Generation max concurrency",
        description=(
            "Maximum concurrent requests during attack generation "
            "(redteam.maxConcurrency / --max-concurrency on redteam generate). "
            "Separate from max_concurrency, which controls evaluation concurrency. "
            "Set to 1 when targeting a slow or rate-limited endpoint such as the leaderboard. "
            "Automatically forced to 1 when generation_delay_ms > 0."
        ),
    )
    max_chars_per_message: int | None = Field(
        default=None,
        ge=1,
        title="Max characters per message",
        description=(
            "Truncates generated attack prompts to this many characters "
            "(redteam.maxCharsPerMessage). Omitted when null."
        ),
    )
    inject_var: str = Field(
        default="",
        title="Inject variable",
        description=(
            "The prompt template variable that receives adversarial input "
            "(redteam.injectVar). Omitted when blank so promptfoo infers it from the "
            "target configuration. For single-input templates this inference is reliable. "
            "Provide an explicit value (e.g. question or message) only for multi-variable "
            "templates where the inference might be ambiguous."
        ),
    )
    contexts: list[ContextModel] = Field(
        default_factory=list,
        title="Redteam contexts",
        description=(
            "Named execution contexts for the redteam run (redteam.contexts). Each context "
            "generates an independent set of attack cases, which can surface context-sensitive "
            "vulnerabilities. Context variables are only injected into target requests when the "
            "target mapping actually references them. IDs must be unique and nonempty."
        ),
    )
    grader_examples: list[GraderExampleModel] = Field(
        default_factory=list,
        title="Grader examples",
        description=(
            "Global grading calibration examples (redteam.graderExamples). Each example "
            "shows the LLM judge a model response and the expected pass/fail verdict. "
            "Use this to teach the judge your domain-specific acceptance criteria."
        ),
    )

    # ─── Escape hatch ─────────────────────────────────────────────────────
    raw_yaml: str = Field(
        default="",
        description=(
            "Power-user escape hatch: paste a complete promptfoo YAML and it's used verbatim. "
            "This overrides ALL other fields, including an uploaded eval-tests CSV. "
            "The AISC `mode` field still controls whether `promptfoo eval` or "
            "`promptfoo redteam generate + eval` is invoked, so ensure your YAML contains "
            "the matching key (`providers`/`tests` for eval; `targets`/`redteam` for redteam). "
            "A YAML that contains `redteam:` with mode set to `eval` will be flagged."
        ),
    )

    @field_validator("languages")
    @classmethod
    def languages_nonempty(cls, v: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(lang.strip() for lang in v if lang.strip()))
        return cleaned if cleaned else ["en"]

    @model_validator(mode="after")
    def warn_http_generation(self) -> "PromptfooConfig":
        if (
            self.mode == "redteam"
            and self.target_provider == "custom-http"
            and self.generation_provider_type == "same_as_target"
        ):
            # Not an error — but the caller (_build_generation_provider) will emit
            # a warning. Keeping the model validator clean to avoid surprises.
            pass
        return self


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
                "stripped automatically. Hidden for custom-http."
            ),
        },
        "http_body_template": {
            "ui:widget": "textarea",
            "ui:options": {"rows": 4},
            "ui:help": (
                "JSON object for the HTTP request body. Use {{prompt}} where the "
                "adversarial or eval prompt should be injected."
            ),
        },
        "http_response_transform": {
            "ui:widget": "textarea",
            "ui:options": {"rows": 2},
            "ui:help": (
                "JavaScript expression to extract the model's reply from the HTTP response."
            ),
        },
        "raw_yaml": {"ui:widget": "textarea", "ui:options": {"rows": 16}},
        "custom_adversarial_prompts": {"ui:widget": "textarea", "ui:options": {"rows": 6}},
        "test_generation_instructions": {"ui:widget": "textarea", "ui:options": {"rows": 5}},
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
            "ui:help": "API key for openai/anthropic/google. Leave blank for ollama/custom-http (or enter Bearer token for custom-http).",
        },
        "generation_credential": {
            "ui:widget": "password",
            "ui:help": "API key for the attack-generation provider. Required when generation_provider_type is not same_as_target.",
        },
        "generation_model": {
            "ui:help": "Model id for the attack-generation provider, e.g. gpt-4o-mini, claude-3-5-haiku-20241022.",
        },
        "contexts": {
            "ui:help": "Named contexts generate independent attack sets. IDs must be unique.",
        },
        "grader_examples": {
            "ui:help": "Calibration examples for the LLM judge. Use 'pass' (the upstream YAML key) in the serialized form.",
            "items": {
                "pass_result": {
                    "ui:title": "pass",
                },
            },
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
        "generation_provider_type",
        "generation_model",
        "generation_credential",
        "generation_endpoint",
        "languages",
        "frameworks",
        "test_generation_instructions",
        "generation_delay_ms",
        "generation_max_concurrency",
        "max_chars_per_message",
        "inject_var",
        "contexts",
        "grader_examples",
    )
    # Fields specific to custom-http target
    HTTP_ONLY_FIELDS = (
        "http_body_template",
        "http_response_transform",
    )
    # Fields hidden for custom-http (endpoint controls its own model)
    NON_HTTP_TARGET_FIELDS = (
        "target_model",
        "temperature",
    )
    # Generation-provider sub-fields hidden when same_as_target
    GENERATION_DETAIL_FIELDS = (
        "generation_model",
        "generation_credential",
        "generation_endpoint",
    )

    @staticmethod
    def _drop_field(form_data, field: str) -> None:
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

        selected_provider = "ollama"
        if isinstance(form_data, dict):
            selected_provider = form_data.get("target_provider") or "ollama"
        elif form_data is not None:
            selected_provider = getattr(form_data, "target_provider", "ollama")

        selected_gen_type = "same_as_target"
        if isinstance(form_data, dict):
            selected_gen_type = form_data.get("generation_provider_type") or "same_as_target"
        elif form_data is not None:
            selected_gen_type = getattr(form_data, "generation_provider_type", "same_as_target")

        props = schema.get("properties", {})
        required = schema.get("required", [])

        def _hide(field: str) -> None:
            props.pop(field, None)
            if field in required:
                required.remove(field)
            ui_schema.setdefault(field, {})["ui:widget"] = "hidden"
            self._drop_field(form_data, field)

        # Mode-based visibility
        hide_for_mode = self.EVAL_ONLY_FIELDS if selected_mode == "redteam" else self.REDTEAM_ONLY_FIELDS
        for field in hide_for_mode:
            _hide(field)

        # Provider-based visibility
        if selected_provider == "custom-http":
            for field in self.NON_HTTP_TARGET_FIELDS:
                _hide(field)
        else:
            for field in self.HTTP_ONLY_FIELDS:
                _hide(field)

        if selected_provider in ("ollama", "custom-http"):
            if "target_credential" in required:
                required.remove("target_credential")
            ui_schema.setdefault("target_credential", {})["ui:widget"] = "hidden"
            self._drop_field(form_data, "target_credential")
        else:
            if "target_endpoint" in required:
                required.remove("target_endpoint")
            ui_schema.setdefault("target_endpoint", {})["ui:widget"] = "hidden"
            self._drop_field(form_data, "target_endpoint")

        # Generation-provider sub-fields visibility (redteam mode only)
        if selected_mode == "redteam" and selected_gen_type == "same_as_target":
            for field in self.GENERATION_DETAIL_FIELDS:
                _hide(field)
        if selected_mode == "redteam" and selected_gen_type != "ollama":
            _hide("generation_endpoint")

        return form_data, schema, ui_schema

    # ── Provider building ────────────────────────────────────────────────

    def _build_provider(self, config: "PromptfooConfig") -> tuple[dict[str, Any], dict[str, str]]:
        """Return (provider_dict, extra_env_vars) for the chosen target."""
        env: dict[str, str] = {}
        provider_id: str
        provider_cfg: dict[str, Any]

        model = _bare_model(config.target_model)

        if config.target_provider == "ollama":
            provider_id = f"ollama:chat:{model or 'llama3:latest'}"
            provider_cfg = {"temperature": config.temperature}
            host = (config.target_endpoint or "http://host.docker.internal:11434").rstrip("/")
            if host.endswith("/v1"):
                host = host[:-3].rstrip("/")
            env["OLLAMA_BASE_URL"] = host
        elif config.target_provider == "openai":
            provider_id = f"openai:{model or 'gpt-4o-mini'}"
            provider_cfg = {"temperature": config.temperature}
            if config.target_credential:
                env["OPENAI_API_KEY"] = config.target_credential
            if config.target_endpoint:
                env["OPENAI_BASE_URL"] = config.target_endpoint.rstrip("/")
        elif config.target_provider == "anthropic":
            provider_id = f"anthropic:messages:{model or 'claude-3-5-sonnet-20241022'}"
            provider_cfg = {"temperature": config.temperature}
            if config.target_credential:
                env["ANTHROPIC_API_KEY"] = config.target_credential
        elif config.target_provider == "google":
            provider_id = f"google:{model or 'gemini-1.5-flash'}"
            provider_cfg = {"temperature": config.temperature}
            if config.target_credential:
                env["GOOGLE_API_KEY"] = config.target_credential
                env.setdefault("GEMINI_API_KEY", config.target_credential)
        elif config.target_provider == "custom-http":
            provider_id = "http"
            # Parse the body template from JSON so promptfoo sees a native object
            # (not a raw string). {{prompt}} is a promptfoo template placeholder, not
            # a Jinja variable, so it round-trips through json.loads correctly.
            try:
                body = json.loads(config.http_body_template)
            except (json.JSONDecodeError, TypeError):
                body = config.http_body_template  # fall back to raw string
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if config.target_credential:
                headers["Authorization"] = f"Bearer {config.target_credential}"
            provider_cfg = {
                "url": config.target_endpoint,
                "method": "POST",
                "headers": headers,
                "body": body,
                "transformResponse": config.http_response_transform,
            }
        else:
            raise ValueError(f"Unknown target_provider: {config.target_provider}")

        return {"id": provider_id, "config": provider_cfg}, env

    def _build_generation_provider(
        self, config: "PromptfooConfig"
    ) -> tuple[str | dict[str, Any] | None, dict[str, str]]:
        """Build the attack-generation / grading provider config.

        Returns (provider, extra_env). `provider` is either a string id (for
        simple LLM providers), a full config dict (for ollama with a custom
        endpoint), or None (when promptfoo should use its own default).

        For non-custom-http targets with same_as_target, the target provider id
        string is reused directly — promptfoo can use any LLM provider for
        generation, not just the target.

        For custom-http targets with same_as_target, the HTTP endpoint cannot
        generate attacks; a warning is logged and None is returned so promptfoo
        falls back to its built-in default (requires a PROMPTFOO_OPENAI_API_KEY
        or equivalent in the environment).
        """
        gen_type = config.generation_provider_type
        env: dict[str, str] = {}

        if gen_type == "same_as_target":
            if config.target_provider == "custom-http":
                self.logger.warning(
                    "generation_provider_type is 'same_as_target' but target_provider is "
                    "'custom-http'. An HTTP endpoint cannot generate adversarial prompts. "
                    "Set generation_provider_type to openai/anthropic/google/ollama and "
                    "supply an API key in generation_credential. Falling back to promptfoo's "
                    "built-in default generation provider (requires OPENAI_API_KEY in env)."
                )
                return None, env
            # Reuse the target provider id string for generation.
            target_provider, _ = self._build_provider(config)
            return target_provider["id"], env

        # Explicit generation provider
        gen_model = (config.generation_model or "gpt-4o-mini").strip()
        if gen_type == "openai":
            provider: str | dict[str, Any] = f"openai:{gen_model}"
            if config.generation_credential:
                env["OPENAI_API_KEY"] = config.generation_credential
        elif gen_type == "anthropic":
            provider = f"anthropic:messages:{gen_model}"
            if config.generation_credential:
                env["ANTHROPIC_API_KEY"] = config.generation_credential
        elif gen_type == "google":
            provider = f"google:{gen_model}"
            if config.generation_credential:
                env["GOOGLE_API_KEY"] = config.generation_credential
                env.setdefault("GEMINI_API_KEY", config.generation_credential)
        elif gen_type == "ollama":
            host = (config.generation_endpoint or "http://host.docker.internal:11434").rstrip("/")
            if host.endswith("/v1"):
                host = host[:-3].rstrip("/")
            env["OLLAMA_BASE_URL"] = host
            provider = f"ollama:chat:{gen_model}"
        else:
            raise ValueError(f"Unknown generation_provider_type: {gen_type}")

        return provider, env

    # ── YAML composition ────────────────────────────────────────────────

    def _build_eval_yaml(
        self,
        config: "PromptfooConfig",
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

        doc: dict[str, Any] = {
            "description": "AISC Promptfoo eval",
            "providers": [provider],
            "prompts": config.prompts or ["{{prompt}}"],
        }
        if default_assertions:
            doc["defaultTest"] = {"assert": default_assertions}

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

    def _build_redteam_yaml(
        self,
        config: "PromptfooConfig",
        provider: dict[str, Any],
        generation_provider: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        plugin_list: list[Any] = list(config.attack_categories)
        custom_intents = [
            line.strip()
            for line in (config.custom_adversarial_prompts or "").splitlines()
            if line.strip()
        ]
        if custom_intents:
            plugin_list.append({"id": "intent", "config": {"intent": custom_intents}})

        redteam_cfg: dict[str, Any] = {
            "purpose": config.purpose,
            "numTests": config.num_tests_per_category,
            "plugins": plugin_list,
            "strategies": list(config.strategies),
        }

        # Attack-generation provider — must be an LLM, not the HTTP target.
        # Pass the full provider dict when it carries a config (e.g. ollama),
        # or just the string id for simple API providers.
        if generation_provider is not None:
            redteam_cfg["provider"] = generation_provider

        # Languages (string when single, list when multiple)
        languages = [l for l in config.languages if l.strip()]
        if languages:
            redteam_cfg["language"] = languages[0] if len(languages) == 1 else languages

        if config.frameworks:
            redteam_cfg["frameworks"] = list(config.frameworks)

        if config.test_generation_instructions.strip():
            redteam_cfg["testGenerationInstructions"] = config.test_generation_instructions.strip()

        if config.generation_delay_ms > 0:
            redteam_cfg["delay"] = config.generation_delay_ms

        # Always write maxConcurrency so the generation phase honours the field.
        # (The CLI --max-concurrency flag on redteam generate overlaps with
        # redteam.maxConcurrency; the YAML key takes precedence when both present.)
        effective_gen_concurrency = config.generation_max_concurrency
        if config.generation_delay_ms > 0:
            effective_gen_concurrency = 1
        redteam_cfg["maxConcurrency"] = effective_gen_concurrency

        if config.max_chars_per_message is not None:
            redteam_cfg["maxCharsPerMessage"] = config.max_chars_per_message

        if config.inject_var.strip():
            redteam_cfg["injectVar"] = config.inject_var.strip()

        if config.contexts:
            ctx_list: list[dict[str, Any]] = []
            for c in config.contexts:
                ctx: dict[str, Any] = {"id": c.id}
                if c.purpose.strip():
                    ctx["purpose"] = c.purpose.strip()
                if c.vars:
                    ctx["vars"] = c.vars
                ctx_list.append(ctx)
            redteam_cfg["contexts"] = ctx_list

        if config.grader_examples:
            redteam_cfg["graderExamples"] = [
                {
                    "output": g.output,
                    "pass": g.pass_result,
                    "score": g.score,
                    "reason": g.reason,
                }
                for g in config.grader_examples
            ]

        return {
            "description": "AISC Promptfoo redteam",
            "targets": [provider],
            "redteam": redteam_cfg,
        }

    def _build_yaml(
        self, config: "PromptfooConfig", tests_csv_name: str | None = None
    ) -> tuple[str, dict[str, str]]:
        if config.raw_yaml.strip():
            # Detect obvious mode/content mismatch so the user gets an actionable error.
            raw_lower = config.raw_yaml.lower()
            if config.mode == "eval" and "redteam:" in raw_lower:
                raise RuntimeError(
                    "mode is 'eval' but raw_yaml contains a 'redteam:' key. "
                    "Either switch mode to 'redteam' or remove the redteam block from raw_yaml."
                )
            if config.mode == "redteam" and "redteam:" not in raw_lower:
                raise RuntimeError(
                    "mode is 'redteam' but raw_yaml has no 'redteam:' key. "
                    "Either switch mode to 'eval' or add a redteam block to raw_yaml."
                )
            return config.raw_yaml, {}

        provider, env = self._build_provider(config)
        if config.mode == "redteam":
            gen_provider, gen_env = self._build_generation_provider(config)
            env.update(gen_env)
            doc = self._build_redteam_yaml(config, provider, gen_provider)
        else:
            doc = self._build_eval_yaml(config, provider, tests_csv_name=tests_csv_name)
        return yaml.safe_dump(doc, sort_keys=False), env

    # ── CLI invocation ──────────────────────────────────────────────────

    @staticmethod
    def _bundled_node_bin() -> Path | None:
        try:
            from nodejs_wheel.executable import ROOT_DIR
        except Exception:
            return None
        bin_dir = Path(ROOT_DIR) / "bin"
        return bin_dir if bin_dir.is_dir() else None

    def _resolve_promptfoo(self) -> list[str]:
        binary = shutil.which("promptfoo")
        if binary:
            return [binary]
        npx = shutil.which("npx")
        if npx:
            return [npx, "-y", "promptfoo@latest"]
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

    @staticmethod
    def _redact_yaml(yaml_text: str) -> str:
        """Replace credential values in a YAML string for safe logging."""
        import re
        # Redact simple key: value patterns for known credential keys.
        result = re.sub(
            r"((?:apiKey|api_key|apikey|password|credential|OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY)\s*:\s*)['\"]?[A-Za-z0-9_\-\.]{8,}['\"]?",
            r"\1<redacted>",
            yaml_text,
            flags=re.IGNORECASE,
        )
        # Redact Authorization header values (Bearer token or bare token).
        result = re.sub(
            r"(Authorization\s*:\s*['\"]?(?:Bearer\s+)?)[A-Za-z0-9_\-\.]{8,}(['\"]?)",
            r"\1<redacted>\2",
            result,
            flags=re.IGNORECASE,
        )
        return result

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
        uploaded_bytes = self.get_input_data("eval-tests-file")
        tests_csv_name: str | None = None

        if uploaded_bytes is not None and config.mode != "eval":
            self.logger.info(
                "An eval-tests CSV was uploaded but mode is 'redteam'; ignoring it "
                "(redteam generates its own adversarial tests). Switch to eval mode "
                "to use the uploaded CSV."
            )
        elif uploaded_bytes is not None and config.mode == "eval":
            if config.raw_yaml.strip():
                self.logger.info(
                    "Both raw_yaml and an eval-tests CSV are present; raw_yaml takes "
                    "precedence, so the uploaded CSV is ignored. Clear raw_yaml to use "
                    "the CSV instead."
                )
            else:
                tests_csv_name = "tests.csv"
                (workspace / tests_csv_name).write_bytes(uploaded_bytes)
                self.upload_artifact(tests_csv_name, uploaded_bytes)
                if config.test_cases:
                    self.logger.info(
                        "An eval-tests CSV was uploaded; it takes precedence over the "
                        "inline test_cases, which are ignored for this run."
                    )

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

        # Upload a redacted copy for diagnostics (never expose credentials).
        redacted_yaml = self._redact_yaml(config_yaml)
        self.upload_artifact("promptfooconfig.redacted.yaml", redacted_yaml.encode("utf-8"))

        results_path = workspace / "output" / "promptfoo_results.json"
        results_path.parent.mkdir(exist_ok=True)

        binary = self._resolve_promptfoo()

        env_vars = {**os.environ, **provider_env}
        bundled_bin = self._bundled_node_bin()
        if bundled_bin:
            env_vars["PATH"] = f"{bundled_bin}{os.pathsep}{env_vars.get('PATH', '')}"
        env_vars.setdefault("PROMPTFOO_DISABLE_TELEMETRY", "1")
        env_vars.setdefault("PROMPTFOO_DISABLE_UPDATE", "1")
        env_vars.setdefault("PROMPTFOO_DISABLE_SHARE_INFO", "1")

        # Log the promptfoo version for diagnostics.
        try:
            ver_proc = subprocess.run(
                binary + ["--version"],
                capture_output=True, text=True, env=env_vars, timeout=30,
            )
            pf_version = (ver_proc.stdout or ver_proc.stderr or "unknown").strip().splitlines()[0]
            self.logger.info(f"Promptfoo version: {pf_version}")
        except Exception as exc:
            self.logger.info(f"Could not determine promptfoo version: {exc}")

        if config.mode == "redteam":
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
                    capture_output=True, text=True, check=False, timeout=30,
                )
            except Exception as exc:
                self.logger.warning(f"promptfoo config set email failed (continuing): {exc}")

        stdout_acc: list[str] = []
        stderr_acc: list[str] = []

        def _run(cmd: list[str]) -> subprocess.CompletedProcess:
            # Log command with binary path; credentials never appear in cmd.
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
            # Generation concurrency: use generation_max_concurrency, forced to 1
            # when generation_delay_ms is set (delay + concurrency > 1 is contradictory).
            gen_concurrency = config.generation_max_concurrency
            if config.generation_delay_ms > 0:
                gen_concurrency = 1
                self.logger.info(
                    f"generation_delay_ms={config.generation_delay_ms} ms is set; "
                    "forcing generation concurrency to 1."
                )

            redteam_yaml = workspace / "redteam.yaml"
            gen_cmd = binary + [
                "redteam", "generate",
                "--config", str(config_path),
                "--output", str(redteam_yaml),
                "--max-concurrency", str(gen_concurrency),
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

        if not results_path.exists():
            tail = (proc.stderr or proc.stdout or "")[-2000:]
            raise RuntimeError(
                f"Promptfoo produced no results file (rc={proc.returncode}). Tail:\n{tail}"
            )

        payload = json.loads(results_path.read_text())
        self.upload_artifact("promptfoo_results.json", results_path.read_bytes())

        try:
            per_test_csv = self._build_per_test_csv(payload, mode=config.mode)
            self.upload_artifact(
                "promptfoo_per_test.csv", per_test_csv.encode("utf-8")
            )
        except Exception as exc:  # noqa: BLE001
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
