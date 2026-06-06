import json
import os
import shutil
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from aisc_plugin_interface import (
    BaseEvaluationPlugin,
    Measure,
    PluginFeatureFlags,
    TaskProgress,
    metric,
    ChartType,
    MetricVisualization,
)


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
    "harmful:sexual",
    "harmful:self-harm",
    "harmful:weapons",
    "harmful:cybercrime",
    "harmful:misinformation-disinformation",
    # Bias
    "bias:gender",
    "bias:race",
    "bias:age",
    "bias:religion",
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
        description="Variable substitutions for the prompt template (e.g. {question: 'What is 2+2?'}).",
    )
    expected_contains: str = Field(
        default="",
        description="Optional: response must contain this substring to pass.",
    )


class PromptfooConfig(BaseModel):
    mode: Mode = Field(
        default="redteam",
        description="`eval` = functional tests against the LLM. `redteam` = automated adversarial security scan.",
    )

    # ─── Target system ─────────────────────────────────────────────────────
    target_provider: TARGET_PROVIDER = Field(
        default="ollama",
        title="Target provider",
        description=(
            "Provider family of the system under test. Promptfoo needs this to build its "
            "provider:model id (e.g. ollama:chat:..., anthropic:messages:..., openai:...). "
            "Pick the family that matches the target model below."
        ),
    )
    target_model: MODELS = Field(
        default="ollama_chat/llama3",
        title="Target model (under test)",
        description=(
            "Model under test. Pick one whose family matches the provider above "
            "(gpt-* -> openai, anthropic/* -> anthropic, gemini/* -> google, "
            "ollama_chat/* -> ollama). The provider prefix is stripped automatically "
            "for promptfoo's provider:model id."
        ),
    )
    target_endpoint: str = Field(
        default="http://host.docker.internal:11434",
        description=(
            "Endpoint URL. Required for `ollama` (e.g. `http://host.docker.internal:11434`) "
            "and `custom-http` (full chat endpoint). Ignored for hosted providers."
        ),
    )
    target_credential: str = Field(
        default="",
        description="API key (for openai/anthropic/google). Leave blank for `ollama`/`custom-http`.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_concurrency: int = Field(default=4, ge=1, le=32)

    # ─── Promptfoo account (redteam only) ─────────────────────────────────
    promptfoo_email: str = Field(
        default="",
        description=(
            "Optional. Promptfoo CLI gates the redteam flow behind an email "
            "verification prompt. Provide your email to bypass it non-interactively. "
            "Leave blank for `eval` mode or to use a placeholder (`vera@localhost`)."
        ),
    )

    # ─── Eval mode ────────────────────────────────────────────────────────
    prompts: list[str] = Field(
        default_factory=lambda: ["{{question}}"],
        description="Prompt templates (use `{{var}}` placeholders).",
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
            "Test cases for `eval` mode. Each case binds variables + optional "
            "expected substring. Must be non-empty in eval mode (promptfoo "
            "produces zero results otherwise)."
        ),
    )
    assertion_types: list[ASSERTION_TYPE] = Field(
        default_factory=lambda: ["llm-rubric", "is-refusal"],
        description="Default assertions applied to every test case (combined with each test's `expected_contains`).",
    )
    llm_rubric_criteria: str = Field(
        default="Response is accurate, helpful, and non-evasive.",
        description="Rubric text for the `llm-rubric` assertion.",
    )

    # ─── Redteam mode ─────────────────────────────────────────────────────
    purpose: str = Field(
        default="General-purpose chatbot.",
        description=(
            "One-sentence description of what the system under test is supposed to do. "
            "Promptfoo uses this to tailor adversarial prompts."
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
            "Built-in adversarial categories. Promptfoo auto-generates the attack prompts. "
            "Note: many `harmful:*`, `pii:*`, `bias:*`, and security categories require "
            "promptfoo's cloud generation service (run `promptfoo auth login` first). "
            "Locally-capable defaults: hallucination, politics, excessive-agency, "
            "overreliance, imitation, donotanswer, harmbench, beavertails, xstest, toxic-chat."
        ),
    )
    strategies: list[STRATEGY] = Field(
        default_factory=lambda: ["basic"],
        description=(
            "Attack delivery strategies. 'basic', 'base64', 'leetspeak', 'multilingual' run locally. "
            "'jailbreak' requires promptfoo cloud or a self-hosted PROMPTFOO_REMOTE_GENERATION_URL."
        ),
    )
    num_tests_per_category: int = Field(default=5, ge=1, le=50)
    custom_adversarial_prompts: str = Field(
        default="",
        description=(
            "Optional. One adversarial seed prompt per line. Adds the `intent` plugin so "
            "promptfoo transforms each seed into a test case alongside the built-in attacks."
        ),
    )

    # ─── Escape hatch ─────────────────────────────────────────────────────
    raw_yaml: str = Field(
        default="",
        description="Power-user: paste a complete promptfoo YAML. Overrides every other field above.",
    )


class PromptfooPlugin(BaseEvaluationPlugin[PromptfooConfig]):
    plugin_name = "Promptfoo"
    ui_icon = "policy"

    form_ui_schema = {
        "target_model": {
            "ui:help": (
                "Choose a model whose family matches the provider. ollama_chat/* are local "
                "(set the endpoint above); the provider prefix is stripped automatically."
            ),
        },
        "raw_yaml": {"ui:widget": "textarea", "ui:options": {"rows": 16}},
        "custom_adversarial_prompts": {"ui:widget": "textarea", "ui:options": {"rows": 6}},
        "prompts": {"ui:options": {"orderable": True}},
        "test_cases": {"ui:options": {"orderable": True}},
        "target_credential": {"ui:widget": "password"},
    }

    @property
    def feature_flags(self) -> PluginFeatureFlags:
        return PluginFeatureFlags(can_parse_config_from_dataset=False)

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        return [
            MetricVisualization(
                chart_type=ChartType.TABLE,
                metrics=["success", "score", "latency_ms", "cost", "refusal"]
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

        # Handle provider-specific fields
        selected_provider = "ollama"
        if isinstance(form_data, dict):
            selected_provider = form_data.get("target_provider") or "ollama"
        elif form_data is not None:
            selected_provider = getattr(form_data, "target_provider", "ollama")

        if selected_provider in ("ollama", "custom-http"):
            # Hide credential
            if "target_credential" in required:
                required.remove("target_credential")
            ui_schema.setdefault("target_credential", {})["ui:widget"] = "hidden"
        else:
            # Hide endpoint
            if "target_endpoint" in required:
                required.remove("target_endpoint")
            ui_schema.setdefault("target_endpoint", {})["ui:widget"] = "hidden"

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

    def _build_eval_yaml(self, config: PromptfooConfig, provider: dict[str, Any]) -> dict[str, Any]:
        default_assertions: list[dict[str, Any]] = []
        for atype in config.assertion_types:
            if atype == "llm-rubric":
                default_assertions.append({"type": "llm-rubric", "value": config.llm_rubric_criteria})
            elif atype == "is-refusal":
                default_assertions.append({"type": "is-refusal"})
            elif atype == "is-json":
                default_assertions.append({"type": "is-json"})
            # text-substring/regex assertions only make sense per test; skip at default level

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

        doc: dict[str, Any] = {
            "description": "AISC Promptfoo eval",
            "providers": [provider],
            "prompts": config.prompts or ["{{prompt}}"],
        }
        if default_assertions:
            doc["defaultTest"] = {"assert": default_assertions}
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

    def _build_yaml(self, config: PromptfooConfig) -> tuple[str, dict[str, str]]:
        if config.raw_yaml.strip():
            return config.raw_yaml, {}

        provider, env = self._build_provider(config)
        if config.mode == "redteam":
            doc = self._build_redteam_yaml(config, provider)
        else:
            doc = self._build_eval_yaml(config, provider)
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

    # ── Main entrypoint ────────────────────────────────────────────────

    def evaluate(self, config_data: dict) -> Any:
        config = self.validate_config_form_data(config_data)

        # Fail fast on a misconfigured `eval` run: without test cases there is
        # no way for promptfoo to substitute template vars, so it produces
        # zero results — confusing silent success. Force the user to add at
        # least one test case OR switch to redteam mode.
        if config.mode == "eval" and not config.raw_yaml.strip() and not config.test_cases:
            raise RuntimeError(
                "Eval mode requires at least one `test_cases` entry "
                "(vars + optional expected_contains). Add a test case in "
                "the config form, or switch `mode` to `redteam` for "
                "auto-generated adversarial tests."
            )

        workspace = Path(os.getcwd())
        config_yaml, provider_env = self._build_yaml(config)
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

    @metric("success")
    def success_metric(self, output: dict) -> list[Measure]:
        mode = output.get("mode") or "eval"
        measures: list[Measure] = []
        for t in output.get("per_test", []):
            success = t.get("success")
            if success is None:
                continue
            measures.append(Measure(
                name="success",
                score=1.0 if success else 0.0,
                unit="bool",
                dimensions=self._test_dims(t, mode),
                description="1.0 if test passed all assertions, else 0.0.",
            ))
        return measures

    @metric("score")
    def score_metric(self, output: dict) -> list[Measure]:
        mode = output.get("mode") or "eval"
        measures: list[Measure] = []
        for t in output.get("per_test", []):
            score = t.get("score")
            if score is None:
                continue
            measures.append(Measure(
                name="score",
                score=float(score),
                unit="score",
                dimensions=self._test_dims(t, mode),
                description="Raw assertion score reported by promptfoo for this test.",
            ))
        return measures

    @metric("latency_ms")
    def latency_metric(self, output: dict) -> list[Measure]:
        mode = output.get("mode") or "eval"
        measures: list[Measure] = []
        for t in output.get("per_test", []):
            latency = t.get("latencyMs")
            if latency is None:
                continue
            measures.append(Measure(
                name="latency_ms",
                score=float(latency),
                unit="ms",
                dimensions=self._test_dims(t, mode),
                description="Time the model took to respond on this test.",
            ))
        return measures

    @metric("cost")
    def cost_metric(self, output: dict) -> list[Measure]:
        mode = output.get("mode") or "eval"
        measures: list[Measure] = []
        for t in output.get("per_test", []):
            cost = t.get("cost")
            if cost is None:
                continue
            measures.append(Measure(
                name="cost",
                score=float(cost),
                unit="USD",
                dimensions=self._test_dims(t, mode),
                description="Per-test API cost in USD.",
            ))
        return measures

    @metric("refusal")
    def refusal_metric(self, output: dict) -> list[Measure]:
        mode = output.get("mode") or "eval"
        measures: list[Measure] = []
        for t in output.get("per_test", []):
            reason = (t.get("reason") or "").lower()
            refused = "refus" in reason
            measures.append(Measure(
                name="refusal",
                score=1.0 if refused else 0.0,
                unit="bool",
                dimensions=self._test_dims(t, mode),
                description="1.0 if the grading reason indicates the model refused, else 0.0.",
            ))
        return measures
