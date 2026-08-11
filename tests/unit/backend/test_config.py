"""Unit tests for `backend/core/config.py`.

The most valuable test here is `TestEnvExampleIsInSync`. `.env.example` is the
only documentation a new engineer reads before their first `make bootstrap`, and
it is the file most likely to rot: a setting gets added to `config.py` and nobody
remembers to document it, or a variable is renamed and the example keeps the old
name. Both failures are silent and both waste somebody's afternoon. The test
makes them loud.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic_settings import BaseSettings

from backend.core.config import (
    Environment,
    LogFormat,
    Settings,
    get_settings,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_EXAMPLE = REPO_ROOT / ".env.example"

# Connector credentials are documented in `.env.example` but deliberately absent
# from `Settings`: `docs/architecture.md` §6.2 rule 2 forbids `connectors/` from
# importing `backend/core/config.py`, so credentials reach a connector as
# constructor arguments supplied by `services/connector_service.py`.
CREDENTIAL_PREFIXES = (
    "GITHUB_", "SLACK_", "JIRA_", "CONFLUENCE_", "NOTION_", "SEMANTIC_SCHOLAR_",
    "HUGGINGFACE_", "GOOGLE_OAUTH_",
)
"""Per-connector secrets, which `.env.example` documents but `config.py` never reads.

Connectors take their credentials at construction rather than from global
settings, so these have no `Settings` field to match and would otherwise trip
`test_every_documented_variable_is_used`.

The list is exactly the eight shipped connectors plus the two step-13 ones. It is
deliberately not a wildcard: when a connector is deleted, its prefix has to be
removed here too, and that failing test is the reminder to delete its block from
`.env.example` rather than leave a key nobody will ever fill in."""


def _declared_in_env_example() -> list[str]:
    text = ENV_EXAMPLE.read_text()
    return [m.group(1) for m in re.finditer(r"^([A-Z][A-Z0-9_]*)=", text, re.MULTILINE)]


def _expected_env_names() -> dict[str, str]:
    """Every variable name `Settings` will actually look up, and its owning group."""
    names: dict[str, str] = {}
    for field in Settings.model_fields.values():
        group = field.default_factory
        if not (isinstance(group, type) and issubclass(group, BaseSettings)):
            continue
        prefix = group.model_config.get("env_prefix", "")
        for name, sub in group.model_fields.items():
            alias = sub.alias or sub.validation_alias
            key = alias if isinstance(alias, str) else f"{prefix}{name}"
            names[key.upper()] = group.__name__
    return names


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every OmniSense variable so tests observe declared defaults.

    Without this the suite passes or fails depending on whose shell it runs in.
    """
    for key in list(_expected_env_names()) + _declared_in_env_example():
        monkeypatch.delenv(key, raising=False)


def _settings(**env: object) -> Settings:
    """Build Settings from the environment only, ignoring any local `.env`."""
    return Settings(_env_file=None, **{})  # type: ignore[call-arg]


class TestEnvExampleActuallyWorks:
    """The template must produce a loadable configuration, not merely a documented one.

    `TestEnvExampleIsInSync` compares two lists of *names*. It passes happily
    while the file is unusable, and it did: `.env.example` shipped `LLM_EFFORT=`
    with a blank value against a strict-enum field, so `cp .env.example .env`
    followed by anything at all died at import with a validation error. Every
    name was documented. The file could not be loaded.

    So this reads the template the way `make env` does -- as the environment --
    and constructs the real `Settings`. It is the only test here that would have
    caught that, and the class of bug it guards is "the first command a new user
    runs does not work".
    """

    @staticmethod
    def _template_environment() -> dict[str, str]:
        """`.env.example` parsed as `KEY=value`, blanks included.

        Blanks are the point and must not be filtered out -- an unset optional is
        exactly the case that broke.
        """
        pairs: dict[str, str] = {}
        for line in ENV_EXAMPLE.read_text().splitlines():
            match = re.match(r"^([A-Z][A-Z0-9_]*)=(.*)$", line)
            if match:
                pairs[match.group(1)] = match.group(2).split("#")[0].strip()
        return pairs

    def test_the_shipped_template_constructs_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key, value in self._template_environment().items():
            monkeypatch.setenv(key, value)
        # `_env_file=None` so the developer's own .env cannot mask a broken
        # template -- the machine that runs this most likely has a working one.
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.llm.provider is not None
        assert settings.postgres.url

    def test_a_blank_optional_enum_reads_as_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`FOO=` means "not set". The regression that motivated the class above.

        Asserted directly as well as through the template, because the template
        could stop leaving these blank and quietly take the coverage with it.
        """
        for key, value in self._template_environment().items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("LLM_EFFORT", "")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "")

        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.llm.effort is None
        assert settings.embedding.provider is None

    def test_a_genuinely_wrong_enum_value_is_still_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blank is unset; a typo is an error. Coercing both would hide real mistakes."""
        for key, value in self._template_environment().items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("LLM_EFFORT", "aggressive")

        with pytest.raises(Exception, match="LLM_EFFORT"):
            Settings(_env_file=None)  # type: ignore[call-arg]


class TestEnvExampleIsInSync:
    def test_no_duplicate_keys(self) -> None:
        declared = _declared_in_env_example()
        duplicates = {k for k in declared if declared.count(k) > 1}
        assert not duplicates, f"duplicated in .env.example: {sorted(duplicates)}"

    def test_every_setting_is_documented(self) -> None:
        declared = set(_declared_in_env_example())
        undocumented = {
            name: group
            for name, group in _expected_env_names().items()
            if name not in declared
        }
        assert not undocumented, (
            "settings read from the environment but missing from .env.example: "
            f"{undocumented}"
        )

    def test_every_documented_variable_is_used(self) -> None:
        expected = set(_expected_env_names())
        orphans = sorted(
            name
            for name in _declared_in_env_example()
            if name not in expected and not name.startswith(CREDENTIAL_PREFIXES)
        )
        assert not orphans, (
            "documented in .env.example but read by nothing in config.py "
            f"(rename or remove): {orphans}"
        )

    def test_choice_fields_document_their_options(self) -> None:
        """Every enum-typed setting must list its alternatives in the example."""
        text = ENV_EXAMPLE.read_text()
        missing: list[str] = []
        for field in Settings.model_fields.values():
            group = field.default_factory
            if not (isinstance(group, type) and issubclass(group, BaseSettings)):
                continue
            prefix = group.model_config.get("env_prefix", "")
            for name, sub in group.model_fields.items():
                annotation = sub.annotation
                if not (isinstance(annotation, type) and issubclass(annotation, str)):
                    continue
                if not hasattr(annotation, "__members__"):
                    continue
                alias = sub.alias or sub.validation_alias
                key = (alias if isinstance(alias, str) else f"{prefix}{name}").upper()
                block = text.split(f"\n{key}=")[0][-700:]
                if "options:" not in block:
                    missing.append(key)
        assert not missing, f"enum settings with no documented options: {missing}"


class TestDefaults:
    def test_local_defaults_load(self, clean_env: None) -> None:
        s = _settings()
        assert s.app.environment is Environment.LOCAL
        assert s.postgres.url.startswith("postgresql+asyncpg://")
        assert s.qdrant.collection == "omnisense_signals"
        assert s.llm.model_planner == "claude-opus-5"

    def test_cors_origins_are_split(self, clean_env: None, monkeypatch) -> None:
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://a.test, http://b.test ,")
        assert _settings().app.cors_origin_list == ["http://a.test", "http://b.test"]

    def test_get_settings_is_cached(self) -> None:
        assert get_settings() is get_settings()


class TestFailFastValidation:
    def test_rejects_sync_postgres_driver(self, clean_env: None, monkeypatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
        with pytest.raises(ValueError, match="async driver"):
            _settings()

    def test_rejects_chunk_overlap_at_or_above_chunk_size(
        self, clean_env: None, monkeypatch
    ) -> None:
        monkeypatch.setenv("EMBEDDING_MAX_CHARS_PER_CHUNK", "500")
        monkeypatch.setenv("EMBEDDING_CHUNK_OVERLAP_CHARS", "500")
        with pytest.raises(ValueError, match="forward progress"):
            _settings()

    def test_rejects_rerank_depth_below_top_k(self, clean_env: None, monkeypatch) -> None:
        monkeypatch.setenv("RETRIEVAL_RERANK_DEPTH", "5")
        monkeypatch.setenv("RETRIEVAL_FINAL_TOP_K", "20")
        with pytest.raises(ValueError, match="never see enough candidates"):
            _settings()

    def test_rejects_unknown_enum_value(self, clean_env: None, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "gpt5")
        with pytest.raises(ValueError):
            _settings()


class TestProductionSafety:
    @pytest.fixture
    def prod_base(self, clean_env: None, monkeypatch) -> pytest.MonkeyPatch:
        monkeypatch.setenv("OMNISENSE_ENV", "prod")
        monkeypatch.setenv("LOG_FORMAT", "json")
        return monkeypatch

    def test_rejects_placeholder_secrets(self, prod_base) -> None:
        with pytest.raises(ValueError, match="SECRET_KEY is still the placeholder"):
            _settings()

    def test_rejects_malformed_fernet_key(self, prod_base) -> None:
        prod_base.setenv("SECRET_KEY", "s" * 48)
        prod_base.setenv("CREDENTIAL_ENCRYPTION_KEY", "too-short")
        with pytest.raises(ValueError, match="valid Fernet key"):
            _settings()

    def test_rejects_wildcard_cors(self, prod_base) -> None:
        prod_base.setenv("SECRET_KEY", "s" * 48)
        prod_base.setenv("CREDENTIAL_ENCRYPTION_KEY", "a" * 43 + "=")
        prod_base.setenv("CORS_ALLOWED_ORIGINS", "*")
        with pytest.raises(ValueError, match=r"CORS_ALLOWED_ORIGINS contains"):
            _settings()

    def test_rejects_sql_echo(self, prod_base) -> None:
        prod_base.setenv("SECRET_KEY", "s" * 48)
        prod_base.setenv("CREDENTIAL_ENCRYPTION_KEY", "a" * 43 + "=")
        prod_base.setenv("POSTGRES_ECHO_SQL", "true")
        with pytest.raises(ValueError, match="ECHO_SQL"):
            _settings()

    def test_accepts_a_correct_production_config(self, prod_base) -> None:
        prod_base.setenv("SECRET_KEY", "s" * 48)
        prod_base.setenv("CREDENTIAL_ENCRYPTION_KEY", "a" * 43 + "=")
        prod_base.setenv("CORS_ALLOWED_ORIGINS", "https://app.example.com")
        s = _settings()
        assert s.app.environment.is_production_like
        assert s.app.log_format is LogFormat.JSON
        assert s.security.fernet_key_is_wellformed()

    def test_local_is_exempt(self, clean_env: None, monkeypatch) -> None:
        """Development defaults are convenient on purpose."""
        monkeypatch.setenv("OMNISENSE_ENV", "local")
        assert _settings().app.environment is Environment.LOCAL


class TestSecretsAreMasked:
    def test_describe_masks_every_secret(self, clean_env: None, monkeypatch) -> None:
        monkeypatch.setenv("POSTGRES_PASSWORD", "hunter2")
        monkeypatch.setenv("SECRET_KEY", "topsecret")
        described = str(_settings().describe())
        assert "hunter2" not in described
        assert "topsecret" not in described
        assert "**********" in described
