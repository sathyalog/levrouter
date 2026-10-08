"""load_config must honour the YAML (router settings and models), and lev_download must report real progress."""

import pytest
import yaml

import tokentriage
from tokentriage import RouterConfig, cli_setup, config_loader, lev_download
from tokentriage.__main__ import main
from tokentriage.config_loader import find_config_file, load_config, resolve_config
from tokentriage.integrations import langchain as lc

YAML = """
enabled: true
router:
  backend: heuristic
  timeout_s: 60.0
  block_on_load: true
models:
  simple:
    providers: [anthropic/claude-haiku-4-5-20251001]
  complex:
    providers: [openrouter/anthropic/claude-opus-5.5, not-a-provider/x]
features:
  track_usage: true
"""


@pytest.fixture(autouse=True)
def _fresh_warning_state(monkeypatch):
    monkeypatch.setattr(config_loader, "_warned_env", set())


def _load(tmp_path, text, **kwargs):
    path = tmp_path / "tokentriage.yaml"
    path.write_text(text)
    return load_config(str(path), **kwargs)[0]


def test_router_settings_and_models_are_applied(tmp_path, capsys):
    cfg = _load(tmp_path, YAML, auto_enable=False)
    assert cfg.backend == "heuristic" and cfg.timeout_s == 60.0 and cfg.block_on_load is True
    assert cfg.tiers["anthropic"]["simple"] == "claude-haiku-4-5-20251001"
    assert cfg.tiers["anthropic"]["standard"] == "claude-sonnet-5"  # untouched tiers keep defaults
    assert cfg.openrouter_families["anthropic"]["complex"] == "anthropic/claude-opus-5.5"
    out = capsys.readouterr().err
    assert "'not-a-provider/x' ignored" in out and "ignoring config keys" in out and "features" in out


def test_enabled_false_does_not_route(tmp_path, capsys):
    _load(tmp_path, "enabled: false\nrouter:\n  backend: heuristic\n")
    assert lc._State.router is None
    assert "disabled" in capsys.readouterr().err


def test_setup_wizard_output_loads(tmp_path, monkeypatch):
    # heuristic backend, OpenRouter, anthropic vendor, no ladder, customise tiers but keep defaults, usage on,
    # no debug, save
    vendor = list(cli_setup.OPENROUTER_FAMILIES).index("anthropic") + 1
    answers = iter(["3", str(len(cli_setup.PROVIDERS) + 1), str(vendor), "n", "y", "", "", "", "y", "n", "n", "y"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    config = cli_setup.setup_interactive()
    cfg = _load(tmp_path, yaml.dump(config), auto_enable=False)
    assert cfg.openrouter_families["anthropic"]["simple"] == "anthropic/claude-haiku-4.5"


def _wizard(monkeypatch, answers):
    monkeypatch.setattr(cli_setup, "_is_lev_installed", lambda: True)
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *_: next(it))
    return cli_setup.setup_interactive()


def test_setup_wizard_lev_local_preloads(monkeypatch):
    anthropic = str(list(cli_setup.PROVIDERS).index("anthropic") + 1)
    config = _wizard(monkeypatch, ["1", anthropic, "n", "y", "n", "n", "y"])
    assert config["router"]["backend"] == "lev-local" and config["router"]["block_on_load"] is True


def test_setup_wizard_lev_http_asks_for_url(tmp_path, monkeypatch):
    anthropic = str(list(cli_setup.PROVIDERS).index("anthropic") + 1)
    config = _wizard(monkeypatch, ["2", anthropic, "https://lev.example.com", "n", "y", "n", "n", "y"])
    cfg = _load(tmp_path, yaml.dump(config), auto_enable=False)
    assert cfg.backend == "lev-http" and cfg.lev_url == "https://lev.example.com" and cfg.allow_remote_lev is True


def test_setup_wizard_openrouter_ladder(tmp_path, monkeypatch):
    vendor = str(list(cli_setup.OPENROUTER_FAMILIES).index("anthropic") + 1)
    openrouter = str(len(cli_setup.PROVIDERS) + 1)
    config = _wizard(monkeypatch, ["3", openrouter, vendor, "y", "anthropic, openai, nope", "n", "y", "n", "n", "y"])
    cfg = _load(tmp_path, yaml.dump(config), auto_enable=False)
    assert cfg.openrouter_mode == "ladder" and cfg.openrouter_vendors == ("anthropic", "openai")


def test_progress_counts_only_byte_bars(capsys):
    progress = lev_download._Progress("base model", total=200)
    byte_bar = progress.bar_class()(unit="B", total=200)
    count_bar = progress.bar_class()(unit="it", total=3)
    count_bar.update(1)
    assert progress.done == 0
    byte_bar.update(50)
    assert progress.done == 50 and "25.0%" in capsys.readouterr().out
    byte_bar.update(150)
    assert progress.done == 200 and "100.0%" in capsys.readouterr().out


def test_failed_download_reports_false(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(lev_download, "_fetch", boom)
    assert lev_download.ensure_lev_cached("interfaze-ai/lev") is False
    assert "no network" in capsys.readouterr().out


# -- one settings source: the file, plus a few environment variables -----------------------------------------


def _file(tmp_path, text, name="tokentriage.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_tokentriage_config_variable_picks_the_file(tmp_path, monkeypatch):
    other = _file(tmp_path, "router:\n  backend: heuristic\n  timeout_s: 9\n", "prod.yaml")
    monkeypatch.setenv("TOKENTRIAGE_CONFIG", str(other))
    resolved = resolve_config()
    assert resolved.source == other and resolved.cfg.timeout_s == 9.0


def test_kept_variables_beat_the_file_and_the_file_beats_defaults(tmp_path, monkeypatch):
    path = _file(tmp_path, "router:\n  backend: heuristic\n  timeout_s: 7\n  log_level: INFO\n")
    monkeypatch.setenv("TOKENTRIAGE_BACKEND", "lev-http")
    monkeypatch.setenv("TOKENTRIAGE_LOG_LEVEL", "debug")
    monkeypatch.setenv("TOKENTRIAGE_MODE", "eval")
    cfg = resolve_config(path).cfg
    assert (cfg.backend, cfg.log_level, cfg.mode) == ("lev-http", "DEBUG", "eval")
    assert cfg.timeout_s == 7.0 and cfg.thread_ttl_s == 300  # file value, then the default


def test_invalid_override_values_are_ignored_with_a_warning(tmp_path, monkeypatch, capsys):
    path = _file(tmp_path, "router:\n  backend: heuristic\n")
    monkeypatch.setenv("TOKENTRIAGE_BACKEND", "banana")
    assert resolve_config(path).cfg.backend == "heuristic"
    assert "TOKENTRIAGE_BACKEND='banana' ignored" in capsys.readouterr().err


def test_removed_variables_are_ignored_and_warned_about(tmp_path, monkeypatch, capsys):
    path = _file(tmp_path, "router:\n  backend: heuristic\n")
    monkeypatch.setenv("TOKENTRIAGE_MIN_TIER", "standard")
    monkeypatch.setenv("TOKENTRIAGE_OPENAI_SIMPLE", "gpt-4o-mini")
    cfg = resolve_config(path).cfg
    assert cfg.min_tier is None and cfg.tiers["openai"]["simple"] != "gpt-4o-mini"
    out = capsys.readouterr().err
    assert "no longer used" in out and "TOKENTRIAGE_MIN_TIER" in out and "TOKENTRIAGE_OPENAI_SIMPLE" in out
    resolve_config(path)
    assert "no longer used" not in capsys.readouterr().err  # said once


def test_lev_api_key_in_a_file_is_dropped(tmp_path, capsys):
    path = _file(tmp_path, "router:\n  backend: lev-http\n  lev_api_key: s3cret-value\n")
    cfg = resolve_config(path).cfg
    out = capsys.readouterr().err
    assert cfg.lev_api_key is None and "TOKENTRIAGE_LEV_API_KEY" in out and "s3cret-value" not in out


def test_enabled_variable_is_a_kill_switch(tmp_path, monkeypatch):
    path = _file(tmp_path, "router:\n  backend: heuristic\n")
    monkeypatch.setenv("TOKENTRIAGE_ENABLED", "false")
    assert resolve_config(path).enabled is False
    load_config(str(path))
    assert lc._State.router is None
    assert tokentriage.enable() == [] and tokentriage.enable(RouterConfig(backend="heuristic")) == []
    assert lc._State.router is None
    monkeypatch.setenv("TOKENTRIAGE_ENABLED", "true")
    assert resolve_config(path).enabled is True


def test_enable_without_arguments_reads_the_yaml(tmp_path, monkeypatch):
    _file(tmp_path, "router:\n  backend: heuristic\n  escalate_after_repeats: 5\nframeworks: [langchain]\n")
    monkeypatch.chdir(tmp_path)
    assert "anthropic" in tokentriage.enable()
    assert lc._State.router.config.escalate_after_repeats == 5


def test_config_search_ignores_the_data_folder(tmp_path, monkeypatch):
    (tmp_path / ".tokentriage").mkdir()  # ~/.tokentriage is the data folder, not a config file
    monkeypatch.chdir(tmp_path)
    assert find_config_file(tmp_path) is None
    assert resolve_config().source is None


def test_ladder_vendors_come_from_the_yaml(tmp_path):
    path = _file(tmp_path, "router:\n  openrouter_mode: ladder\n  openrouter_vendors: [anthropic, google]\n")
    cfg = resolve_config(path).cfg
    assert cfg.openrouter_mode == "ladder" and cfg.openrouter_vendors == ("anthropic", "google")


def test_check_shows_the_yaml(tmp_path, monkeypatch, capsys):
    path = _file(tmp_path, "router:\n  backend: heuristic\n  timeout_s: 9.5\nframeworks: [langchain, openai]\n")
    monkeypatch.chdir(tmp_path)
    main(["check"])
    out = capsys.readouterr().out
    assert str(path) in out and "timeout_s=9.5" in out and "frameworks=langchain, openai" in out


def test_setup_wizard_advanced_step(tmp_path, monkeypatch):
    anthropic = str(list(cli_setup.PROVIDERS).index("anthropic") + 1)
    # heuristic, anthropic, default tiers, usage y, debug n, advanced y, frameworks, sticky n, repeats 4, ttl 3600, save
    config = _wizard(monkeypatch, ["3", anthropic, "n", "y", "n", "y", "langchain, openai, nope", "n", "4", "3600", "y"])
    assert config["frameworks"] == ["langchain", "openai"]
    assert config["router"]["sticky_threads"] is False
    assert config["router"]["escalate_after_repeats"] == 4 and config["router"]["thread_ttl_s"] == 3600
    cfg = _load(tmp_path, yaml.dump(config), auto_enable=False)
    assert cfg.thread_ttl_s == 3600 and cfg.sticky_threads is False


def test_setup_wizard_advanced_step_writes_only_changes(monkeypatch):
    anthropic = str(list(cli_setup.PROVIDERS).index("anthropic") + 1)
    config = _wizard(monkeypatch, ["3", anthropic, "n", "y", "n", "y", "", "y", "2", "300", "y"])
    assert "frameworks" not in config
    assert not {"sticky_threads", "escalate_after_repeats", "thread_ttl_s"} & set(config["router"])


def test_lev_headers_map_header_names_to_env_var_names(tmp_path, capsys):
    cfg = _load(tmp_path, "router:\n  backend: heuristic\n  lev_headers:\n    Modal-Key: MODAL_KEY\n    Modal-Secret: MODAL_SECRET\n",
                auto_enable=False)
    assert cfg.lev_headers == {"Modal-Key": "MODAL_KEY", "Modal-Secret": "MODAL_SECRET"}
    assert "ignored" not in capsys.readouterr().err


def test_bad_lev_headers_entries_are_dropped_with_a_warning(tmp_path, capsys):
    text = ("router:\n  backend: heuristic\n  lev_headers:\n    Good-Header: GOOD_VAR\n"
            "    'Bad Header': SOME_VAR\n    Another: 'not a var name!'\n    Third: 5\n")
    cfg = _load(tmp_path, text, auto_enable=False)
    err = capsys.readouterr().err
    assert cfg.lev_headers == {"Good-Header": "GOOD_VAR"}
    assert "'Bad Header' ignored" in err and "'Another' ignored" in err and "'Third' ignored" in err
    assert "NAME of an environment variable" in err


def test_lev_headers_that_is_not_a_mapping_is_ignored(tmp_path, capsys):
    cfg = _load(tmp_path, "router:\n  backend: heuristic\n  lev_headers: [a, b]\n", auto_enable=False)
    assert cfg.lev_headers == {} and "lev_headers ignored" in capsys.readouterr().err


def test_env_vars_named_under_lev_headers_are_not_reported_as_removed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TOKENTRIAGE_MODAL_KEY", "wk-x")
    monkeypatch.setenv("TOKENTRIAGE_SOMETHING_ELSE", "1")
    _load(tmp_path, "router:\n  backend: heuristic\n  lev_headers:\n    Modal-Key: TOKENTRIAGE_MODAL_KEY\n", auto_enable=False)
    err = capsys.readouterr().err
    assert "TOKENTRIAGE_SOMETHING_ELSE" in err and "TOKENTRIAGE_MODAL_KEY" not in err


def test_startup_messages_respect_log_level(tmp_path, capsys):
    text = "router:\n  backend: heuristic\n  log_level: ERROR\n  bogus_key: 1\n"
    _load(tmp_path, text, auto_enable=False)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_startup_warnings_appear_at_the_default_level_on_stderr_not_stdout(tmp_path, capsys):
    _load(tmp_path, "router:\n  backend: heuristic\n  bogus_key: 1\n", auto_enable=False)
    captured = capsys.readouterr()
    assert captured.out == "" and "bogus_key" in captured.err


def test_log_level_none_hands_startup_messages_to_the_apps_logging(tmp_path, capsys, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="tokentriage"):
        _load(tmp_path, "router:\n  backend: heuristic\n  log_level: null\n  bogus_key: 1\n", auto_enable=False)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "" and "bogus_key" in caplog.text


def test_warm_on_load_sends_one_background_request_only_when_enabled(tmp_path, monkeypatch):
    import httpx

    from tokentriage.classifiers import lev_http

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"answers": {}})

    real_client = httpx.Client
    monkeypatch.setattr(lev_http.LevHttpClassifier, "classify", lambda *a, **k: None)
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    base = "router:\n  backend: lev-http\n  lev_url: https://lev.example.com\n  allow_remote_lev: true\n"
    import threading

    _load(tmp_path, base, auto_enable=False)
    assert calls == []
    _load(tmp_path, base + "  warm_on_load: true\n", auto_enable=False)
    for t in threading.enumerate():
        if t.name == "tokentriage-warmup":
            t.join(5)
    assert calls == ["/v1/systemone"]


def test_a_failing_warm_up_never_raises_and_is_reported(tmp_path, monkeypatch, capsys):
    import threading

    import httpx

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        transport=httpx.MockTransport(lambda r: httpx.Response(503)), **kw))
    _load(tmp_path, "router:\n  backend: lev-http\n  lev_url: https://lev.example.com\n  allow_remote_lev: true\n  warm_on_load: true\n",
          auto_enable=False)
    for t in threading.enumerate():
        if t.name == "tokentriage-warmup":
            t.join(5)
    assert "warm-up failed" in capsys.readouterr().err
