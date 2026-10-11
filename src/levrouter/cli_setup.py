"""Interactive setup wizard for levrouter configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from .config import DEFAULT_TIERS, TIERS, Backend
from .providers import OPENROUTER_FAMILIES, PROVIDERS
from .setup_check import _is_lev_installed, show_lev_installation_guide


def _prompt_choice(question: str, options: list[str]) -> str:
    """Prompt user to select from a list of options."""
    print(f"\n{question}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(options)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
            print(f"Invalid choice. Enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")


def _prompt_yes_no(question: str, default: bool = False) -> bool:
    """Prompt user for yes/no answer."""
    default_str = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{question} [{default_str}]: ").strip().lower()
        if answer == "":
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'")


def _prompt_input(prompt: str, default: str = "") -> str:
    """Prompt user for text input."""
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "
    answer = input(full_prompt).strip()
    return answer if answer else default


def _prompt_int(prompt: str, default: int) -> int:
    """Prompt for a whole number >= 0."""
    while True:
        answer = _prompt_input(prompt, default=str(default))
        if answer.isdigit():
            return int(answer)
        print("Please enter a whole number (0 or more)")


def setup_interactive() -> dict | None:
    """
    Interactive setup wizard for levrouter configuration.

    Returns a dict that can be saved as levrouter.yaml, or None if cancelled.
    """
    print("\n" + "=" * 80)
    print("levrouter Interactive Setup Wizard")
    print("=" * 80)
    print("\nWelcome! This wizard will help you set up levrouter.\n")

    # Step 1: Backend selection
    print("STEP 1: Choose your routing backend")
    print("-" * 80)
    print("""
        LEV-local:  In-process 4B model, 89% accurate (downloads 9.2GB on first run)
        LEV-http:   Cloud-hosted, 89% accurate (requires internet)
        Heuristic:  Rule-based, <1ms, 75% accurate (no downloads needed)
        """)

    backend = _prompt_choice(
        "Which backend would you like to use?", ["lev-local", "lev-http", "heuristic"]
    )

    # Step 2: Check LEV prerequisites if lev-local
    if backend == "lev-local":
        print("\nSTEP 2: Checking LEV prerequisites...")
        print("-" * 80)

        if not _is_lev_installed():
            print("\n❌ LEV is required for lev-local backend but is not installed.")
            show_lev_installation_guide()
            print("\nPlease install LEV first, then run 'levrouter setup' again.")
            return None
        else:
            print("✅ LEV is installed and ready!")

    # Step 3: Provider selection
    print("\n\nSTEP 3: Choose your primary provider")
    print("-" * 80)
    print("""
Direct Providers: Use your own API keys for each provider
  - anthropic, openai, gemini, groq, deepseek, mistral, xai, huggingface

OpenRouter: Single API key to access multiple providers
  - Acts as a gateway/proxy
  - Access 200+ models from multiple providers
  - Requires OpenRouter API key
""")
    
    provider_list = list(PROVIDERS.keys()) + ["openrouter"]
    print(f"Available providers: {', '.join(provider_list)}\n")

    provider = _prompt_choice("Which provider would you like to use?", provider_list)
    
    # Special handling for OpenRouter
    if provider == "openrouter":
        print("\n⚠️  OpenRouter Gateway Selected")
        print("You can access 200+ models from multiple providers via OpenRouter.")
        print("Set OPENROUTER_API_KEY environment variable before running.")
        openrouter_provider = _prompt_choice(
            "\nWhich vendor's models would you like (for default tiers)?",
            list(OPENROUTER_FAMILIES.keys())
        )
        # OpenRouter ids already carry the vendor: "anthropic/claude-haiku-4.5".
        defaults = OPENROUTER_FAMILIES[openrouter_provider]
        prefix = "openrouter/"

        print("""
Ladder mode lets levrouter switch vendors: for each request it picks the cheapest
capable model among the vendors you allow (e.g. Opus -> an OpenAI model for a simple
request). Prompts may then go to a different vendor than the one in your code.
""")
        ladder_vendors = None
        if _prompt_yes_no("Let levrouter switch vendors to the cheapest capable model (ladder mode)?"):
            known = list(OPENROUTER_FAMILIES)
            raw = _prompt_input("Vendors it may use (comma-separated)", default=", ".join(known))
            chosen = [v.strip() for v in raw.split(",") if v.strip()]
            unknown = [v for v in chosen if v not in OPENROUTER_FAMILIES]
            if unknown:
                print(f"   Ignoring unknown vendors: {', '.join(unknown)} (known: {', '.join(known)})")
            ladder_vendors = [v for v in chosen if v in OPENROUTER_FAMILIES] or known
    else:
        openrouter_provider = None
        ladder_vendors = None
        defaults = DEFAULT_TIERS[provider]
        prefix = f"{provider}/"

    # Step 4: Customize model tiers (optional)
    config_dict = {
        "enabled": True,
        "router": {
            "backend": backend,
            "timeout_s": 2.0,
        },
        "models": {},
    }
    if ladder_vendors:
        config_dict["router"]["openrouter_mode"] = "ladder"
        config_dict["router"]["openrouter_vendors"] = ladder_vendors
    if backend == "lev-local":
        # Load lev during load_config() so the first calls are not answered by the fallback.
        config_dict["router"]["block_on_load"] = True
    elif backend == "lev-http":
        from .classifiers.lev_http import _is_private
        from .providers import host_of

        url = _prompt_input("\nURL of your `lev serve` server", default="http://localhost:8000")
        config_dict["router"]["lev_url"] = url
        if not _is_private(host_of(url) or ""):
            print("   That server is outside your private network, so request text is sent over the internet.")
            print("   levrouter requires https for it, and reads the bearer key from LEVROUTER_LEV_API_KEY.")
            print("   If it needs other headers (Modal, Cloudflare Access, a gateway), map each header to the NAME of an")
            print("   environment variable under router.lev_headers, for example {Modal-Key: MODAL_KEY}.")
            config_dict["router"]["allow_remote_lev"] = True

    print("\n\nSTEP 4: Model tier configuration")
    print("-" * 80)
    print("""
Tiers determine which model to use:
  - simple: Quick, cheaper queries (e.g., summarization, classification)
  - standard: Balanced accuracy/cost (e.g., content generation)
  - complex: Advanced reasoning (e.g., analysis, coding)

If "No": Uses recommended defaults for your provider
If "Yes": You'll customize each tier's model selection
""")

    tiers_dict = dict(defaults)
    if _prompt_yes_no("\nCustomize model tiers for your provider?"):
        print(f"\nDefault models for {openrouter_provider or provider}:")
        for tier in TIERS:
            print(f"  {tier}: {defaults[tier]}")
        for tier in TIERS:
            tiers_dict[tier] = _prompt_input(f"Model for {tier} tier", default=defaults[tier])

    for tier in TIERS:
        config_dict["models"][tier] = {"providers": [prefix + tiers_dict[tier]]}

    # Step 5: Optional features
    print("\n\nSTEP 5: Optional features")
    print("-" * 80)

    if _prompt_yes_no(
        "\nEnable usage tracking (token counts and costs)?", default=True
    ):
        config_dict["router"]["usage_file"] = True
        config_dict["router"]["usage_live"] = True

    if _prompt_yes_no("Enable detailed logging?"):
        config_dict["router"]["log_level"] = "DEBUG"
    else:
        config_dict["router"]["log_level"] = "INFO"

    # Step 6: Optional advanced settings
    print("\n\nSTEP 6: Advanced settings (frameworks and conversations)")
    print("-" * 80)
    print("""
Frameworks: which libraries to route. LangChain is routed by default; add anthropic / openai
to also route direct Anthropic SDK / OpenAI SDK calls.
Conversations: keep each conversation on one model (so prompt caching keeps working), and move it
up a tier when the user asks the same question again.
""")
    if _prompt_yes_no("Configure advanced settings?"):
        known = ("langchain", "anthropic", "openai")
        raw = _prompt_input("Frameworks to route (comma-separated: langchain, anthropic, openai)", default="langchain")
        chosen = [f.strip() for f in raw.split(",") if f.strip()]
        unknown = [f for f in chosen if f not in known]
        if unknown:
            print(f"   Ignoring unknown frameworks: {', '.join(unknown)}")
        chosen = [f for f in chosen if f in known] or ["langchain"]
        if chosen != ["langchain"]:
            config_dict = {"enabled": True, "frameworks": chosen,
                           **{k: v for k, v in config_dict.items() if k != "enabled"}}
        if not _prompt_yes_no("Keep each conversation on one model (sticky_threads)?", default=True):
            config_dict["router"]["sticky_threads"] = False
        repeats = _prompt_int("Move up a tier after how many re-asked questions? (0 = off)", 2)
        if repeats != 2:
            config_dict["router"]["escalate_after_repeats"] = repeats
        ttl = _prompt_int("Seconds a quiet conversation keeps its model", 300)
        if ttl != 300:
            config_dict["router"]["thread_ttl_s"] = ttl

    # Step 7: Review and save
    print("\n\nSTEP 7: Review configuration")
    print("-" * 80)
    print("\nYour configuration:")
    print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))

    if not _prompt_yes_no("Save this configuration?", default=True):
        print("\nSetup cancelled.")
        return None

    return config_dict


def write_config_file(config_dict: dict, path: Path | str = "levrouter.yaml") -> bool:
    """
    Write configuration to a YAML file.

    Args:
        config_dict: Configuration dictionary
        path: Path to save the file (default: levrouter.yaml in current directory)

    Returns:
        True if successful, False otherwise
    """
    path = Path(path)

    try:
        # Check if file already exists
        if path.exists():
            print(f"\n⚠️  File {path} already exists.")
            if not _prompt_yes_no("Overwrite?"):
                print("File not saved.")
                return False

        # Write the file
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        print(f"\n✅ Configuration saved to {path.absolute()}")
        return True
    except Exception as e:
        print(f"\n❌ Error saving configuration: {e}", file=sys.stderr)
        return False


def setup_cmd() -> int:
    """
    Main setup command handler.

    Returns:
        0 if successful, 1 if cancelled/failed
    """
    print("\nlevrouter setup wizard")
    print("=" * 80)

    config_dict = setup_interactive()

    if config_dict is None:
        return 1

    # Save to levrouter.yaml in current directory
    if write_config_file(config_dict, "levrouter.yaml"):
        print("\n" + "=" * 80)
        print("Setup complete! 🎉")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Review the generated levrouter.yaml file")
        print("  2. Add to your code entry point:")
        print("     import levrouter")
        print("     levrouter.load_config()")
        print("  3. Run your application - routing will start automatically!")
        print("\nFor help: https://github.com/sathyalog/levrouter")
        return 0
    else:
        return 1
