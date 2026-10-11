"""Setup validation for levrouter prerequisites."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _is_lev_installed() -> bool:
    """Check if lev package is installed."""
    return importlib.util.find_spec("lev") is not None


def _get_project_root() -> Path | None:
    """Find project root by looking for pyproject.toml or requirements.txt."""
    current = Path.cwd()
    for _ in range(10):  # Search up to 10 levels
        if (current / "pyproject.toml").exists() or (current / "requirements.txt").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def _check_requirements_file(project_root: Path) -> bool:
    """Check if lev is listed in requirements.txt."""
    req_file = project_root / "requirements.txt"
    if not req_file.exists():
        return False

    try:
        with open(req_file) as f:
            content = f.read()
            return "lev" in content.lower()
    except Exception:
        return False


def _check_pyproject_toml(project_root: Path) -> bool:
    """Check if lev is listed in pyproject.toml dependencies."""
    pyproject = project_root / "pyproject.toml"
    if not pyproject.exists():
        return False

    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return False

    try:
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

            # Check dependencies
            deps = data.get("project", {}).get("dependencies", [])
            if any("lev" in str(dep).lower() for dep in deps):
                return True

            # Check optional dependencies
            optional_deps = data.get("project", {}).get("optional-dependencies", {})
            all_optional = []
            for dep_list in optional_deps.values():
                all_optional.extend(dep_list)

            if any("lev" in str(dep).lower() for dep in all_optional):
                return True

            return False
    except Exception:
        return False


def _check_pip_list() -> bool:
    """Check if lev is in pip list."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False

        packages = json.loads(result.stdout)
        return any(pkg["name"].lower() == "lev" for pkg in packages)
    except Exception:
        return False


def show_lev_installation_guide() -> None:
    """Display LEV installation instructions."""
    guide = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    ⚠️  LEV PREREQUISITE NOT FOUND                          ║
╚════════════════════════════════════════════════════════════════════════════╝

levrouter requires LEV, a 4B decision model from InterfazeAI that classifies
LLM requests by complexity (simple/standard/complex) with 89% accuracy.

LEV is NOT available on PyPI - it must be installed separately.

════════════════════════════════════════════════════════════════════════════

INSTALLATION OPTIONS:

Option 1: Install LEV from GitHub (Recommended)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For pip:
  pip install 'lev[serve] @ git+https://github.com/InterfazeAI/lev#subdirectory=packages/lev'

For uv:
  uv add 'lev[serve] @ git+https://github.com/InterfazeAI/lev#subdirectory=packages/lev'

For poetry:
  poetry add 'lev[serve] @ {git = "https://github.com/InterfazeAI/lev", subdirectory = "packages/lev"}'

Requirements:
  - Python 3.12+
  - ~9.2GB disk space (for model cache)
  - Git installed
  - (Optional) CUDA GPU for faster inference


Option 2: Use Heuristic Backend (No LEV needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If you don't want to install LEV, you can use the heuristic backend:

Update your levrouter.yaml:
  router:
    backend: "heuristic"  # Rule-based, <1ms, 75% accurate

Benefits:
  ✅ No downloads needed
  ✅ Works immediately (<1ms per request)
  ✅ 75% accuracy (good enough for most cases)
  ✅ No GPU needed

Then run:
  python your_script.py


Option 3: Use Remote LEV (Cloud API)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use InterfazeAI's hosted LEV service (no download needed):

Update your levrouter.yaml:
  router:
    backend: "lev-http"
    lev_url: "https://api.interfaze.ai/lev"
    lev_api_key: ${LEV_API_KEY}

Then:
  export LEV_API_KEY="your-api-key-from-interfaze"
  python your_script.py

Contact InterfazeAI for API access: https://www.interfaze.ai

════════════════════════════════════════════════════════════════════════════

QUICK START:

Step 1: Run the interactive setup wizard

  levrouter setup

  This will guide you through:
    - Choosing your backend (lev-local, heuristic, or lev-http)
    - Selecting your LLM provider (Anthropic, OpenAI, etc.)
    - Customizing model tiers (optional)
    - Automatic levrouter.yaml generation

Step 2: Add to your code

  import levrouter
  levrouter.load_config()

  # Your LLM calls are now routed automatically!

Step 3: (Optional) Install LEV for lev-local backend

If using requirements.txt:
  echo "lev[serve] @ git+https://github.com/InterfazeAI/lev#subdirectory=packages/lev" >> requirements.txt
  pip install -r requirements.txt

If using pyproject.toml:
  Add to dependencies:
    "lev[serve] @ git+https://github.com/InterfazeAI/lev#subdirectory=packages/lev"

════════════════════════════════════════════════════════════════════════════

TROUBLESHOOTING:

Q: "ERROR: failed building wheel for lev"
A: This is a known issue with the GitHub repo. Options:
   - Use heuristic backend (no build needed)
   - Use lev-http backend (remote, no build needed)
   - Contact InterfazeAI for pre-built binaries
   - Wait for PyPI release

Q: "ImportError: No module named 'lev'"
A: LEV is not installed. Follow the installation steps above.

Q: "LEV model download is very slow"
A: This is normal for 9.2GB. Consider:
   - Using a GPU for faster inference
   - Using heuristic backend for development
   - Using lev-http for production

Q: "Can I use levrouter without LEV?"
A: Yes! Set backend: "heuristic" in levrouter.yaml
   It gives 75% accuracy with <1ms latency.

════════════════════════════════════════════════════════════════════════════

NEXT STEPS:

1. Run: levrouter setup
2. Answer the interactive prompts (backend, provider, tiers)
3. levrouter.yaml is created automatically
4. Add import levrouter; levrouter.load_config() to your code
5. (Optional) Install LEV if using lev-local backend
6. Your LLM calls will now be routed automatically!

════════════════════════════════════════════════════════════════════════════

For more information:
  📖 levrouter: https://github.com/sathyalog/levrouter
  📖 LEV: https://github.com/InterfazeAI/lev
  💬 Support: Check GitHub issues at https://github.com/sathyalog/levrouter/issues

════════════════════════════════════════════════════════════════════════════
"""
    print(guide)


def check_lev_prerequisites(skip_check: bool = False) -> bool:
    """
    Check if LEV is installed and show installation guide if not.

    Args:
        skip_check: If True, skip the check (useful for testing)

    Returns:
        True if LEV is installed or check is skipped, False otherwise
    """
    if skip_check:
        return True

    # Check if lev is directly importable
    if _is_lev_installed():
        return True

    # Check if lev is in project dependencies
    project_root = _get_project_root()
    if project_root:
        if _check_requirements_file(project_root) or _check_pyproject_toml(project_root):
            # Listed in dependencies but not installed yet
            print(
                "⚠️  LEV is listed in dependencies but not yet installed.\n"
                "   Run: pip install -r requirements.txt  (or similar)\n"
            )
            return True

    # Check if lev is in pip list
    if _check_pip_list():
        return True

    # LEV not found - show guide
    show_lev_installation_guide()
    return False


def validate_setup() -> None:
    """Validate setup on import - runs once per process."""
    # Only check once per process
    if not hasattr(validate_setup, "_checked"):
        validate_setup._checked = True

        # Skip check in certain environments
        if not check_lev_prerequisites():
            # LEV not found, but don't fail immediately
            # User can still use heuristic backend
            pass
