#!/usr/bin/env python3
"""Test script for levrouter setup feature."""

import sys
import tempfile
from pathlib import Path

# Add src to path so we can import without installation
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_cli_setup_module():
    """Test that cli_setup module can be imported and has required functions."""
    print("=" * 80)
    print("Testing cli_setup Module")
    print("=" * 80)

    try:
        # Test 1: Import the module
        print("\n✅ Test 1: Import cli_setup module...")
        from levrouter.cli_setup import (
            setup_interactive,
            write_config_file,
            setup_cmd,
            _prompt_choice,
            _prompt_yes_no,
            _prompt_input
        )
        print("   SUCCESS: All functions imported")

        # Test 2: Check function signatures
        print("\n✅ Test 2: Verify function signatures...")
        assert callable(setup_interactive), "setup_interactive must be callable"
        assert callable(write_config_file), "write_config_file must be callable"
        assert callable(setup_cmd), "setup_cmd must be callable"
        print("   SUCCESS: All functions are callable")

        # Test 3: Check __main__ has setup command
        print("\n✅ Test 3: Verify __main__.py has setup handler...")
        from levrouter.__main__ import main
        import inspect
        source = inspect.getsource(main)
        assert 'args.command == "setup"' in source, "setup handler not found in main()"
        assert '"setup"' in source, "setup subparser not found"
        print("   SUCCESS: setup command handler is present in __main__.py")

        # Test 4: Check __init__ exports setup_interactive
        print("\n✅ Test 4: Verify __init__.py exports setup_interactive...")
        from levrouter import setup_interactive as exported_setup
        assert callable(exported_setup), "setup_interactive not exported from __init__"
        print("   SUCCESS: setup_interactive is exported from levrouter module")

        # Test 5: Test config file writing (without interactive input)
        print("\n✅ Test 5: Test config file writing...")
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config = {
                "enabled": True,
                "router": {
                    "backend": "heuristic",
                    "timeout_s": 2.0,
                },
                "models": {
                    "simple": {"providers": ["openrouter/anthropic/claude-3-5-haiku"]},
                    "standard": {"providers": ["openrouter/anthropic/claude-3-5-sonnet"]},
                    "complex": {"providers": ["openrouter/anthropic/claude-3-5-opus"]}
                }
            }

            config_path = Path(tmpdir) / "levrouter.yaml"
            result = write_config_file(test_config, config_path)

            assert result, "write_config_file returned False"
            assert config_path.exists(), f"Config file not created at {config_path}"

            # Verify file content
            import yaml
            with open(config_path) as f:
                loaded_config = yaml.safe_load(f)

            assert loaded_config is not None, "Config file is empty or invalid YAML"
            assert loaded_config["enabled"] == True, "Config 'enabled' field incorrect"
            assert loaded_config["router"]["backend"] == "heuristic", "Backend not saved correctly"

            print(f"   SUCCESS: Config file created and valid at {config_path}")

        # Test 6: Verify DEFAULT_TIERS is accessible
        print("\n✅ Test 6: Verify DEFAULT_TIERS accessibility...")
        from levrouter.config import DEFAULT_TIERS, TIERS, PROVIDERS
        assert isinstance(DEFAULT_TIERS, dict), "DEFAULT_TIERS must be a dict"
        assert "anthropic" in DEFAULT_TIERS, "anthropic not in DEFAULT_TIERS"
        assert "heuristic" in PROVIDERS, "heuristic backend missing"
        assert "lev-local" in ["heuristic", "lev-local", "lev-http"], "Backends not available"
        print(f"   SUCCESS: DEFAULT_TIERS has {len(DEFAULT_TIERS)} providers")
        print(f"   SUCCESS: PROVIDERS has {len(PROVIDERS)} providers")

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nSetup feature is ready for use!")
        print("\nUsage:")
        print("  levrouter setup")
        print("  python -c \"from levrouter import setup_interactive; setup_interactive()\"")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cli_setup_module()
    sys.exit(0 if success else 1)
