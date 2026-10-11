import pytest

import levrouter
from levrouter.classifiers.base import Signals
from levrouter.features import RequestFeatures


class FixedClassifier:
    """Returns whatever Signals the test sets; counts calls."""

    name = "fixed"

    def __init__(self, signals: Signals | None = None, error: Exception | None = None):
        self.signals = signals
        self.error = error
        self.calls = 0

    def classify(self, features: RequestFeatures, state: str) -> Signals:
        self.calls += 1
        if self.error:
            raise self.error
        return self.signals


def signals(simple=0.1, standard=0.2, complex=0.7, reasoning=0.5, code=0.0, source="fixed") -> Signals:
    return Signals({"simple": simple, "standard": standard, "complex": complex}, reasoning, code, source)


SIMPLE = signals(0.9, 0.08, 0.02, 0.05, 0.0)
STANDARD = signals(0.2, 0.7, 0.1, 0.2, 0.1)
COMPLEX = signals(0.05, 0.15, 0.8, 0.9, 0.6)


@pytest.fixture(autouse=True)
def _clean(tmp_path_factory, monkeypatch):
    # Keep usage files and live sockets out of the developer's real ~/.levrouter.
    monkeypatch.setenv("LEVROUTER_HOME", str(tmp_path_factory.mktemp("levrouter_home")))
    yield
    levrouter.disable()
    from levrouter import _usage_warned
    from levrouter.config_loader import _warned_env

    _warned_env.clear()
    _usage_warned.clear()
    from levrouter.integrations._sdk_common import _failed

    _failed.clear()
