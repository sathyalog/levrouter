"""A question asked again and again moves the conversation to a better model."""

from conftest import SIMPLE, FixedClassifier
from langchain_core.messages import AIMessage, HumanMessage

import levrouter
from levrouter import Router, RouterConfig
from levrouter.converters.anthropic import extract as extract_anthropic
from levrouter.features import RequestFeatures, extract
from levrouter.repeats import repeat_count

Q1 = "How do I configure retries in the payment client?"
Q2 = "How do I set up retries for the payment client?"
Q3 = "That's wrong. How do I configure payment client retries?"
Q4 = "Still wrong: configure retries in the payment client please"


def _features(last, *earlier):
    return RequestFeatures("", last, 0, 0, 0, earlier_user=earlier)


def _chat(*questions):
    msgs = []
    for q in questions:
        msgs += [HumanMessage(q), AIMessage("an answer")]
    return extract(msgs[:-1])


def test_repeat_detection():
    assert repeat_count(_features(Q2, Q1)) == 1
    assert repeat_count(_features(Q3, Q1, Q2)) == 2
    assert repeat_count(_features("thanks", "thanks", "thanks")) == 0
    assert repeat_count(_features("What is the refund policy for EU customers?", Q1, Q2)) == 0
    assert repeat_count(_features("that's wrong, try again", "Explain the invoice totals")) == 1
    assert repeat_count(_features(Q1)) == 0


def test_converters_collect_earlier_user_messages():
    assert _chat("a b c", "d e f", "g h i").earlier_user == ("a b c", "d e f")
    msgs = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "x"},
            {"role": "user", "content": [{"type": "text", "text": "second"}]}]
    assert extract_anthropic(msgs).earlier_user == ("first",)


def _decide(router, features, thread=None, override=None):
    return router.decide("anthropic", "claude-sonnet-5", features, override, thread=thread)


def test_third_ask_moves_up_and_further_repeats_move_up_again():
    router = Router(RouterConfig(), FixedClassifier(SIMPLE))
    assert _decide(router, _chat(Q1)).model == "claude-haiku-4-5"
    assert _decide(router, _chat(Q1, Q2)).model == "claude-haiku-4-5"
    third = _decide(router, _chat(Q1, Q2, Q3))
    assert third.model == "claude-sonnet-5" and "asked 3 times: moved up to standard" in third.reason
    assert _decide(router, _chat(Q1, Q2, Q3, Q4)).model == "claude-opus-5-5"


def test_escalation_can_be_turned_off_and_overrides_win():
    off = Router(RouterConfig(escalate_after_repeats=0), FixedClassifier(SIMPLE))
    assert _decide(off, _chat(Q1, Q2, Q3)).model == "claude-haiku-4-5"
    on = Router(RouterConfig(), FixedClassifier(SIMPLE))
    assert _decide(on, _chat(Q1, Q2, Q3), override="simple").model == "claude-haiku-4-5"


def test_thread_keeps_the_better_model_until_reset():
    router = Router(RouterConfig(), FixedClassifier(SIMPLE))
    levrouter.enable(router=router)
    assert _decide(router, _chat(Q1, Q2, Q3), thread="c1").model == "claude-sonnet-5"
    assert _decide(router, _chat(Q1, Q2, Q3, "thanks"), thread="c1").model == "claude-sonnet-5"
    levrouter.reset_thread("c1")
    assert _decide(router, _chat(Q1, Q2, Q3, "thanks"), thread="c1").model == "claude-haiku-4-5"
