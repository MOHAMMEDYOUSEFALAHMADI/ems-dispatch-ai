import json

import pytest

from ems_dispatch_ai.triage import TriageAssistant, normalize, rule_triage, validate


@pytest.mark.parametrize("text,priority,category", [
    ("Caller says her father collapsed and is not breathing", "P1", "cardiac_arrest"),
    ("أبوي طاح وما يتنفس", "P1", "cardiac_arrest"),
    ("رجل عمره ٦٠ سنة عنده ألم في الصدر وتعرق", "P1", "chest_pain"),
    ("Sudden face droop and slurred speech, 70 year old", "P1", "stroke"),
    ("حادث سيارة، المصاب واعي وعنده كسر في اليد", "P2", "trauma"),
    ("Pregnant woman, contractions every 5 minutes", "P2", "obstetric"),
    ("طفل عنده جرح بسيط في الإصبع", "P3", "minor"),
    ("Patient is breathing normally, mild ankle sprain", "P3", "minor"),
])
def test_rule_triage(text, priority, category):
    r = rule_triage(text)
    assert (r.priority, r.category) == (priority, category)


def test_language_and_normalisation():
    assert rule_triage("نزيف شديد من الساق").language == "ar"
    assert normalize("إغماءٌ") == "اغماء"


def test_no_keywords_defaults_low_confidence():
    r = rule_triage("caller hung up")
    assert r.priority == "P3" and r.confidence < 0.5


def test_validate_rejects_bad_json():
    with pytest.raises(ValueError):
        validate({"priority": "P0", "category": "trauma"})
    assert validate({"priority": "P2", "category": "trauma", "red_flags": [], "confidence": 3}).confidence == 1.0


def test_default_backend_is_mock(monkeypatch):
    monkeypatch.delenv("TRIAGE_LLM_BASE_URL", raising=False)
    assert TriageAssistant().backend == "mock"


def test_llm_failure_falls_back_to_rules(monkeypatch):
    monkeypatch.setenv("TRIAGE_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("TRIAGE_LLM_MODEL", "x")
    r = TriageAssistant().triage("not breathing")
    assert r.priority == "P1" and r.source.startswith("rules-fallback")


def test_llm_cannot_down_triage_p1(monkeypatch):
    ta = TriageAssistant(backend="llm")
    monkeypatch.setattr(ta, "_llm", lambda text: {"priority": "P3", "category": "minor", "red_flags": [],
                                                  "language": "ar", "confidence": 0.9, "rationale": "x"})
    r = ta.triage("المريض لا يتنفس")
    assert r.priority == "P1" and r.source == "llm+rules-override"
    json.dumps(r.to_dict(), ensure_ascii=False)
