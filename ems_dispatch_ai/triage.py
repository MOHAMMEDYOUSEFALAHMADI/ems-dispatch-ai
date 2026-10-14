"""Call-note triage assistant: turns free-text Arabic/English call notes into a structured priority.

Default backend is a transparent keyword-rule mock (no network, no keys). An optional
OpenAI-compatible LLM backend is used only when TRIAGE_LLM_BASE_URL and TRIAGE_LLM_MODEL are set;
its JSON is schema-validated and any failure falls back to the rules.

Research prototype - NOT a validated dispatch protocol (e.g. MPDS) and not for clinical use.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass, field

PRIORITIES = ("P1", "P2", "P3")
CATEGORIES = ("cardiac_arrest", "breathing", "chest_pain", "stroke", "trauma", "bleeding", "seizure",
              "allergy", "obstetric", "medical", "minor")

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u0640]")


def normalize(text: str) -> str:
    t = _DIACRITICS.sub("", text.lower())
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t)


# (priority, category, red_flag_label, [patterns]) - first matching P1 wins; patterns are normalised.
RULES = [
    ("P1", "cardiac_arrest", "no breathing / no pulse",
     ["not breathing", "stopped breathing", "no pulse", "cardiac arrest", "لا يتنفس", "ما يتنفس", "توقف عن التنفس",
      "بدون نبض", "لا يوجد نبض", "توقف قلب", "سكته قلبيه"]),
    ("P1", "breathing", "airway / breathing compromise",
     ["choking", "can't breathe", "cannot breathe", "difficulty breathing", "shortness of breath", "turning blue",
      "اختناق", "ضيق تنفس", "ضيق في التنفس", "صعوبه في التنفس", "ما يقدر يتنفس", "ازرق"]),
    ("P1", "chest_pain", "chest pain",
     ["chest pain", "crushing chest", "الم في الصدر", "الم صدر", "الم بالصدر", "ضغط على الصدر"]),
    ("P1", "stroke", "stroke signs",
     ["face droop", "slurred speech", "one side weak", "weakness on one side", "stroke", "ارتخاء الوجه",
      "ميلان الوجه", "صعوبه في الكلام", "ثقل في اللسان", "جلطه دماغيه", "شلل نصفي", "ضعف في جهه"]),
    ("P1", "medical", "unconscious",
     ["unconscious", "unresponsive", "passed out and not waking", "فاقد الوعي", "مغمي عليه", "لا يستجيب", "ما يصحى"]),
    ("P1", "bleeding", "severe bleeding",
     ["severe bleeding", "heavy bleeding", "bleeding heavily", "نزيف شديد", "نزيف حاد", "نزيف قوي"]),
    ("P1", "trauma", "major trauma",
     ["stabbed", "stab wound", "gunshot", "shot", "rollover", "ejected", "pedestrian struck", "hit by a car",
      "طعن", "طلق ناري", "انقلاب", "دهس", "سقوط من ارتفاع", "غرق", "drowning"]),
    ("P1", "allergy", "anaphylaxis",
     ["throat swelling", "anaphylaxis", "tongue swelling", "تورم الحلق", "تورم اللسان", "حساسيه شديده"]),
    ("P1", "seizure", "active seizure",
     ["seizing", "seizure", "convulsing", "تشنج", "نوبه صرع", "يتشنج"]),
    ("P2", "obstetric", None, ["labor", "contractions", "pregnant", "ولاده", "طلق", "حامل"]),
    ("P2", "trauma", None, ["fracture", "broken", "fell", "fall", "car accident", "collision", "كسر", "سقط", "سقوط",
                            "حادث", "اصطدام"]),
    ("P2", "medical", None, ["abdominal pain", "vomiting", "high fever", "diabetic", "low sugar", "fainted",
                             "الم في البطن", "استفراغ", "ترجيع", "حراره", "حمي", "سكر", "هبوط", "اغماء", "دوخه"]),
    ("P3", "minor", None, ["minor cut", "small cut", "sprain", "mild", "جرح بسيط", "التواء", "خفيف"]),
]
_NEGATIONS = ["no chest pain", "not unconscious", "breathing normally", "يتنفس بشكل طبيعي", "لا يوجد الم في الصدر",
              "ما فيه الم صدر", "واعي"]


@dataclass
class TriageResult:
    priority: str
    category: str
    red_flags: list[str] = field(default_factory=list)
    language: str = "en"
    confidence: float = 0.5
    rationale: str = ""
    source: str = "rules"

    def to_dict(self) -> dict:
        return asdict(self)


def detect_language(text: str) -> str:
    ar = sum("\u0600" <= ch <= "\u06ff" for ch in text)
    lat = sum(ch.isascii() and ch.isalpha() for ch in text)
    return "ar" if ar >= lat else "en"


def rule_triage(text: str) -> TriageResult:
    t = normalize(text)
    for neg in _NEGATIONS:
        t = t.replace(normalize(neg), " ")
    lang = detect_language(text)
    flags, best = [], None
    for pri, cat, flag, pats in RULES:
        if any(normalize(p) in t for p in pats):
            if flag and flag not in flags:
                flags.append(flag)
            if best is None or PRIORITIES.index(pri) < PRIORITIES.index(best[0]):
                best = (pri, cat)
    if best is None:
        return TriageResult("P3", "minor", [], lang, 0.3, "No red-flag keywords found; defaulting to P3 - "
                            "call-taker should ask structured questions.")
    conf = 0.85 if best[0] == "P1" else 0.6
    return TriageResult(best[0], best[1], flags, lang, conf,
                        f"Matched keyword rules for {best[1]}" + (f"; red flags: {', '.join(flags)}" if flags else ""))


SYSTEM = ("You are an EMS call-triage assistant for Saudi Red Crescent call-takers (research prototype). "
          "Given a call note in Arabic or English, return ONLY JSON with keys: priority (P1|P2|P3), category ("
          + "|".join(CATEGORIES) + "), red_flags (list of short English strings), language (ar|en), confidence "
          "(0-1), rationale (one English sentence). P1 = immediately life-threatening. When unsure, over-triage.")


def validate(d: dict) -> TriageResult:
    if d.get("priority") not in PRIORITIES or d.get("category") not in CATEGORIES:
        raise ValueError("invalid priority/category")
    flags = d.get("red_flags") or []
    if not isinstance(flags, list):
        raise ValueError("red_flags must be a list")
    conf = float(d.get("confidence", 0.5))
    return TriageResult(d["priority"], d["category"], [str(f) for f in flags], d.get("language", "en"),
                        min(max(conf, 0.0), 1.0), str(d.get("rationale", "")), "llm")


class TriageAssistant:
    def __init__(self, backend: str | None = None):
        self.base_url = os.environ.get("TRIAGE_LLM_BASE_URL", "").rstrip("/")
        self.model = os.environ.get("TRIAGE_LLM_MODEL", "")
        self.api_key = os.environ.get("TRIAGE_LLM_API_KEY", "")
        self.backend = backend or ("llm" if self.base_url and self.model else "mock")

    def _llm(self, text: str) -> dict:
        body = {"model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            content = json.loads(r.read())["choices"][0]["message"]["content"]
        return json.loads(re.search(r"\{.*\}", content, re.S).group())

    def triage(self, text: str) -> TriageResult:
        rules = rule_triage(text)
        if self.backend != "llm":
            return rules
        try:
            res = validate(self._llm(text))
        except Exception as e:
            rules.source = f"rules-fallback ({type(e).__name__})"
            return rules
        # Safety net: never let the LLM down-triage a rule-detected P1 red flag.
        if rules.priority == "P1" and res.priority != "P1":
            res.priority, res.source = "P1", "llm+rules-override"
            res.red_flags = sorted(set(res.red_flags) | set(rules.red_flags))
        return res
