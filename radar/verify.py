"""Checks every number, deal id and direction word in a forecast note against the week's SQL facts."""

import re

TOKEN = re.compile(
    r"(?P<cite>\[F\d+\])"
    r"|(?P<entity>\bD\d{3,}\b)"
    r"|(?P<money>\$\d+(?:,\d{3})*(?:\.\d+)?[MK]?)"
    r"|(?P<pct>(?<![\w.])\d+(?:\.\d+)?%)"
    r"|(?P<quarter>\bQ\d+\b)"
    r"|(?P<num>(?<![\w.$])\d+(?:,\d{3})*(?:\.\d+)?(?![\d%]))"
)
UP = {"up", "rose", "increased", "grew", "gained", "added", "higher", "climbed", "rise", "increase", "gain"}
DOWN = {"down", "fell", "decreased", "dropped", "declined", "lower", "shrank", "drop", "decline", "decrease", "fall"}
WORD = re.compile(r"[A-Za-z]+")
SENTENCE_END = re.compile(r"[.;!?](?=\s|$)")
MINUS = "-\u2212\u2013\u2014"  # hyphen-minus, minus sign, en dash, em dash
# Quantities the tokenizer cannot parse are rejected, not skipped. "one" is left out because it is usually a pronoun.
SPELLED = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|"
    r"trillion|dozen|dollars?|percent|bn|mn)\b",
    re.IGNORECASE,
)


def _parse(tok_kind, text):
    """Return (value, tolerance) for a numeric token, honoring the precision it was written with."""
    if tok_kind == "money":
        body = text[1:].replace(",", "")
        mult = {"M": 1e6, "K": 1e3}.get(body[-1], 1.0)
        body = body.rstrip("MK")
        decimals = len(body.split(".")[1]) if "." in body else 0
        return float(body) * mult, 0.5 * 10 ** (-decimals) * mult
    if tok_kind == "quarter":
        return float(text[1:]), 0.0
    body = text.rstrip("%").replace(",", "")
    decimals = len(body.split(".")[1]) if "." in body else 0
    return float(body), 0.5 * 10 ** (-decimals)


def _negative(note, start):
    """A minus sign or dash before the token, directly or after one space, but not a hyphen inside a word or range.

    A spaced dash used as punctuation ("Commit - $6.8M") also reads as negative, so it fails against a positive
    total. That is deliberate: rejecting an ambiguous sign is safer than passing a wrong one.
    """
    i = start - 1
    if i >= 0 and note[i] == " ":
        i -= 1
        return i >= 0 and note[i] in MINUS and (i == 0 or note[i - 1].isspace())
    return i >= 0 and note[i] in MINUS and (i == 0 or not note[i - 1].isalnum())


COMPATIBLE = {"money": {"money", "money_delta"}, "num": {"count", "week"}, "quarter": {"count"}, "pct": set()}


def _matches(fact, kind, value, tol, signed):
    """Totals compare signed. An unsigned change amount compares by magnitude; its direction is checked separately."""
    if fact["kind"] not in COMPATIBLE[kind]:
        return False
    target = float(fact["value"])
    if fact["kind"] == "money_delta" and not signed:
        target = abs(target)
    return abs(target - value) <= tol + 1e-9


def _direction(note, start, end):
    """Sign of the direction word nearest the amount, within its own sentence."""
    lo = max([m.end() for m in SENTENCE_END.finditer(note, 0, start)], default=0)
    nxt = SENTENCE_END.search(note, end)
    hi = nxt.start() if nxt else len(note)
    best = None
    for m in WORD.finditer(note, lo, hi):
        w = m.group().lower()
        if w not in UP and w not in DOWN:
            continue
        dist = start - m.end() if m.end() <= start else m.start() - end
        if best is None or dist < best[0]:
            best = (dist, 1 if w in UP else -1)
    return best[1] if best else 0


def verify(note: str, facts: dict, require_citations: bool = False) -> dict:
    tokens = [(m.lastgroup, m.group(), m.start(), m.end()) for m in TOKEN.finditer(note)]
    entities = {str(f["value"]) for f in facts.values() if f["kind"] == "entity"}
    issues, checked = [], 0
    for m in SPELLED.finditer(note):
        issues.append(f"'{m.group()}' is a quantity the checker cannot verify; write it in digits")
    for i, (kind, text, start, end) in enumerate(tokens):
        if kind == "cite":
            if text[1:-1] not in facts:
                issues.append(f"cites missing fact {text}")
            continue
        checked += 1
        cite = None
        if i + 1 < len(tokens) and tokens[i + 1][0] == "cite":
            between = note[end : tokens[i + 1][2]]
            if len(WORD.findall(between)) <= 3 and not re.search(r"[.;!?]\s", between):
                cite = tokens[i + 1][1][1:-1]  # same clause, at most three words away, no other value in between
        if require_citations and cite is None and kind != "entity":  # the prompt requires a citation after every number
            issues.append(f"{text} has no citation")
            continue
        if kind == "entity":
            ok = text in entities if cite is None else (cite in facts and str(facts[cite]["value"]) == text)
            if not ok:
                issues.append(f"deal {text} is not in the facts" if cite is None else f"deal {text} does not match {cite}")
            continue
        value, tol = _parse(kind, text)
        signed = _negative(note, start)
        if signed:
            value = -value
        candidates = [facts[cite]] if cite in facts else ([] if cite else list(facts.values()))
        hits = [f for f in candidates if _matches(f, kind, value, tol, signed)]
        if not hits:
            where = f"cited fact {cite}" if cite else "any fact"
            issues.append(f"{'-' if signed else ''}{text} does not match {where}")
            continue
        deltas = [f for f in hits if f["kind"] == "money_delta"]
        if deltas and len(deltas) == len(hits) and not signed:
            sign = 1 if float(deltas[0]["value"]) >= 0 else -1
            said = _direction(note, start, end)
            if said == 0:
                issues.append(f"{text} is a change but the note gives no direction")
            elif said != sign:
                issues.append(f"{text} is described in the wrong direction")
    return {"ok": not issues, "issues": issues, "checked": checked}


def mutate(note: str, facts: dict, rng, kind: str):
    """Plant one error. Returns the corrupted note, or None if this note offers nothing to corrupt."""
    tokens = [m for m in TOKEN.finditer(note) if m.lastgroup in ("money", "num")]
    if kind == "number" and tokens:
        m = tokens[rng.integers(len(tokens))]
        value, _ = _parse(m.lastgroup, m.group())
        factor = rng.choice([rng.uniform(0.5, 0.85), rng.uniform(1.15, 1.6)])
        new = value * factor
        if m.lastgroup == "money":
            text = m.group()
            if text.endswith("M"):
                rep = f"${new / 1e6:.1f}M"
            elif text.endswith("K"):
                rep = f"${new / 1e3:.0f}K"
            else:
                rep = f"${new:.0f}"
        else:
            rep = str(max(int(round(new)), int(value) + 1))
        return None if rep == m.group() else note[: m.start()] + rep + note[m.end() :]
    if kind == "direction":
        flip = {"up": "down", "down": "up", "fell": "rose", "rose": "fell", "dropped": "grew", "grew": "dropped",
                "declined": "increased", "increased": "declined"}
        found = [(m.start(), m.end(), m.group()) for m in WORD.finditer(note) if m.group().lower() in flip]
        if not found:
            return None
        s, e, w = found[rng.integers(len(found))]
        return note[:s] + flip[w.lower()] + note[e:]
    if kind == "entity":
        ids = list(re.finditer(r"\bD\d{3,}\b", note))
        if not ids:
            return None
        m = ids[rng.integers(len(ids))]
        return note[: m.start()] + f"D{int(m.group()[1:]) + int(rng.integers(1, 400))}" + note[m.end() :]
    return None
