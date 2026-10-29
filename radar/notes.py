"""Weekly forecast notes. Every fact comes from a SQL query over the CRM view, never from the model."""

import json
import sqlite3
import subprocess

import pandas as pd

from .simulate import WEEKS_PER_QUARTER

CRM_COLUMNS = ["deal_id", "week", "seller_id", "segment", "amount", "stage", "crm_close_week", "crm_status", "pushed_this_week"]


def load_db(all_weeks: pd.DataFrame, flags: pd.DataFrame) -> sqlite3.Connection:
    """CRM view only: no latent health, no truth columns. Radar flags sit in their own table."""
    con = sqlite3.connect(":memory:")
    all_weeks[CRM_COLUMNS].to_sql("crm_weeks", con, index=False)
    flags[["deal_id", "week"]].to_sql("radar_flags", con, index=False)
    return con


def _one(con, sql, **p):
    return con.execute(sql, p).fetchone()


def week_facts(con: sqlite3.Connection, t: int) -> dict:
    q = t // WEEKS_PER_QUARTER
    p = {"t": t, "prev": t - 1, "qs": q * WEEKS_PER_QUARTER, "qe": q * WEEKS_PER_QUARTER + WEEKS_PER_QUARTER - 1}

    def category(week, stages):
        marks = ",".join(f"'{s}'" for s in stages)
        return _one(
            con,
            f"""SELECT COALESCE(SUM(amount),0) FROM crm_weeks WHERE week=:w AND crm_status='open'
                AND stage IN ({marks}) AND crm_close_week BETWEEN :qs AND :qe""",
            w=week, qs=p["qs"], qe=p["qe"],
        )[0]

    commit, commit_prev = category(t, ["Commit"]), category(t - 1, ["Commit"])
    best, best_prev = category(t, ["Propose", "Negotiate"]), category(t - 1, ["Propose", "Negotiate"])
    won_qtd = _one(con, "SELECT COALESCE(SUM(amount),0) FROM crm_weeks WHERE crm_status='won' AND week BETWEEN :qs AND :t", **p)[0]
    lost_n, lost_amt = _one(con, "SELECT COUNT(*), COALESCE(SUM(amount),0) FROM crm_weeks WHERE crm_status='lost' AND week=:t", **p)
    slip_sql = """FROM crm_weeks a JOIN crm_weeks b ON a.deal_id=b.deal_id AND a.week=:prev AND b.week=:t
                  WHERE a.crm_close_week BETWEEN :qs AND :qe AND b.crm_close_week > :qe AND b.crm_status='open'"""
    slip_n, slip_amt = _one(con, f"SELECT COUNT(*), COALESCE(SUM(b.amount),0) {slip_sql}", **p)
    top_slip = _one(con, f"SELECT b.deal_id, b.amount, a.crm_close_week, b.crm_close_week {slip_sql} ORDER BY b.amount DESC LIMIT 1", **p)
    risk_n, risk_amt = _one(
        con,
        """SELECT COUNT(*), COALESCE(SUM(c.amount),0) FROM crm_weeks c JOIN radar_flags r ON c.deal_id=r.deal_id AND c.week=r.week
           WHERE c.week=:t AND c.crm_status='open' AND c.stage='Commit' AND c.crm_close_week BETWEEN :qs AND :qe""",
        **p,
    )

    facts = {
        "F1": {"label": "commit this quarter", "kind": "money", "value": commit},
        "F2": {"label": "commit change vs last week", "kind": "money_delta", "value": commit - commit_prev},
        "F3": {"label": "best case this quarter", "kind": "money", "value": best},
        "F4": {"label": "best case change vs last week", "kind": "money_delta", "value": best - best_prev},
        "F5": {"label": "won quarter to date", "kind": "money", "value": won_qtd},
        "F6": {"label": "deals slipped out of quarter this week", "kind": "count", "value": slip_n},
        "F7": {"label": "amount slipped out of quarter this week", "kind": "money", "value": slip_amt},
        "F8": {"label": "deals marked lost this week", "kind": "count", "value": lost_n},
        "F9": {"label": "amount marked lost this week", "kind": "money", "value": lost_amt},
        "F10": {"label": "commit deals flagged by radar", "kind": "count", "value": risk_n},
        "F11": {"label": "commit amount flagged by radar", "kind": "money", "value": risk_amt},
        "F12": {"label": "current week", "kind": "week", "value": t},
        "F13": {"label": "quarter number", "kind": "count", "value": q + 1},
    }
    if top_slip:
        facts.update(
            {
                "F14": {"label": "largest slipped deal", "kind": "entity", "value": top_slip[0]},
                "F15": {"label": "largest slipped deal amount", "kind": "money", "value": top_slip[1]},
                "F16": {"label": "largest slipped deal old close week", "kind": "week", "value": top_slip[2]},
                "F17": {"label": "largest slipped deal new close week", "kind": "week", "value": top_slip[3]},
            }
        )
    return facts


def money(x: float) -> str:
    x = abs(x)
    if x >= 1e6:
        return f"${x / 1e6:.1f}M"
    if x >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:.0f}"


def _deals(n: int) -> str:
    return f"{n} deal" if n == 1 else f"{n} deals"


def template_note(f: dict) -> str:
    v = {k: d["value"] for k, d in f.items()}
    direction = "up" if v["F2"] >= 0 else "down"
    parts = [
        f"Week {v['F12']}, Q{v['F13']}. Commit is {money(v['F1'])}, {direction} {money(v['F2'])} from last week; best case is {money(v['F3'])}.",
        f"Won to date this quarter: {money(v['F5'])}.",
    ]
    if v["F6"]:
        parts.append(
            f"{_deals(v['F6'])} worth {money(v['F7'])} slipped out of the quarter, led by {v['F14']} ({money(v['F15'])}, close moved from week {v['F16']} to week {v['F17']})."
        )
    parts.append(f"{_deals(v['F8'])} worth {money(v['F9'])} {'was' if v['F8'] == 1 else 'were'} marked lost.")
    if v["F10"]:
        parts.append(
            f"Radar flags {_deals(v['F10']).replace('deal', 'commit deal')} worth {money(v['F11'])}; review them before the forecast call."
        )
    return " ".join(parts)


LLM_PROMPT = """You write the weekly forecast note a VP of Sales reads before the forecast call.
Use only the facts in the JSON below. Under 110 words, plain prose, no headings or bullets.
Lead with what changed and what it means for the quarter. After every number you state, cite
its fact id in square brackets, like "$1.2M [F1]". Write money as $X.XM or $XK. Do not compute
new numbers (no sums, ratios or percentages that are not in the facts). Name deals only by the ids given.

FACTS:
{facts}
"""


def llm_note(f: dict, timeout: int = 180) -> str | None:
    """Optional: ask Claude (via the local `claude` CLI) to write the note. Returns None if unavailable."""
    facts = {k: {"label": d["label"], "value": d["value"]} for k, d in f.items()}
    try:
        out = subprocess.run(
            ["claude", "-p", LLM_PROMPT.format(facts=json.dumps(facts, indent=1))],
            capture_output=True, text=True, timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
