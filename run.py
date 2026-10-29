"""End-to-end run: simulate, train, freeze, score the held-out quarter, write and verify forecast notes, report.

    .venv/bin/python run.py          # template notes only
    .venv/bin/python run.py --llm    # also ask Claude (local `claude` CLI) to write notes, then verify them
    .venv/bin/python run.py --reuse-notes   # re-verify the Claude notes already saved in report/results.json
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from radar import detect, notes, verify
from radar.simulate import Config, simulate

TRAIN_WEEKS = range(13, 61)  # first quarter is burn-in; labels look ahead to week 64
TEST_WEEKS = range(65, 74)  # held-out final quarter; labels need weeks through 77
EVENT_WEEKS = range(66, 78)
# Recall counts only events whose whole four-week warning window falls inside the test weeks
FULL_WINDOW_EVENT_WEEKS = range(TEST_WEEKS.start + 4, TEST_WEEKS.stop + 1)
NOTE_WEEKS = range(70, 74)
K = 2  # nudges per seller per week: precision matters more than recall, or sellers learn to ignore them
SCORES = {"activity_rule": "score_rule", "crm_fields_model": "score_crm", "radar": "score_radar"}
ALL_FEATURES = detect.CRM_FEATURES + detect.SIGNAL_FEATURES


def run_world(cfg: Config, freeze_path: str | None = None) -> dict:
    deals, weekly, _ = simulate(cfg)
    snap, all_weeks = detect.build_features(weekly)
    train = snap[snap["week"].isin(TRAIN_WEEKS)]
    test = snap[snap["week"].isin(TEST_WEEKS)].copy()

    crm = detect.fit(train, detect.CRM_FEATURES)
    radar = detect.fit(train, ALL_FEATURES)
    frozen = detect.freeze({"crm_fields_model": (crm, detect.CRM_FEATURES), "radar": (radar, ALL_FEATURES)}, freeze_path) if freeze_path else None

    # Scored only after the models are frozen
    test["score_rule"] = test["weeks_since_logged"] + test["amount"] / 1e9
    test["score_crm"] = crm.predict_proba(test[detect.CRM_FEATURES])[:, 1]
    test["score_radar"] = radar.predict_proba(test[ALL_FEATURES])[:, 1]
    results = {name: detect.evaluate(test, all_weeks, col, K, FULL_WINDOW_EVENT_WEEKS) for name, col in SCORES.items()}
    return {
        "deals": deals, "all_weeks": all_weeks, "train": train, "test": test,
        "radar": radar, "results": results, "frozen": frozen,
        "base_rate": round(float(test["label"].mean()), 3),
    }


def hidden_risk(world: dict) -> dict:
    """Added after the first run: bad news on deals the CRM does not already show as due (close date 3+ weeks out)."""
    sub = world["test"][world["test"]["weeks_to_close"] >= 3]
    out = {"snapshots": int(len(sub)), "base_rate": round(float(sub["label"].mean()), 3)}
    for name, col in SCORES.items():
        flags = detect.budget_flags(sub, col, K)
        pos = set(sub.loc[sub["label"] == 1, "deal_id"])
        out[name] = {
            "auc": round(float(roc_auc_score(sub["label"], sub[col])), 3),
            "precision": round(float(flags["label"].mean()), 3),
            "positive_deals_flagged": round(len(pos & set(flags.loc[flags["label"] == 1, "deal_id"])) / max(len(pos), 1), 3),
        }
    return out


def zombies(world: dict) -> dict:
    """Added after the first run: deals the buyer has already walked away from that the CRM still shows open."""
    test, w = world["test"], world["all_weeks"]
    truth = w[w["truth"] == "lost"].groupby("deal_id")["week"].min().rename("true_loss_week")
    crm = w[w["crm_status"] == "lost"].groupby("deal_id")["week"].min().rename("crm_lost_week")
    test = test.join(truth, on="deal_id")
    test["zombie"] = (test["true_loss_week"] <= test["week"]).astype(int)
    out = {"zombie_snapshots": int(test["zombie"].sum()), "open_snapshots": int(len(test))}
    lag = (crm - truth).dropna()
    out["median_weeks_dead_before_marked_lost"] = float(lag.median())
    out["zombie_amount_in_commit_and_best_case_week_73"] = float(
        test[(test["week"] == 73) & (test["zombie"] == 1) & (test["stage"].isin(["Propose", "Negotiate", "Commit"]))]["amount"].sum()
    )
    for name, col in SCORES.items():
        flags = detect.budget_flags(test, col, K)
        z = flags[flags["zombie"] == 1].groupby("deal_id")["week"].min()
        z = z.to_frame("first_flag").join(crm)
        earlier = (z["crm_lost_week"] - z["first_flag"]).dropna()
        out[name] = {
            "auc": round(float(roc_auc_score(test["zombie"], test[col])), 3),
            "zombie_deals_flagged": int(len(z)),
            "median_weeks_flagged_before_marked_lost": float(earlier.median()) if len(earlier) else None,
        }
    out["zombie_deals"] = int(test.loc[test["zombie"] == 1, "deal_id"].nunique())
    return out


def push_sizes(all_weeks: pd.DataFrame, weeks: range) -> pd.Series:
    w = all_weeks.sort_values(["deal_id", "week"])
    jump = w.groupby("deal_id")["crm_close_week"].diff()
    return jump[(w["pushed_this_week"] == 1) & w["week"].isin(weeks)]


def drafted_updates(world: dict, median_push: int) -> dict:
    """Score the seller-facing draft per distinct push: the earliest flag since the deal's previous push suggests a date
    (CRM close plus the training median push); compare it to this push, which is the next recorded close date after that flag."""
    test, all_weeks = world["test"], world["all_weeks"]
    flags = detect.budget_flags(test, "score_radar", K)
    pushes = all_weeks[all_weeks["pushed_this_week"] == 1].sort_values(["deal_id", "week"])[["deal_id", "week", "crm_close_week"]]
    pushes["prev_push_week"] = pushes.groupby("deal_id")["week"].shift(1)
    pushes = pushes[pushes["week"].isin(EVENT_WEEKS)]
    errs_draft, errs_stale = [], []
    for p in pushes.itertuples():
        lo = p.week - detect.HORIZON
        if not pd.isna(p.prev_push_week):
            lo = max(lo, int(p.prev_push_week))  # a flag from before an earlier push belongs to that push
        before = flags[(flags["deal_id"] == p.deal_id) & (flags["week"] < p.week) & (flags["week"] >= lo)]
        if before.empty:
            continue
        first = before.sort_values("week").iloc[0]
        errs_draft.append(abs(first["crm_close_week"] + median_push - p.crm_close_week))
        errs_stale.append(abs(first["crm_close_week"] - p.crm_close_week))
    return {
        "distinct_pushes_scored": len(errs_draft),
        "median_abs_error_weeks_draft": float(np.median(errs_draft)) if errs_draft else None,
        "median_abs_error_weeks_unchanged": float(np.median(errs_stale)) if errs_stale else None,
    }


def example_nudges(world: dict, median_push: int, week: int, n: int = 3) -> list[str]:
    # Drawn from the real two-nudge policy; shown only where the CRM close date is still 3+ weeks out
    flags = detect.budget_flags(world["test"], "score_radar", K)
    flags = flags[flags["weeks_to_close"] >= 3]
    wk = flags[flags["week"] == week].assign(dollars_at_risk=lambda d: d["score_radar"] * d["amount"])
    top = wk.sort_values("dollars_at_risk", ascending=False).head(n)  # illustration only, not a metric
    pl = lambda n, word: f"{n} {word}" if n == 1 else f"{n} {word}s"
    out = []
    for r in top.itertuples():
        out.append(
            f"{r.deal_id} ({notes.money(r.amount)}, {r.stage}, CRM close week {r.crm_close_week}): "
            f"{pl(int(r.buyer_emails_2w), 'buyer email')} in two weeks, {pl(int(r.meetings_cancelled_4w), 'meeting')} cancelled in four, "
            f"{pl(int(r.stakeholders_4w), 'buyer stakeholder')} engaged; last logged activity {pl(int(r.weeks_since_logged), 'week')} ago. "
            + (
                f"Buyer has gone quiet. Still active? [Yes, move close to week {int(r.crm_close_week + median_push)}] [Mark lost] [Not now]"
                if r.buyer_emails_2w == 0
                else f"Suggested update: move close to week {int(r.crm_close_week + median_push)}. [Confirm] [Edit] [Not now]"
            )
        )
    return out


def note_checks(world: dict, use_llm: bool, seed: int, saved: dict | None = None) -> dict:
    flags = detect.budget_flags(world["test"], "score_radar", K)
    con = notes.load_db(world["all_weeks"], flags)
    rng = np.random.default_rng(seed)
    written, mutations = [], {"template": {}, "llm": {}}
    for t in NOTE_WEEKS:
        facts = notes.week_facts(con, t)
        for source in ("template", "llm") if use_llm else ("template",):
            if source == "template":
                text = notes.template_note(facts)
            else:  # model output varies run to run; --reuse-notes re-verifies the exact notes already on file
                text = saved.get(t) if saved is not None else notes.llm_note(facts)
            if text is None:
                continue
            check = verify.verify(text, facts, require_citations=(source == "llm"))
            written.append({"week": t, "source": source, "note": text, **check})
            if not check["ok"]:
                continue
            for kind in ("number", "direction", "entity"):
                tally = mutations[source].setdefault(kind, {"planted": 0, "caught": 0})
                for _ in range(40):
                    bad = verify.mutate(text, facts, rng, kind)
                    if bad is None or bad == text:
                        continue
                    tally["planted"] += 1
                    tally["caught"] += int(not verify.verify(bad, facts, require_citations=(source == "llm"))["ok"])
    return {"notes": written, "mutations": mutations}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--reuse-notes", action="store_true", help="with --llm, verify the Claude notes saved in report/results.json instead of writing new ones")
    args = ap.parse_args()
    saved = None
    if args.reuse_notes:
        with open("report/results.json") as f:
            saved = {n["week"]: n["note"] for n in json.load(f)["forecast_notes"]["notes"] if n["source"] == "llm"}

    main_world = run_world(Config(seed=7, buyer_signal_strength=1.0), freeze_path="report/frozen_models.json")
    sensitivity = {"1.0": main_world["results"]}
    posthoc = {"1.0": {"hidden_risk": hidden_risk(main_world), "zombies": zombies(main_world)}}
    for s in (0.5, 0.25, 0.0):
        world = run_world(Config(seed=7, buyer_signal_strength=s))
        sensitivity[str(s)] = world["results"]
        posthoc[str(s)] = {"hidden_risk": hidden_risk(world), "zombies": zombies(world)}
    seeds = {"7": main_world["results"]}
    fresh_slices = {}  # the post-hoc slices, rechecked on worlds that played no part in choosing them
    for seed in (11, 23):
        world = run_world(Config(seed=seed, buyer_signal_strength=1.0))
        seeds[str(seed)] = world["results"]
        null = run_world(Config(seed=seed, buyer_signal_strength=0.0))
        fresh_slices[str(seed)] = {
            "signal": {"hidden_risk": hidden_risk(world), "zombies": zombies(world)},
            "null": {"hidden_risk": hidden_risk(null), "zombies": zombies(null)},
        }

    median_push = int(push_sizes(main_world["all_weeks"], TRAIN_WEEKS).median())
    coef = dict(zip(ALL_FEATURES, main_world["radar"].named_steps["logisticregression"].coef_[0].round(2).tolist()))

    out = {
        "config": {"train_weeks": [TRAIN_WEEKS.start, TRAIN_WEEKS.stop - 1], "test_weeks": [TEST_WEEKS.start, TEST_WEEKS.stop - 1],
                   "horizon_weeks": detect.HORIZON, "nudges_per_seller_week": K, "frozen_models_sha256_12": main_world["frozen"]},
        "data": {"deals": int(len(main_world["deals"])), "train_snapshots": int(len(main_world["train"])),
                 "test_snapshots": int(len(main_world["test"])), "test_base_rate": main_world["base_rate"]},
        "main": main_world["results"],
        "sensitivity_by_signal_strength": sensitivity,
        "seeds": seeds,
        "posthoc_by_signal_strength": posthoc,
        "posthoc_fresh_seeds": fresh_slices,
        "radar_coefficients_standardized": coef,
        "median_push_weeks": median_push,
        "drafted_updates": drafted_updates(main_world, median_push),
        "example_nudges_week_73": example_nudges(main_world, median_push, 73),
        "forecast_notes": note_checks(main_world, args.llm or args.reuse_notes, seed=7, saved=saved),
    }
    with open("report/results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({k: out[k] for k in ("config", "data", "main", "drafted_updates")}, indent=2))


if __name__ == "__main__":
    main()
