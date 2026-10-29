"""Features, labels, the three rankers, and the nudge-budget evaluation."""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = 4  # weeks: "bad news" means a close-date push or a lost mark within the next four weeks
FORECAST_WINDOW = 13  # only score deals the CRM says close within a quarter

CRM_FEATURES = [
    "stage_idx",
    "weeks_in_stage",
    "crm_pushes",
    "weeks_to_close",
    "past_due",
    "log_amount",
    "weeks_since_logged",
    "deal_age",
]
SIGNAL_FEATURES = [
    "buyer_emails_2w",
    "buyer_email_trend",
    "buyer_latency_2w",
    "meetings_held_4w",
    "meetings_cancelled_4w",
    "stakeholders_4w",
    "seller_emails_2w",
    "buyer_share_2w",
]


def build_features(weekly: pd.DataFrame):
    """Return (snapshots, all_weeks). A snapshot is a deal still open in the CRM at week end, with a ground-truth label."""
    w = weekly.sort_values(["deal_id", "week"]).copy()
    g = w.groupby("deal_id", sort=False)

    def roll(col, n, how="sum"):
        r = g[col].rolling(n, min_periods=1)
        return (r.sum() if how == "sum" else r.mean() if how == "mean" else r.max()).reset_index(level=0, drop=True)

    w["buyer_emails_2w"] = roll("buyer_emails", 2)
    buyer_4w = roll("buyer_emails", 4)
    w["buyer_email_trend"] = (w["buyer_emails_2w"] + 1) / (buyer_4w - w["buyer_emails_2w"] + 1)
    w["buyer_latency_2w"] = roll("buyer_latency", 2, "mean").fillna(10.0)
    w["meetings_held_4w"] = roll("meetings_held", 4)
    w["meetings_cancelled_4w"] = roll("meetings_cancelled", 4)
    w["stakeholders_4w"] = roll("stakeholders", 4, "max")
    w["seller_emails_2w"] = roll("seller_emails", 2)
    w["buyer_share_2w"] = w["buyer_emails_2w"] / (w["buyer_emails_2w"] + w["seller_emails_2w"] + 1)

    w["weeks_to_close"] = w["crm_close_week"] - w["week"]
    w["past_due"] = (w["weeks_to_close"] < 0).astype(int)
    w["log_amount"] = np.log(w["amount"])

    # Ground truth from the CRM's own future: did bad news arrive in the next HORIZON weeks?
    w["bad_now"] = ((w["pushed_this_week"] == 1) | (w["crm_status"] == "lost")).astype(int)
    g = w.groupby("deal_id", sort=False)
    future = sum(g["bad_now"].shift(-k).fillna(0) for k in range(1, HORIZON + 1))
    w["label"] = (future > 0).astype(int)

    snap = w[(w["crm_status"] == "open") & (w["weeks_to_close"] <= FORECAST_WINDOW)].copy()
    return snap, w


def fit(train: pd.DataFrame, features):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(train[features], train["label"])
    return model


def freeze(models: dict, path: str) -> str:
    """Write coefficients before any test scoring; return the hash so the report can cite it."""
    payload = {}
    for name, (model, features) in models.items():
        lr = model.named_steps["logisticregression"]
        sc = model.named_steps["standardscaler"]
        payload[name] = {
            "features": features,
            "coef": lr.coef_[0].round(6).tolist(),
            "intercept": round(float(lr.intercept_[0]), 6),
            "mean": sc.mean_.round(6).tolist(),
            "scale": sc.scale_.round(6).tolist(),
        }
    blob = json.dumps(payload, indent=2, sort_keys=True)
    with open(path, "w") as f:
        f.write(blob)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def budget_flags(test: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    """Each seller gets at most k nudges a week: their k highest-scored deals."""
    ranked = test.sort_values(["week", "seller_id", score_col, "amount"], ascending=[True, True, False, False])
    return ranked.groupby(["week", "seller_id"]).head(k)


def evaluate(test: pd.DataFrame, all_weeks: pd.DataFrame, score_col: str, k: int, event_weeks: range) -> dict:
    flags = budget_flags(test, score_col, k)
    flagged = set(zip(flags["deal_id"], flags["week"]))

    events = all_weeks[(all_weeks["bad_now"] == 1) & (all_weeks["week"].isin(event_weeks))][["deal_id", "week", "amount"]]
    caught, leads, caught_amt = 0, [], 0.0
    for e in events.itertuples():
        hits = [t for t in range(e.week - HORIZON, e.week) if (e.deal_id, t) in flagged]
        if hits:
            caught += 1
            caught_amt += e.amount
            leads.append(e.week - min(hits))
    n_seller_weeks = test.groupby(["week", "seller_id"]).ngroups
    return {
        "auc": round(float(roc_auc_score(test["label"], test[score_col])), 3),
        "nudges": int(len(flags)),
        "nudges_per_seller_week": round(len(flags) / n_seller_weeks, 2),
        "precision": round(float(flags["label"].mean()), 3),
        "events": int(len(events)),
        "recall": round(caught / max(len(events), 1), 3),
        "recall_by_amount": round(float(caught_amt / max(events["amount"].sum(), 1)), 3),
        "median_lead_days": float(np.median(leads) * 7) if leads else None,
    }
