"""Simulated B2B sales org.

Each deal has a latent health the CRM never sees. Buyer-side communication
(replies, reply latency, meetings, stakeholders) tracks that health with a
tunable strength. Seller-side activity mostly tracks the seller's habits.
CRM fields are what sellers log, with seller-specific diligence: wins are
logged at once, losses and close-date pushes arrive late.

The simulation encodes one hypothesis on purpose: buyer engagement carries
the slip signal and seller activity mostly does not. `buyer_signal_strength`
scales it, and 0.0 is the null world where communication carries no
information about health.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

STAGES = ["Qualify", "Discover", "Propose", "Negotiate", "Commit"]
SEGMENTS = ["Large", "Mid", "Scaled"]
WEEKS_PER_QUARTER = 13


@dataclass
class Config:
    seed: int = 7
    weeks: int = 78  # six quarters
    sellers: int = 80
    new_deals_per_seller_week: float = 0.9
    buyer_signal_strength: float = 1.0
    shock_rate: float = 0.03  # weekly chance a champion leaves or a budget freezes


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(cfg: Config):
    """Return (deals, weekly) DataFrames. `weekly` has one row per deal per week the deal is open in the CRM."""
    rng = np.random.default_rng(cfg.seed)
    s = cfg.buyer_signal_strength

    sellers = pd.DataFrame(
        {
            "seller_id": np.arange(cfg.sellers),
            "segment": rng.choice(SEGMENTS, cfg.sellers, p=[0.25, 0.35, 0.40]),
            "diligence": rng.uniform(0.15, 0.9, cfg.sellers),  # chance per week of updating a stale field
            "habit": rng.uniform(0.5, 4.0, cfg.sellers),  # seller emails per deal-week, regardless of health
        }
    )
    amount_scale = {"Large": 11.5, "Mid": 10.6, "Scaled": 9.6}  # log-dollars

    deals = []
    rows = []
    open_deals = []
    next_id = 1000

    for week in range(cfg.weeks):
        # New deals
        for sel in sellers.itertuples():
            for _ in range(rng.poisson(cfg.new_deals_per_seller_week)):
                cycle = int(rng.integers(8, 21))
                d = {
                    "deal_id": f"D{next_id}",
                    "seller_id": sel.seller_id,
                    "segment": sel.segment,
                    "amount": float(np.round(np.exp(rng.normal(amount_scale[sel.segment], 0.6)), -2)),
                    "created_week": week,
                    "health": rng.normal(0.4, 1.0),
                    "stage": 0,
                    "stage_since": week,
                    "true_close": week + cycle,
                    "crm_close": week + cycle,
                    "pushes": 0,
                    "truth": "open",  # open | won | lost
                    "truth_week": None,
                    "crm_status": "open",
                    "crm_status_week": None,
                    "last_logged": week,
                    "diligence": sel.diligence,
                    "habit": sel.habit,
                }
                next_id += 1
                deals.append(d)
                open_deals.append(d)

        still_open = []
        for d in open_deals:
            push = False
            # Latent dynamics, only while the buyer is still deciding
            if d["truth"] == "open":
                d["health"] += rng.normal(-0.02, 0.25)
                if rng.random() < cfg.shock_rate:
                    d["health"] -= rng.uniform(1.5, 3.0)
                h = d["health"]
                if d["stage"] < 4 and rng.random() < _sigmoid(-2.2 + 1.1 * h):
                    d["stage"] += 1
                    d["stage_since"] = week
                if h < -1.5 and rng.random() < 0.15:
                    d["truth"], d["truth_week"] = "lost", week
                elif week >= d["true_close"]:
                    if d["stage"] >= 3 and rng.random() < _sigmoid(-0.5 + 1.5 * h):
                        d["truth"], d["truth_week"] = "won", week
                    elif h < -0.5 and rng.random() < 0.5:
                        d["truth"], d["truth_week"] = "lost", week
                    else:
                        d["true_close"] += int(rng.integers(2, 9))

            # What the buyer and seller actually do this week
            h_eff = -3.0 if d["truth"] == "lost" else float(np.clip(d["health"], -3, 3))
            buyer_emails = rng.poisson(np.exp(0.3 + s * 0.6 * h_eff))
            latency = float(np.exp(rng.normal(np.log(2.0) - s * 0.4 * h_eff, 0.5))) if buyer_emails else np.nan
            meetings_held = rng.poisson(np.exp(-0.7 + s * 0.5 * h_eff))
            meetings_cancelled = rng.poisson(np.exp(-2.0 - s * 0.6 * h_eff))
            stakeholders = int(min(6, 1 + rng.poisson(np.exp(s * 0.4 * h_eff))))
            seller_emails = rng.poisson(d["habit"] * (1 + s * 0.1 * np.tanh(h_eff)))

            # What lands in the CRM
            touches = seller_emails + meetings_held
            if touches and rng.random() < 1 - (1 - d["diligence"]) ** touches:
                d["last_logged"] = week
            past_due = week > d["crm_close"]
            update_p = max(d["diligence"], 0.5) if past_due else d["diligence"]
            if d["truth"] == "won":
                d["crm_status"], d["crm_status_week"] = "won", week  # wins get logged immediately
            elif d["truth"] == "lost":
                if rng.random() < update_p:
                    d["crm_status"], d["crm_status_week"] = "lost", week
            elif d["crm_close"] != d["true_close"] and rng.random() < update_p:
                push = d["true_close"] > d["crm_close"]
                d["pushes"] += int(push)
                d["crm_close"] = d["true_close"]

            rows.append(
                {
                    "deal_id": d["deal_id"],
                    "week": week,
                    "seller_id": d["seller_id"],
                    "segment": d["segment"],
                    "amount": d["amount"],
                    "stage": STAGES[d["stage"]],
                    "stage_idx": d["stage"],
                    "weeks_in_stage": week - d["stage_since"],
                    "crm_close_week": d["crm_close"],
                    "crm_pushes": d["pushes"],
                    "pushed_this_week": int(push),
                    "weeks_since_logged": week - d["last_logged"],
                    "deal_age": week - d["created_week"],
                    "crm_status": d["crm_status"],
                    "buyer_emails": buyer_emails,
                    "buyer_latency": latency,
                    "meetings_held": meetings_held,
                    "meetings_cancelled": meetings_cancelled,
                    "stakeholders": stakeholders,
                    "seller_emails": seller_emails,
                    "true_health": d["health"] if d["truth"] == "open" else np.nan,
                    "truth": d["truth"],
                }
            )
            if d["crm_status"] == "open":
                still_open.append(d)
        open_deals = still_open

    deals_df = pd.DataFrame(deals)[
        ["deal_id", "seller_id", "segment", "amount", "created_week", "truth", "truth_week", "crm_status", "crm_status_week", "pushes"]
    ]
    return deals_df, pd.DataFrame(rows), sellers
