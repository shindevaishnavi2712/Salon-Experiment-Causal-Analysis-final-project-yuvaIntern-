"""Generate reproducible synthetic salon appointment data."""
import argparse
import numpy as np
import pandas as pd

def generate(rows=300000, seed=42):
    rng = np.random.default_rng(seed)
    outlets = [f"OUTLET_{i:02d}" for i in range(1, 23)]
    outlet = rng.choice(outlets, rows)
    customer_num = rng.integers(1, 90001, rows)
    walkin = rng.random(rows) < 0.12
    customer = np.where(walkin, None, [f"CUST_{x:06d}" for x in customer_num])
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, rows), unit="D")
    services = rng.choice(["Haircut", "Hair Colour", "Facial", "Spa", "Manicure"], rows,
                          p=[.30, .18, .22, .17, .13])
    channels = rng.choice(["App", "Phone", "Walk-in", "Website"], rows, p=[.40, .20, .12, .28])
    stylist = [f"STY_{x:03d}" for x in rng.integers(1, 181, rows)]
    treatment = rng.random(rows) < .50
    period_post = rng.random(rows) < .50
    # Separate outlet-level assignment for the historical DiD illustration.
    did_treated_outlets = set(rng.choice(outlets, size=11, replace=False))
    did_group = np.where(np.isin(outlet, list(did_treated_outlets)), "treatment", "control")
    # modest simulated treatment uplift in revenue and repeat behaviour
    base = pd.Series(services).map({"Haircut": 500, "Hair Colour": 1800, "Facial": 1200,
                                    "Spa": 1600, "Manicure": 700}).to_numpy()
    outlet_effect = pd.Series(outlet).str[-2:].astype(int).to_numpy() * 3
    discount = np.clip(rng.normal(8, 5, rows), 0, 30)
    revenue = np.maximum(100, base + outlet_effect - discount * 5 + rng.normal(0, 180, rows))
    revenue = revenue + treatment * period_post * 55
    no_show_prob = np.clip(.09 - treatment * period_post * .012 + (pd.Series(channels).eq("Phone").to_numpy() * .025), .01, .30)
    no_show = rng.random(rows) < no_show_prob
    revenue = np.where(no_show, 0, revenue)
    repeat_prob = np.clip(.30 + treatment * period_post * .035 + (revenue > 1000) * .04, .05, .85)
    repeat = (rng.random(rows) < repeat_prob).astype(int)
    prior = np.maximum(0, rng.normal(900, 350, rows))
    # Historical rollout: treatment outlets receive the initiative in post period.
    group = np.where(treatment, "treatment", "control")
    period = np.where(period_post, "post", "pre")
    return pd.DataFrame({
        "appointment_id": np.arange(1, rows + 1),
        "outlet_id": outlet,
        "appointment_date": dates,
        "customer_id": customer,
        "service_category": services,
        "stylist_id": stylist,
        "channel": channels,
        "group": group,
        "period": period,
        "did_group": did_group,
        "revenue": np.round(revenue, 2),
        "discount_pct": np.round(discount, 2),
        "no_show": no_show.astype(int),
        "repeat_within_60d": repeat,
        "prior_revenue": np.round(prior, 2)
    })

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=300000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="data/salon_appointments.csv")
    a = p.parse_args()
    df = generate(a.rows, a.seed)
    df.to_csv(a.output, index=False)
    print(f"Saved {len(df):,} synthetic rows to {a.output}")
