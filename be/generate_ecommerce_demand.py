#!/usr/bin/env python3
"""Generate a realistic synthetic e-commerce daily demand dataset.

Output: ecommerce_demand.csv with columns (in order):
- date (YYYY-MM-DD)
- sku_id (SKU_001, SKU_002, SKU_003)
- units_sold (non-negative integer)
- price (float)
- promotion_flag (0 or 1)

Uses only NumPy and pandas. Reproducible via fixed random seed.
"""

import numpy as np
import pandas as pd

# -----------------------------
# Configuration / constants
# -----------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
SKUS = ["SKU_001", "SKU_002", "SKU_003"]

# Per-SKU parameters (chosen to produce realistic but synthetic behaviour)
BASE_DEMAND = {
    "SKU_001": 80.0,  # higher-volume product
    "SKU_002": 45.0,  # mid-volume
    "SKU_003": 18.0,  # low-volume / niche
}
BASE_PRICE = {
    "SKU_001": 19.99,
    "SKU_002": 9.99,
    "SKU_003": 49.99,
}
# relative weekly/yearly seasonality strengths (fraction of base demand)
WEEKLY_STRENGTH = {"SKU_001": 0.20, "SKU_002": 0.28, "SKU_003": 0.12}
YEARLY_STRENGTH = {"SKU_001": 0.30, "SKU_002": 0.22, "SKU_003": 0.08}
# mild upward trend over whole period (fractional increase over ~3 years)
TREND_FRACTION = {"SKU_001": 0.12, "SKU_002": 0.10, "SKU_003": 0.18}

# Price evolution noise parameters
DAILY_PRICE_NOISE_STD = 0.007  # slightly larger daily fluctuations (~0.7%) to avoid repeated identical prices
# Per-SKU adjustments
DAILY_PRICE_NOISE_STD_SKU = {"SKU_001": DAILY_PRICE_NOISE_STD, "SKU_002": DAILY_PRICE_NOISE_STD, "SKU_003": DAILY_PRICE_NOISE_STD * 3.0}
PRICE_ELASTICITY_SKU = {"SKU_001": -1.5, "SKU_002": -1.5, "SKU_003": -0.8}

# Promotion behaviour
PROMO_PROB_RANGE = (0.15, 0.25)  # per-SKU, fraction of days in promotion
PROMO_DISCOUNT_RANGE = (0.05, 0.20)  # 5% - 20% discount when applied
PROMO_DISCOUNT_PROB = 0.6  # chance a promotion day actually has a price discount
PROMO_LIFT_RANGE = (0.25, 0.60)  # fractional demand uplift during promotions
# For SKU_003, generate multi-day promo windows for realism
PROMO_WINDOW_MEAN = 4
PROMO_WINDOW_MIN = 2
PROMO_WINDOW_MAX = 8

# Demand shocks and overdispersion (for realism on SKU_003)
SHOCK_PROB = 0.01  # rare shock days
SHOCK_MULT_RANGE = (1.5, 3.0)
DISPERSION_SHAPE_SKU3 = 0.9  # gamma shape to increase variance in Poisson sampling for SKU_003

# -----------------------------
# Helper / generate arrays
# -----------------------------

dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
n_days = len(dates)

day_idx = np.arange(n_days)  # 0..n-1
weekday = dates.weekday.values  # 0 (Mon) .. 6 (Sun)
day_of_year = dates.dayofyear.values  # 1..366

rows = []

for sku in SKUS:
    base = BASE_DEMAND[sku]
    base_price = BASE_PRICE[sku]

    # Seasonality components
    weekly_component = WEEKLY_STRENGTH[sku] * base * np.sin(2 * np.pi * day_idx / 7)
    yearly_component = YEARLY_STRENGTH[sku] * base * np.sin(2 * np.pi * day_of_year / 365.25)

    # Trend: linear growth over the whole period
    trend_total = base * TREND_FRACTION[sku]
    trend = trend_total * (day_idx / float(n_days - 1))

    # Random noise (additive)
    noise_std = max(1.0, base * 0.12)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=n_days)

    # Promotion flags and windows (per-SKU)
    promo_p = np.random.uniform(PROMO_PROB_RANGE[0], PROMO_PROB_RANGE[1])
    if sku == "SKU_003":
        # Build multi-day promotion windows for SKU_003 for realism
        expected_promo_days = int(np.round(promo_p * n_days))
        avg_len = PROMO_WINDOW_MEAN
        num_windows = max(1, expected_promo_days // max(1, avg_len))
        promotion_flag = np.zeros(n_days, dtype=int)
        starts = np.random.choice(np.arange(n_days), size=num_windows, replace=False)
        for s in starts:
            length = np.random.randint(PROMO_WINDOW_MIN, PROMO_WINDOW_MAX)
            end = min(n_days, s + length)
            promotion_flag[s:end] = 1
        # discounts more consistent in windows
        discount_pct = np.zeros(n_days)
        window_days = promotion_flag.astype(bool)
        if window_days.sum() > 0:
            discount_pct[window_days] = np.random.uniform(0.08, 0.25, size=window_days.sum())
        promo_lift = np.random.uniform(PROMO_LIFT_RANGE[0] * 1.2, PROMO_LIFT_RANGE[1] * 1.2)
    else:
        promotion_flag = np.random.binomial(1, promo_p, size=n_days)
        promo_lift = np.random.uniform(PROMO_LIFT_RANGE[0], PROMO_LIFT_RANGE[1])
        will_discount = (np.random.rand(n_days) < PROMO_DISCOUNT_PROB) & (promotion_flag == 1)
        discount_pct = np.zeros(n_days)
        if will_discount.sum() > 0:
            discount_pct[will_discount] = np.random.uniform(PROMO_DISCOUNT_RANGE[0], PROMO_DISCOUNT_RANGE[1], size=will_discount.sum())

    # Price evolution: mean-reverting random walk + occasional promotional drops
    price = np.empty(n_days, dtype=float)
    price[0] = base_price * (1.0 + np.random.normal(0.0, 0.003))
    # per-sku reversion and bounds
    if sku == "SKU_003":
        min_factor = 0.7
        max_factor = 1.3
        reversion_rate = 0.02
        local_price_noise = DAILY_PRICE_NOISE_STD_SKU[sku]
    else:
        min_factor = 0.8
        max_factor = 1.2
        reversion_rate = 0.04
        local_price_noise = DAILY_PRICE_NOISE_STD_SKU[sku]

    for t in range(1, n_days):
        # mean reversion toward base price plus multiplicative noise
        drift = reversion_rate * (base_price - price[t - 1])
        p_noise = price[t - 1] * np.random.normal(0.0, local_price_noise)
        price[t] = price[t - 1] + drift + p_noise
        # apply promotional discount on promotional days
        if discount_pct[t] > 0:
            price[t] = price[t] * (1.0 - discount_pct[t])
        # clip and add small jitter
        price[t] = float(np.clip(price[t], base_price * min_factor, base_price * max_factor))
        price[t] += np.random.normal(0.0, 0.03)
    price = np.round(price.clip(min=0.01), 2)

    # Price effect (change relative to mean price) scaled by elasticity (per-SKU)
    price_mean = price.mean()
    price_effect = PRICE_ELASTICITY_SKU[sku] * (price - price_mean) / price_mean * base

    # Raw demand (additive components)
    units_raw = base + trend + weekly_component + yearly_component + noise + price_effect

    # Occasional demand shocks
    shocks = (np.random.rand(n_days) < SHOCK_PROB)
    if shocks.any():
        shock_mults = np.random.uniform(SHOCK_MULT_RANGE[0], SHOCK_MULT_RANGE[1], size=shocks.sum())
        units_raw[shocks] = units_raw[shocks] * shock_mults

    # Apply promotional uplift multiplicatively on promotion days
    units_promoted = units_raw * (1.0 + (promotion_flag * promo_lift))

    # Finalize: sample integer counts from a (possibly overdispersed) Poisson
    lam = np.clip(units_promoted, a_min=0.1, a_max=None)
    if sku == "SKU_003":
        # Introduce overdispersion by sampling lambda from a Gamma before Poisson
        shape = DISPERSION_SHAPE_SKU3
        # avoid division-by-zero for small lam
        lam_gamma = np.random.gamma(shape, (lam / shape).clip(min=1e-6))
        units_final = np.random.poisson(lam_gamma).astype(int)
    else:
        units_final = np.random.poisson(lam).astype(int)
    # Safety: ensure non-negative
    units_final = np.maximum(0, units_final)

    # Append rows for this SKU
    sku_col = np.array([sku] * n_days)
    rows.append(pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "sku_id": sku_col,
        "units_sold": units_final,
        "price": price,
        "promotion_flag": promotion_flag,
    }))

# Concatenate, sort and save
df = pd.concat(rows, ignore_index=True)
df = df.sort_values(["date", "sku_id"]).reset_index(drop=True)

# Sanity checks
assert df["date"].is_unique is False  # there are multiple SKUs per date
# Check no missing dates per SKU
for sku in SKUS:
    dcount = df[df["sku_id"] == sku]["date"].nunique()
    if dcount != n_days:
        raise RuntimeError(f"Missing dates for {sku}: {dcount} != {n_days}")

# Save CSV
out_file = "ecommerce_demand.csv"
df.to_csv(out_file, index=False)

# Output a quick preview
print(df.head(5).to_string(index=False))
print(f"\nDataset shape: {df.shape} (rows, cols)")
print(f"Saved as: {out_file}")

# Also print promotion coverage per SKU for transparency
promo_summary = df.groupby("sku_id")["promotion_flag"].mean().rename("promo_fraction")
print("\nPromotion fraction per SKU:\n", promo_summary.to_string())

# End of script
