#!/usr/bin/env python3
# file: make_events_private_small.py
"""
Many-to-many preserving event generator with privacy & size controls.

- Select users who interacted with popular (shared) products.
- For each user, keep up to MAX_PRODUCTS_PER_USER products, preferring items shared across users.
- Optionally cap total users (GLOBAL_USER_CAP).
- Optionally enforce a minimum number of users per product in the final subset.
- Event timestamps are randomized into the last N days (no original timestamps leaked).
- Output sorted by (userId, timestamp, eventType).

Input CSV must include:
  Transaction_ID, Customer_ID, Product_ID, Transaction_Date, Category

Output columns:
  userId, promotionId, categoryName, eventType, timestamp, source
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# --------------------------- User Configuration --------------------------- #
SRC = "data/synthetic_ecommerce_data.csv"   # input transactions
OUT = "data/synthetic_events.csv"           # output events

# Privacy / size knobs
MAX_PRODUCTS_PER_USER = 50                   # max promotions per user
FILTER_USERS_MIN_PRODUCTS = 1               # require at least this many distinct products to include a user (before caps)
GLOBAL_USER_CAP = 200                       # keep only top-N overlap-heavy users; None = keep all
GLOBAL_PAIR_CAP = None                      # optional final safety cap on (user,product) pairs; None = no cap

# Overlap rules
MIN_USERS_PER_PRODUCT = 2                   # enforce many-to-many: each product must link to >= this many users
ENFORCE_MIN_USERS_PER_PRODUCT = True        # drop products (and their pairs) that don't meet the threshold

# Event timing
LAST_N_DAYS = 30                            # randomize events into the last N days (from NOW)
SEED = 42

# Engagement ranges (inclusive)
VIEWS_PER_PURCHASE: Tuple[int, int] = (6, 14)
CLICKS_PER_PURCHASE: Tuple[int, int] = (3, 9)

# Time windows BEFORE redeem (in minutes)
VIEW_WINDOW_MINUTES: Tuple[int, int] = (60, 14 * 24 * 60)   # 1 hour .. 14 days before
CLICK_WINDOW_MINUTES: Tuple[int, int] = (5, int(36 * 60))   # 5 minutes .. 36 hours before
# ------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Config:
    src: Path
    out: Path
    last_n_days: int
    seed: int

def _random_recent_time(now: pd.Timestamp, last_n_days: int, rng: np.random.Generator) -> pd.Timestamp:
    offset_days = int(rng.integers(0, last_n_days + 1))
    offset_secs = int(rng.integers(0, 24 * 3600))
    return now - pd.Timedelta(days=offset_days, seconds=offset_secs)

def _iso(ts: pd.Timestamp) -> str:
    return pd.to_datetime(ts).isoformat()

def _build_pairs_overlap_aware(
    df: pd.DataFrame,
    max_products_per_user: int,
    min_products_per_user: int,
    global_user_cap: Optional[int],
    min_users_per_product: int,
    enforce_min_users_per_product: bool,
    global_pair_cap: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """
    Build DISTINCT (Customer_ID, Product_ID, Category, Transaction_Date) pairs from full history,
    then choose users & products to preserve many-to-many overlap.

    Steps:
      1) Distinct pairs from all history (keep most recent txn per pair).
      2) Compute product popularity = #unique users per product.
      3) Rank users by 'overlap score' = sum(popularity of their products), tie-break by last_seen.
      4) Keep top GLOBAL_USER_CAP users (if set).
      5) For each kept user, select up to MAX_PRODUCTS_PER_USER products with highest popularity (then recency).
      6) Optionally enforce that products in the final subset have >= MIN_USERS_PER_PRODUCT users.
      7) Optional GLOBAL_PAIR_CAP (fair, per-user).

    Returns: DataFrame with columns [Customer_ID, Product_ID, Category, Transaction_Date]
    """
    # 1) Distinct pairs with most recent occurrence
    pairs_all = (
        df.sort_values(["Customer_ID", "Transaction_Date"], ascending=[True, False])
          .drop_duplicates(["Customer_ID", "Product_ID"], keep="first")
          .copy()
    )

    # Filter out users with too few distinct products (pre-cap)
    user_counts = pairs_all.groupby("Customer_ID")["Product_ID"].nunique()
    keep_users = user_counts[user_counts >= max(1, min_products_per_user)].index
    pairs_all = pairs_all[pairs_all["Customer_ID"].isin(keep_users)]
    if pairs_all.empty:
        return pairs_all

    # 2) Product popularity (how many users share this product)
    users_per_prod = (
        pairs_all.groupby("Product_ID")["Customer_ID"].nunique().rename("users_per_prod")
    )

    # 3) User overlap score = sum over their products of users_per_prod
    tmp = pairs_all.merge(users_per_prod, on="Product_ID", how="left")
    user_overlap = tmp.groupby("Customer_ID")["users_per_prod"].sum().rename("overlap_score")

    # last seen per user for tie-break
    last_seen = df.groupby("Customer_ID")["Transaction_Date"].max().rename("last_seen")

    user_rank = (
        pd.concat([user_overlap, user_counts.rename("n_distinct"), last_seen], axis=1)
          .sort_values(["overlap_score", "n_distinct", "last_seen"], ascending=[False, False, False])
    )

    # 4) Cap users (keep the overlap-heavy ones)
    if global_user_cap is not None:
        if global_user_cap <= 0:
            raise ValueError("GLOBAL_USER_CAP must be positive or None.")
        rng = np.random.default_rng(seed)
        if len(user_rank) > global_user_cap:
            # slight deterministic jitter for stable tie-breaking
            jitter = pd.Series(rng.random(len(user_rank)), index=user_rank.index, name="jitter")
            user_rank = user_rank.join(jitter).sort_values(
                ["overlap_score", "n_distinct", "last_seen", "jitter"],
                ascending=[False, False, False, False]
            )
            keep_users2 = user_rank.head(global_user_cap).index
        else:
            keep_users2 = user_rank.index
    else:
        keep_users2 = user_rank.index

    pairs_kept_users = pairs_all[pairs_all["Customer_ID"].isin(keep_users2)]
    if pairs_kept_users.empty:
        return pairs_kept_users

    # 5) For each user, pick up to MAX_PRODUCTS_PER_USER with highest product popularity (then recency)
    pairs_scored = pairs_kept_users.merge(users_per_prod, on="Product_ID", how="left")
    pairs_scored = pairs_scored.sort_values(
        ["Customer_ID", "users_per_prod", "Transaction_Date"],
        ascending=[True, False, False]
    )
    if max_products_per_user is not None:
        pairs_scored = (
            pairs_scored.groupby("Customer_ID", group_keys=False)
                        .head(max_products_per_user)
        )

    # 6) Enforce minimum users per product in the final subset (many-to-many guarantee)
    if enforce_min_users_per_product:
        # Keep only products that appear with >= min_users_per_product users in the selected set
        final_u_per_p = pairs_scored.groupby("Product_ID")["Customer_ID"].nunique()
        good_products = final_u_per_p[final_u_per_p >= max(1, min_users_per_product)].index
        pairs_scored = pairs_scored[pairs_scored["Product_ID"].isin(good_products)]

        # (Optional) If some users now have 0 products due to this filter, drop them
        if not pairs_scored.empty:
            valid_users = pairs_scored["Customer_ID"].unique()
            pairs_scored = pairs_scored[pairs_scored["Customer_ID"].isin(valid_users)]

    # 7) Optional global pair cap (fair per-user)
    if global_pair_cap is not None and not pairs_scored.empty:
        n_users = max(1, pairs_scored["Customer_ID"].nunique())
        per_user_target = max(1, global_pair_cap // n_users)
        pairs_scored = (
            pairs_scored.sort_values(["Customer_ID", "users_per_prod", "Transaction_Date"],
                                     ascending=[True, False, False])
                        .groupby("Customer_ID", group_keys=False)
                        .head(per_user_target)
        )
        if len(pairs_scored) > global_pair_cap:
            pairs_scored = pairs_scored.sort_values("Transaction_Date", ascending=False).head(global_pair_cap).copy()

    # Return minimal columns
    cols = ["Customer_ID", "Product_ID", "Category", "Transaction_Date"]
    return pairs_scored[cols].copy()

def generate_events(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    # Load & prep
    df = pd.read_csv(cfg.src)
    required = {"Transaction_ID", "Customer_ID", "Product_ID", "Transaction_Date", "Category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {sorted(missing)}")

    df = df[["Customer_ID", "Product_ID", "Category", "Transaction_Date"]].copy()
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    df = df.dropna(subset=["Transaction_Date"])
    if df.empty:
        return pd.DataFrame(columns=["userId", "promotionId", "categoryName", "eventType", "timestamp", "source"])

    # Build many-to-many preserving pairs
    pairs = _build_pairs_overlap_aware(
        df=df,
        max_products_per_user=MAX_PRODUCTS_PER_USER,
        min_products_per_user=FILTER_USERS_MIN_PRODUCTS,
        global_user_cap=GLOBAL_USER_CAP,
        min_users_per_product=MIN_USERS_PER_PRODUCT,
        enforce_min_users_per_product=ENFORCE_MIN_USERS_PER_PRODUCT,
        global_pair_cap=GLOBAL_PAIR_CAP,
        seed=cfg.seed,
    )

    if pairs.empty:
        return pd.DataFrame(columns=["userId", "promotionId", "categoryName", "eventType", "timestamp", "source"])

    # Generate events
    rows: List[dict] = []
    now = pd.Timestamp.utcnow()

    v_low, v_high = VIEWS_PER_PURCHASE
    c_low, c_high = CLICKS_PER_PURCHASE
    vw_low, vw_high = VIEW_WINDOW_MINUTES
    cw_low, cw_high = CLICK_WINDOW_MINUTES

    for rec in pairs.itertuples(index=False):
        user = str(rec.Customer_ID)
        promo = str(rec.Product_ID)
        category = str(rec.Category) if pd.notna(rec.Category) else "Unknown"

        # Random redeem time in the last N days
        redeem_time = _random_recent_time(now, cfg.last_n_days, rng)

        # Views BEFORE redeem
        n_views = int(rng.integers(v_low, v_high + 1))
        if n_views:
            view_offsets = np.unique(rng.integers(vw_low, vw_high + 1, size=n_views))
            for minutes in sorted(view_offsets, reverse=True):
                t = redeem_time - pd.Timedelta(minutes=int(minutes))
                if t < redeem_time:
                    rows.append({
                        "userId": user,
                        "promotionId": promo,
                        "categoryName": category,
                        "eventType": "view",
                        "timestamp": _iso(t),
                        "source": "synthetic:real-blend",
                    })

        # Clicks BEFORE redeem
        n_clicks = int(rng.integers(c_low, c_high + 1))
        if n_clicks:
            click_offsets = np.unique(rng.integers(cw_low, cw_high + 1, size=n_clicks))
            for minutes in sorted(click_offsets, reverse=True):
                t = redeem_time - pd.Timedelta(minutes=int(minutes))
                if t < redeem_time:
                    rows.append({
                        "userId": user,
                        "promotionId": promo,
                        "categoryName": category,
                        "eventType": "click",
                        "timestamp": _iso(t),
                        "source": "synthetic:real-blend",
                    })

        # Redeem LAST
        rows.append({
            "userId": user,
            "promotionId": promo,
            "categoryName": category,
            "eventType": "redeem",
            "timestamp": _iso(redeem_time),
            "source": "synthetic:real-blend",
        })

    out_df = pd.DataFrame.from_records(rows, columns=[
        "userId", "promotionId", "categoryName", "eventType", "timestamp", "source"
    ])

    # Sort by user, then time and type
    if not out_df.empty:
        out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], errors="coerce")
        out_df.sort_values(["userId", "timestamp", "eventType"], inplace=True, kind="stable")
        out_df.reset_index(drop=True, inplace=True)

    return out_df

def main() -> None:
    cfg = Config(src=Path(SRC), out=Path(OUT), last_n_days=LAST_N_DAYS, seed=SEED)
    out_df = generate_events(cfg)

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(cfg.out, index=False)

    if not out_df.empty:
        redeems = out_df[out_df.eventType == "redeem"]
        per_user = redeems.groupby("userId")["promotionId"].nunique()
        users_per_prod = redeems.groupby("promotionId")["userId"].nunique()

        print(f"[OK] Wrote → {cfg.out}")
        print(f"rows={len(out_df)} users={out_df['userId'].nunique()} promos={out_df['promotionId'].nunique()}")
        print(f"avg distinct promotions per user (redeems): {per_user.mean():.2f}")
        print(f"avg users per promotion (redeems): {users_per_prod.mean():.2f}")
        if ENFORCE_MIN_USERS_PER_PRODUCT:
            print(f"promotions with >= {MIN_USERS_PER_PRODUCT} users: {(users_per_prod >= MIN_USERS_PER_PRODUCT).mean():.2%}")
        print(f"time range: {out_df['timestamp'].min()} → {out_df['timestamp'].max()}")
    else:
        print("[WARN] No events generated (check input and filters).")

if __name__ == "__main__":
    main()
