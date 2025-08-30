#!/usr/bin/env python3
"""
Train LightFM recommendation model with category-aware features
and evaluate using category alignment + classic metrics.
"""

import argparse, json, os, pickle
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k, reciprocal_rank

# ---------------- Config ---------------- #
DEFAULT_OUTDIR = "models"
DEFAULT_CSV = "data/synthetic_events.csv"
EVENT_WEIGHTS = {"view": 0.2, "click": 1.0, "redeem": 3.0}
HALF_LIFE_DAYS = 14
TOP_USER_CATEGORIES = 3
N_COMPONENTS, EPOCHS, LEARNING_RATE, LOSS = 64, 20, 0.05, "warp"
DEFAULT_K, DEFAULT_DAYS = 10, 30

# ---------------- Data loading ---------------- #
def load_events(source, csv_path=DEFAULT_CSV):
    if source == "synthetic":
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    else:
        from get_events_from_mongodb import fetch_events
        df = fetch_events()
    if df.empty:
        raise RuntimeError("No events loaded.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def ensure_schema(df):
    req_cols = {"userId", "promotionId", "categoryName", "eventType", "timestamp"}
    if missing := req_cols - set(df.columns):
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=["timestamp"]).copy()
    df["userId"] = df["userId"].astype(str)
    df["promotionId"] = df["promotionId"].astype(str)
    df["categoryName"] = df["categoryName"].fillna("Unknown").astype(str)
    df["eventType"] = df["eventType"].str.lower()
    return df

# ---------------- Feature engineering ---------------- #
def time_decay(ts, now, half_life_days):
    dt = (now - ts).dt.total_seconds() / 86400
    return np.exp(-np.log(2) * dt / half_life_days)

def aggregate_interactions(df):
    now = df["timestamp"].max()
    w = df["eventType"].map(EVENT_WEIGHTS).fillna(0) * time_decay(df["timestamp"], now, HALF_LIFE_DAYS)
    df = df.assign(weight=w)
    return df.groupby(["userId", "promotionId"], as_index=False)["weight"].sum()

def build_features(df):
    now = df["timestamp"].max()
    w = df["eventType"].map(EVENT_WEIGHTS).fillna(0) * time_decay(df["timestamp"], now, HALF_LIFE_DAYS)
    df = df.assign(weight=w)

    # User preferences
    user_feats, user_aff = {}, {}
    for uid, grp in df.groupby("userId"):
        top = grp.groupby("categoryName")["weight"].sum().nlargest(TOP_USER_CATEGORIES)
        user_feats[uid] = [f"u_cat:{c}" for c in top.index]
        user_aff[uid] = [{"categoryName": c, "score": float(s)} for c, s in top.items()]

    # Item features
    item_feats, item_catalog = {}, {}
    item_pop_rank = df.groupby("promotionId")["weight"].sum().rank(pct=True)
    for pid, grp in df.groupby("promotionId"):
        cat = grp["categoryName"].iloc[0]
        pop_bucket = "low" if item_pop_rank[pid] < 0.33 else "med" if item_pop_rank[pid] < 0.66 else "high"
        item_feats[pid] = [f"i_cat:{cat}", f"i_pop:{pop_bucket}"]
        item_catalog[pid] = {"categoryName": cat, "pop_bucket": pop_bucket}

    cats = sorted(df["categoryName"].unique())
    u_tokens = [f"u_cat:{c}" for c in cats]
    i_tokens = [f"i_cat:{c}" for c in cats] + [f"i_pop:{b}" for b in ["low", "med", "high"]]
    return u_tokens, i_tokens, user_feats, item_feats, user_aff, item_catalog

# ---------------- Dataset building ---------------- #
def build_dataset(df):
    agg = aggregate_interactions(df)
    users, items = agg["userId"].unique().tolist(), agg["promotionId"].unique().tolist()
    u_tokens, i_tokens, u_feats, i_feats, user_aff, item_catalog = build_features(df)

    ds = Dataset()
    ds.fit(
        users, items,
        user_features=u_tokens + [f"u_id:{u}" for u in users],
        item_features=i_tokens + [f"i_id:{i}" for i in items]
    )

    inter, weights = ds.build_interactions((u, i, w) for u, i, w in agg.itertuples(index=False))
    user_features = ds.build_user_features((u, u_feats.get(u, []) + [f"u_id:{u}"]) for u in users)
    item_features = ds.build_item_features((i, i_feats.get(i, []) + [f"i_id:{i}"]) for i in items)

    meta = {"users": users, "items": items}
    return inter, weights, user_features, item_features, meta, user_aff, item_catalog

# ---------------- Training & evaluation ---------------- #
def train_model(inter, ufeat, ifeat, weights):
    model = LightFM(loss=LOSS, no_components=N_COMPONENTS, learning_rate=LEARNING_RATE)
    model.fit(inter, user_features=ufeat, item_features=ifeat, sample_weight=weights, epochs=EPOCHS)
    return model

def category_alignment(model, meta, user_aff, item_catalog, k):
    users, items = meta["users"], meta["items"]
    item_cats = [item_catalog[i]["categoryName"] for i in items]
    hits = []
    for uid in users:
        fav_cats = [d["categoryName"] for d in user_aff.get(uid, [])]
        if not fav_cats:
            continue
        scores = model.predict(users.index(uid), np.arange(len(items)))
        top_cats = [item_cats[i] for i in np.argsort(-scores)[:k]]
        hits.append(sum(c in fav_cats for c in top_cats) / k)
    return float(np.mean(hits)) if hits else 0.0

def classic_metrics(model, inter, ufeat, ifeat, k):
    return {
        f"precision@{k}": float(precision_at_k(model, inter, k=k, user_features=ufeat, item_features=ifeat).mean()),
        f"recall@{k}": float(recall_at_k(model, inter, k=k, user_features=ufeat, item_features=ifeat).mean()),
        "mrr": float(reciprocal_rank(model, inter, user_features=ufeat, item_features=ifeat).mean())
    }

# ---------------- Main ---------------- #
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--mongo", action="store_true")
    ap.add_argument("--out-dir", default=DEFAULT_OUTDIR)
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    args = ap.parse_args()

    source = "mongo" if args.mongo else "synthetic"
    df = ensure_schema(load_events(source))
    df = df[df["timestamp"] >= pd.Timestamp.utcnow() - pd.Timedelta(days=args.days)]
    if df.empty:
        raise RuntimeError(f"No events in the last {args.days} days.")

    inter, weights, ufeat, ifeat, meta, user_aff, item_catalog = build_dataset(df)
    model = train_model(inter, ufeat, ifeat, weights)

    metrics = {
        "CategoryAlignment": category_alignment(model, meta, user_aff, item_catalog, args.k),
        "Classic": classic_metrics(model, inter, ufeat, ifeat, args.k)
    }

    # Save everything in one file for serving
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "model.pkl"), "wb") as f:
        pickle.dump({
        "model": model,
        "users": meta["users"],            # direct for API
        "items": meta["items"],            # direct for API
        "meta": meta,                      # still keep full meta
        "user_aff": user_aff,
        "item_catalog": item_catalog
        }, f)

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Print metrics
    print("\n[Evaluation Results]")
    print(f"  Category Alignment@{args.k}: {metrics['CategoryAlignment']:.4f}")
    print(f"  Precision@{args.k}: {metrics['Classic'][f'precision@{args.k}']:.4f}")
    print(f"  Recall@{args.k}: {metrics['Classic'][f'recall@{args.k}']:.4f}")
    print(f"  MRR: {metrics['Classic']['mrr']:.4f}")
    print(f"\n[OK] Model + meta saved → {os.path.join(args.out_dir, 'model.pkl')}")
    print(f"[OK] Metrics saved → {os.path.join(args.out_dir, 'metrics.json')}")

if __name__ == "__main__":
    main()
