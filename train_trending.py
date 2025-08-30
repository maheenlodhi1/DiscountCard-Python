#!/usr/bin/env python3
"""
Train XGBoost trending model on last N days of events.
Exports:
  - Model + meta
  - Top-K predictions (overall + per-category)
  - Metrics JSON
  - Plots: ROC, PR, Confusion Matrix (counts + normalized)
"""

import argparse, os, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)

# ---------------- Paths / Config ---------------- #
MODEL_PATH = "models/xgb_trending_model.json"
META_PATH  = "models/xgb_trending_meta.json"
TOPK_PATH  = "models/trending_topk.json"
ART_DIR    = Path("artifacts")

FEATURE_COLUMNS = [
    "view", "click", "redeem", "click_per_view", "redeem_per_click", "redeem_per_view",
    "log_view", "log_click", "log_redeem", "total_count", "pop_percentile",
    "unique_users", "category", "age_days", "recency_hours",
    "view_per_day", "click_per_day", "redeem_per_day", "d_view", "d_click", "d_redeem"
]

# ---------------- Utilities ---------------- #
def _ensure_art_dir():
    ART_DIR.mkdir(parents=True, exist_ok=True)

def _savefig_safely(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"[SAVED] {path.resolve()}")

def _placeholder_plot(title: str, note: str, outpath: Path):
    """Create a simple placeholder figure so artifacts always exist."""
    try:
        _ensure_art_dir()
        plt.figure(figsize=(6, 4))
        plt.axis("off")
        plt.title(title)
        plt.text(0.5, 0.5, note, ha="center", va="center", wrap=True)
        plt.tight_layout()
        _savefig_safely(outpath)
    except Exception as e:
        print(f"[PLOT-PLACEHOLDER-ERROR] {e}", file=sys.stderr)

def _safe_list_artifacts():
    try:
        print(f"[ARTIFACTS DIR] {ART_DIR.resolve()}")
        for p in sorted(ART_DIR.glob("*")):
            print(" -", p.name)
    except Exception as e:
        print(f"[LIST-ERROR] {e}", file=sys.stderr)

# ---------------- Data ---------------- #
def load_events(source):
    if source == "synthetic":
        df = pd.read_csv("data/synthetic_events.csv", parse_dates=["timestamp"])
    elif source == "mongo":
        from get_events_from_mongodb import fetch_events
        df = fetch_events()
    else:
        raise ValueError("Invalid source")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df.dropna(subset=["timestamp"])

def filter_last_n_days(df, days):
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    filtered = df[df["timestamp"] >= cutoff].copy()
    print(f"[INFO] Last {days} days: {len(filtered)}/{len(df)} events")
    return filtered

# ---------------- Features ---------------- #
def safe_div(a, b): 
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return np.where(b > 0, a / b, 0.0)

def engineer_features(df, decay_half_life=7.0):
    df = df.assign(
        promotionId=df["promotionId"].astype(str),
        userId=df["userId"].astype(str),
        categoryName=df["categoryName"].fillna("Unknown").astype(str),
        eventType=df["eventType"].astype(str).str.lower()
    )
    # Counts
    counts = df.pivot_table(index="promotionId", columns="eventType",
                            values="userId", aggfunc="count", fill_value=0).reset_index()
    for col in ("view", "click", "redeem"):
        counts[col] = counts.get(col, 0)

    # Category mapping
    cat_map = df.groupby("promotionId")["categoryName"].agg(lambda x: x.mode().iat[0]).to_dict()
    counts["categoryName"] = counts["promotionId"].map(cat_map)
    cats = counts["categoryName"].astype("category")
    counts["category"] = cats.cat.codes
    counts.attrs["category_mapping"] = dict(enumerate(cats.cat.categories))

    # Unique users
    counts["unique_users"] = counts["promotionId"].map(
        df.groupby("promotionId")["userId"].nunique()
    ).fillna(0).astype(int)

    # Ratios, logs, totals
    counts["click_per_view"]   = safe_div(counts["click"], counts["view"])
    counts["redeem_per_click"] = safe_div(counts["redeem"], counts["click"])
    counts["redeem_per_view"]  = safe_div(counts["redeem"], counts["view"])
    for col in ("view", "click", "redeem"):
        counts[f"log_{col}"] = np.log1p(counts[col])
    counts["total_count"] = counts[["view", "click", "redeem"]].sum(axis=1)
    counts["pop_percentile"] = counts["total_count"].rank(pct=True)

    # Time features
    ts_min = df.groupby("promotionId")["timestamp"].min()
    ts_max = df.groupby("promotionId")["timestamp"].max()
    now = df["timestamp"].max()
    counts["first_ts"] = counts["promotionId"].map(ts_min)
    counts["last_ts"]  = counts["promotionId"].map(ts_max)
    counts["age_days"] = ((now - counts["first_ts"]).dt.total_seconds() / 86400).fillna(0)
    counts["recency_hours"] = ((now - counts["last_ts"]).dt.total_seconds() / 3600).fillna(0)
    span_days = np.maximum(counts["age_days"], 1.0)
    counts["view_per_day"]   = counts["view"] / span_days
    counts["click_per_day"]  = counts["click"] / span_days
    counts["redeem_per_day"] = counts["redeem"] / span_days

    # Exponential decay sums
    lam = np.log(2.0) / max(decay_half_life, 1e-6)
    df["_decay"] = np.exp(-lam * (now - df["timestamp"]).dt.total_seconds() / 86400)
    decay = df.pivot_table(index="promotionId", columns="eventType",
                           values="_decay", aggfunc="sum", fill_value=0).reset_index()
    for col in ("view", "click", "redeem"):
        decay[col] = decay.get(col, 0.0)
    counts = counts.merge(
        decay.rename(columns={"view":"d_view","click":"d_click","redeem":"d_redeem"}),
        on="promotionId", how="left"
    )
    return counts

# ---------------- Labels ---------------- #
def wilson_lower_bound(successes, trials, z=1.96):
    successes = np.asarray(successes, dtype=float); trials = np.asarray(trials, dtype=float)
    p = np.where(trials > 0, successes / trials, 0.0)
    denom  = 1 + (z*z)/np.maximum(trials, 1)
    center = p + (z*z)/(2*np.maximum(trials,1))
    margin = z * np.sqrt((p*(1-p) + (z*z)/(4*np.maximum(trials,1))) / np.maximum(trials,1))
    return np.where(trials > 0, (center - margin) / denom, 0.0)

def make_labels(feats):
    d_view_norm = feats["d_view"] / (feats["d_view"].max() or 1.0)
    ctr = wilson_lower_bound(feats["click"], feats["view"])
    cvr = wilson_lower_bound(feats["redeem"], feats["click"])
    score = 0.5*d_view_norm + 2.0*ctr + 3.0*cvr
    if np.all(score <= 0):
        score = feats["view"]
    thresh = np.nanquantile(score, 0.8 if len(feats) >= 200 else 0.6)
    y = (score >= thresh).astype(int)
    if y.sum() < (20 if len(feats) >= 200 else 2):
        y[:] = 0
        y.iloc[np.argsort(-score)[:(20 if len(feats) >= 200 else 2)]] = 1
    return y

# ---------------- Model / Splits / Eval ---------------- #
def make_model(spw):
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        n_jobs=4, eval_metric="aucpr", scale_pos_weight=spw,
        random_state=42, tree_method="hist"
    )

def chronological_indices(feats, train_frac=0.7, val_frac=0.1):
    order = feats.sort_values("last_ts").index.to_numpy()
    n = len(order)
    i_tr = int(round(train_frac * n))
    i_val = int(round((train_frac + val_frac) * n))
    return order[:i_tr], order[i_tr:i_val], order[i_val:]

def choose_threshold_on_val(y_val, p_val):
    if len(y_val) == 0:
        return 0.5, 0.0, 0.0, 0.0
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    if len(prec) <= 1:
        return 0.5, float(prec[0]), float(rec[0]), 0.0
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    if len(f1) == 0 or np.all(np.isnan(f1)):
        return 0.5, float(prec[0]), float(rec[0]), 0.0
    best = int(np.nanargmax(f1))
    return float(thr[best]) if len(thr) else 0.5, float(prec[best]), float(rec[best]), float(f1[best])

def _plot_confusions(cm, prefix):
    # counts
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-trending (0)", "Trending (1)"]).plot(
            cmap="Blues", ax=ax, values_format="d"  # no colorbar arg for broad compat
        )
        ax.set_title("XGBoost Trending — Confusion Matrix (counts)")
        plt.tight_layout()
        _savefig_safely(ART_DIR / f"{prefix}_confusion_counts.png")
    except Exception as e:
        print(f"[PLOT-COUNTS-ERROR] {e}", file=sys.stderr)
        _placeholder_plot("Confusion Matrix (counts)", f"Plot error: {e}", ART_DIR / f"{prefix}_confusion_counts.png")

    # normalized (safe divide)
    try:
        row_sums = cm.sum(axis=1, keepdims=True)
        cmn = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix=cmn, display_labels=["Non-trending (0)", "Trending (1)"]).plot(
            cmap="Blues", ax=ax, values_format=".2f"
        )
        ax.set_title("XGBoost Trending — Confusion Matrix (normalized)")
        plt.tight_layout()
        _savefig_safely(ART_DIR / f"{prefix}_confusion_norm.png")
    except Exception as e:
        print(f"[PLOT-NORM-ERROR] {e}", file=sys.stderr)
        _placeholder_plot("Confusion Matrix (normalized)", f"Plot error: {e}", ART_DIR / f"{prefix}_confusion_norm.png")

def evaluate_and_plots(y_true, p_prob, threshold, prefix="trending"):
    """Robust evaluation + artifact plots that always write files."""
    _ensure_art_dir()
    y_true = np.asarray(y_true).astype(int)
    p_prob = np.asarray(p_prob).astype(float)
    y_pred = (p_prob >= threshold).astype(int)

    n = len(y_true)
    has_data = n > 0
    has_both = has_data and (len(np.unique(y_true)) == 2)
    has_pos = has_data and (np.sum(y_true) > 0)

    # Basic metrics (safe)
    acc  = accuracy_score(y_true, y_pred) if has_data else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0) if has_data else 0.0
    rec  = recall_score(y_true, y_pred, zero_division=0) if has_data else 0.0
    f1   = f1_score(y_true, y_pred, zero_division=0) if has_data else 0.0

    # Confusion matrix (2x2)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) if has_data else np.array([[0,0],[0,0]])
    _plot_confusions(cm, prefix)

    # ROC
    roc_path = ART_DIR / f"{prefix}_roc.png"
    if has_both:
        try:
            fpr, tpr, _ = roc_curve(y_true, p_prob)
            roc_auc = float(roc_auc_score(y_true, p_prob))
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], '--', lw=0.8)
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("ROC — XGBoost Trending")
            plt.legend(); plt.tight_layout()
            _savefig_safely(roc_path)
        except Exception as e:
            print(f"[ROC-ERROR] {e}", file=sys.stderr)
            _placeholder_plot("ROC — XGBoost Trending", f"Plot error: {e}", roc_path)
            roc_auc = float("nan")
    else:
        roc_auc = float("nan")
        _placeholder_plot("ROC — XGBoost Trending", "ROC unavailable: need both classes in y_true.", roc_path)

    # PR
    pr_path = ART_DIR / f"{prefix}_pr.png"
    if has_pos:
        try:
            precision_arr, recall_arr, _ = precision_recall_curve(y_true, p_prob)
            pr_auc = float(average_precision_score(y_true, p_prob))
            plt.figure()
            plt.plot(recall_arr, precision_arr, label=f"AP={pr_auc:.3f}")
            plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.title("Precision–Recall — XGBoost Trending")
            plt.legend(); plt.tight_layout()
            _savefig_safely(pr_path)
        except Exception as e:
            print(f"[PR-ERROR] {e}", file=sys.stderr)
            _placeholder_plot("Precision–Recall — XGBoost Trending", f"Plot error: {e}", pr_path)
            pr_auc = float("nan")
    else:
        pr_auc = float("nan")
        _placeholder_plot("Precision–Recall — XGBoost Trending", "PR unavailable: no positive samples in y_true.", pr_path)

    # Write metrics JSON
    tn, fp, fn, tp = (cm.ravel().tolist() if cm.size == 4 else [0,0,0,0])
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": None if np.isnan(roc_auc) else float(roc_auc),
        "pr_auc":  None if (isinstance(pr_auc, float) and np.isnan(pr_auc)) else float(pr_auc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "class_prevalence": float(np.mean(y_true)) if has_data else 0.0,
        "n_samples": int(n)
    }
    with open(ART_DIR / f"{prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[METRICS] {metrics}")
    return y_pred, metrics

def train_with_holdout(feats, labels):
    X = feats[FEATURE_COLUMNS].astype(float)
    y = labels.astype(int)

    idx_tr, idx_val, idx_te = chronological_indices(feats, train_frac=0.7, val_frac=0.1)
    X_tr, y_tr = X.loc[idx_tr], y.loc[idx_tr]
    X_val, y_val = X.loc[idx_val], y.loc[idx_val]
    X_te,  y_te  = X.loc[idx_te],  y.loc[idx_te]

    pos, neg = int(y_tr.sum()), int(len(y_tr) - y_tr.sum())
    spw = (neg / max(pos, 1)) if pos > 0 else 1.0

    model = make_model(spw)
    model.fit(X_tr, y_tr)

    p_val = model.predict_proba(X_val)[:,1] if len(X_val) else np.zeros(len(y_val))
    thr, p_prec, p_rec, p_f1 = choose_threshold_on_val(y_val, p_val)
    print(f"[VAL] threshold={thr:.4f}  P={p_prec:.3f} R={p_rec:.3f} F1={p_f1:.3f}")

    p_te = model.predict_proba(X_te)[:,1] if len(X_te) else np.zeros(len(y_te))
    _pred_test, test_metrics = evaluate_and_plots(y_te, p_te, thr, prefix="trending")

    pos_all, neg_all = int(y.sum()), int(len(y) - y.sum())
    model_final = make_model((neg_all / max(pos_all,1)) if pos_all > 0 else 1.0)
    model_final.fit(X, y)

    return model_final, thr, test_metrics

def train_no_holdout(feats, labels):
    X = feats[FEATURE_COLUMNS].astype(float)
    y = labels.astype(int)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    spw = (neg / max(pos,1)) if pos > 0 else 1.0
    model = make_model(spw)
    model.fit(X, y)
    # Evaluate on train to emit plots
    p = model.predict_proba(X)[:,1]
    _pred_train, _ = evaluate_and_plots(y_true=y, p_prob=p, threshold=0.5, prefix="trending_train")
    return model, 0.5, None

# ---------------- Save ---------------- #
def save_outputs(model, feats, topk, per_cat_m, window_days, decay_half_life):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    _ensure_art_dir()

    model.save_model(MODEL_PATH)
    print(f"[SAVED] {Path(MODEL_PATH).resolve()}")

    meta = {
        "features": FEATURE_COLUMNS,
        "category_mapping": feats.attrs.get("category_mapping", {}),
        "cutoff_utc": pd.Timestamp.utcnow().isoformat(),
        "window_days": window_days,
        "decay_half_life_days": decay_half_life,
        "n_promotions": int(len(feats))
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[SAVED] {Path(META_PATH).resolve()}")

    # Top-K predictions
    probs = model.predict_proba(feats[FEATURE_COLUMNS].astype(float))[:,1]
    out = feats[["promotionId", "categoryName"]].assign(prob_trending=probs)\
            .sort_values("prob_trending", ascending=False)

    topk_df = out.head(max(topk, 20) if len(out) >= 200 else topk)
    with open(TOPK_PATH, "w") as f:
        json.dump(topk_df.to_dict(orient="records"), f, default=str)
    print(f"[SAVED] {Path(TOPK_PATH).resolve()}")

    per_cat = out.groupby("categoryName", group_keys=False).head(per_cat_m)
    with open(TOPK_PATH.replace(".json", "_per_category.json"), "w") as f:
        json.dump(per_cat.to_dict(orient="records"), f, default=str)
    print(f"[SAVED] {Path(TOPK_PATH.replace('.json','_per_category.json')).resolve()}")

# ---------------- Main ---------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--mongo", action="store_true")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--per-category-m", type=int, default=5)
    ap.add_argument("--decay-half-life", type=float, default=7.0)
    args = ap.parse_args()

    print("[CWD]", Path.cwd().resolve())

    source = "mongo" if args.mongo else "synthetic"
    df = filter_last_n_days(load_events(source), args.days)
    if df.empty:
        _placeholder_plot("Confusion Matrix (counts)", "No recent events → no data.", ART_DIR / "trending_confusion_counts.png")
        _placeholder_plot("Confusion Matrix (normalized)", "No recent events → no data.", ART_DIR / "trending_confusion_norm.png")
        _placeholder_plot("ROC — XGBoost Trending", "No recent events → no data.", ART_DIR / "trending_roc.png")
        _placeholder_plot("Precision–Recall — XGBoost Trending", "No recent events → no data.", ART_DIR / "trending_pr.png")
        _safe_list_artifacts()
        raise ValueError("No recent events.")

    feats = engineer_features(df, args.decay_half_life)
    labels = make_labels(feats)

    # choose pathway
    can_holdout = labels.nunique() == 2 and labels.value_counts().min() >= 5 and len(labels) >= 60
    if can_holdout:
        model, threshold, _ = train_with_holdout(feats, labels)
        print("[Holdout Eval] complete.")
    else:
        print("[WARN] Not enough signal for holdout; training on all samples.")
        model, threshold, _ = train_no_holdout(feats, labels)

    # Always emit a final set of artifacts on ALL data so confusion matrices exist for sure
    try:
        X_all = feats[FEATURE_COLUMNS].astype(float)
        y_all = labels.astype(int).to_numpy()
        p_all = model.predict_proba(X_all)[:, 1]
        evaluate_and_plots(y_true=y_all, p_prob=p_all, threshold=threshold, prefix="trending_all")
    except Exception as e:
        print(f"[FINAL-ARTIFACTS-ERROR] {e}", file=sys.stderr)
        _placeholder_plot("Confusion Matrix (counts)", f"Final artifacts error: {e}", ART_DIR / "trending_all_confusion_counts.png")
        _placeholder_plot("Confusion Matrix (normalized)", f"Final artifacts error: {e}", ART_DIR / "trending_all_confusion_norm.png")

    save_outputs(model, feats, args.topk, args.per_category_m, args.days, args.decay_half_life)
    _safe_list_artifacts()

if __name__ == "__main__":
    main()
