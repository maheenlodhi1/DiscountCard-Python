from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import Optional, Any, Dict, List, Literal
from datetime import datetime, timedelta, date
import numpy as np, os, pickle, asyncio, json
from bson import ObjectId
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime, timedelta, date, timezone
import sys, subprocess, logging

logger = logging.getLogger("retrain")
logger.setLevel(logging.INFO)

from db import get_db
from mongo_utils import get_promotions_by_ids, to_object_ids

load_dotenv()

# -------- env --------
MONGO_DB        = os.getenv("MONGO_DB", "discount_card")
EVENTS_COL      = os.getenv("EVENTS_COLLECTION", "promotionevents")

# Try common recommender artifact names; first that exists wins
_RECO_CANDIDATES = [
    os.getenv("RECO_MODEL_PATH", "models/lightfm_model.pkl"),
    "models/lightfm_model.pkl",
    "models/model.pkl",
]
TREND_GLOBAL    = os.getenv("TRENDING_TOPK_PATH", "models/trending_topk.json")
TREND_PER_CAT   = os.getenv("TRENDING_PER_CAT_PATH", "models/trending_topk_per_category.json")
TRENDING_DAYS   = int(os.getenv("TRENDING_DAYS", "30"))

app = FastAPI(title="DiscountCard AI APIs")

# -------- runtime state --------
_reco_lock = asyncio.Lock()
_reco = {"loaded": False, "model": None, "users": [], "items": [], "mtime": None, "path": None}
_trend = {"loaded": False, "global_ids": [], "per_cat": {}}

# ---------- utils: make any Mongo doc JSON-safe ----------
def clean_json(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, list):
        return [clean_json(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out["promotionId" if k == "_id" else k] = clean_json(v)
        return out
    return obj

def _normalize_ids(seq) -> List[str]:
    out: List[str] = []
    if not isinstance(seq, (list, tuple)):
        return out
    for x in seq:
        sid = None
        if isinstance(x, dict):
            if "promotionId" in x:
                sid = str(x["promotionId"])
            elif "_id" in x:
                v = x["_id"]
                sid = str(v.get("$oid")) if isinstance(v, dict) and "$oid" in v else str(v)
        else:
            sid = str(x)
        if sid:
            out.append(sid)
    return out

def _load_json(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _try_load_trending():
    global_ids: List[str] = []
    per_cat: Dict[str, List[str]] = {}

    data = _load_json(TREND_GLOBAL)
    if data is not None:
        if isinstance(data, list):
            global_ids = _normalize_ids(data)
        elif isinstance(data, dict):
            for key in ("topk", "top_k", "global", "ids", "top_ids", "top"):
                if key in data:
                    global_ids = _normalize_ids(data[key])
                    break

    data_pc = _load_json(TREND_PER_CAT)
    # train_trending.py saves a *flat list* of records; we take Top-M per category already computed.
    # If you instead store a mapping {categoryName: [ids...]}, this handles that too.
    if isinstance(data_pc, dict):
        per_cat = {k: _normalize_ids(v) for k, v in data_pc.items() if isinstance(v, (list, tuple))}
    elif isinstance(data_pc, list):
        # Accept list of {promotionId, categoryName, prob_trending}
        per_cat_tmp: Dict[str, List[str]] = {}
        for row in data_pc:
            if not isinstance(row, dict):
                continue
            cat = str(row.get("categoryName", "Unknown"))
            pid = str(row.get("promotionId") or row.get("_id") or "")
            if pid:
                per_cat_tmp.setdefault(cat, []).append(pid)
        per_cat = per_cat_tmp

    _trend["global_ids"] = global_ids
    _trend["per_cat"] = per_cat
    _trend["loaded"] = bool(global_ids or per_cat)

def _find_existing_reco_path() -> Optional[str]:
    for p in _RECO_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None

# ---------- models ----------
class EventType(str, Enum):
    view = "view"
    click = "click"
    redeem = "redeem"

class LogEvent(BaseModel):
    userId: str = Field(..., min_length=1)
    promotionId: str = Field(..., min_length=1)
    categoryName: Optional[str] = None
    categoryId: Optional[str] = None
    eventType: EventType
    timestamp: Optional[datetime] = None
    source: Optional[str] = "app"

    @field_validator("timestamp", mode="before")
    def parse_ts(cls, v):
        if v is None:
            return datetime.utcnow()
        if isinstance(v, datetime):
            return v
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))

# ---------- model loaders ----------
async def _try_load_recommender():
    path = _find_existing_reco_path()
    if not path:
        return
    mtime = os.path.getmtime(path)
    if _reco["loaded"] and _reco["mtime"] == mtime and _reco["path"] == path:
        return
    async with _reco_lock:
        if _reco["loaded"] and _reco["mtime"] == mtime and _reco["path"] == path:
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = data.get("model", None)
        users = data.get("users", [])
        items = data.get("items", [])
        _reco.update({
            "loaded": model is not None,
            "model": model,
            "users": list(users or []),
            "items": list(items or []),
            "mtime": mtime,
            "path": path,
        })

# -------------------- MINIMAL RETRAIN JOB ADDITION --------------------
async def _retrain_now():
    """Run your two training scripts."""
    py = sys.executable
    def _run():
        logger.info("[Retrain] Recommender (mongo)…")
        subprocess.run([py, "train_recommend.py", "--mongo"], check=True)
        logger.info("[Retrain] Trending (mongo)…")
        subprocess.run([py, "train_trending.py", "--mongo"], check=True)
        logger.info("[Retrain] Completed.")
    await asyncio.to_thread(_run)
# ---------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    await _try_load_recommender()
    _try_load_trending()

    # -------------------- MINIMAL SCHEDULER SETUP --------------------
    # Run once immediately, then every 12 hours thereafter.
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        _retrain_now,
        "interval",
        hours=12,
        id="retrain_job",
        coalesce=True,
        max_instances=1,
        next_run_time=datetime.utcnow(),  # immediate first run
    )
    scheduler.start()
    app.state.scheduler = scheduler
    # ----------------------------------------------------------------

# ---------- endpoints ----------
@app.get("/health")
async def health():
    return {
        "ok": True,
        "recommender_loaded": _reco["loaded"],
        "users_count": len(_reco["users"] or []),
        "items_count": len(_reco["items"] or []),
        "reco_model_path": _reco["path"],
        "reco_model_mtime": _reco["mtime"],
        "trending_loaded": _trend["loaded"],
        "trending_global_count": len(_trend["global_ids"] or []),
        "trending_per_cat_keys": sorted((_trend["per_cat"] or {}).keys()),
    }

@app.post("/log_event")
async def log_event(ev: LogEvent):
    db = get_db()
    doc = ev.model_dump()
    try:
        doc["userId"] = ObjectId(doc["userId"])
        doc["promotionId"] = ObjectId(doc["promotionId"])
        if doc.get("categoryId"):
            doc["categoryId"] = ObjectId(doc["categoryId"])
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid_object_id"})
    await db[EVENTS_COL].insert_one(doc)
    return {"status": "ok"}

@app.get("/trending_deals")
async def trending_deals(
    top_n: int = Query(10, ge=1, le=100),
    categoryName: Optional[str] = None,
    days: Optional[int] = Query(None, ge=1, le=365),  # kept for fallback only
):
    """Serve model-based trending if present; otherwise fallback to last-N-day redeems in Mongo."""
    # 1) Model artifacts
    if _trend["loaded"]:
        if categoryName and categoryName in _trend["per_cat"]:
            ids_str = _trend["per_cat"][categoryName][:top_n]
        else:
            ids_str = _trend["global_ids"][:top_n]
        if ids_str:
            promos = await get_promotions_by_ids(to_object_ids(ids_str))
            pm = {str(p["_id"]): p for p in promos}
            ordered = [pm.get(pid) for pid in ids_str if pid in pm]
            return {"trending_deals": [clean_json(p) for p in ordered if p]}

    # 2) Fallback → last-N-days top redeems
    db = get_db()
    window_days = days or TRENDING_DAYS
    since = datetime.utcnow() - timedelta(days=window_days)
    match: Dict[str, Any] = {"eventType": "redeem", "timestamp": {"$gte": since}}
    if categoryName:
        match["categoryName"] = categoryName
    pipeline = [
        {"$match": match},
        {"$group": {"_id": "$promotionId", "cnt": {"$sum": 1}}},
        {"$sort": {"cnt": -1}},
        {"$limit": top_n},
    ]
    rows = await db[EVENTS_COL].aggregate(pipeline).to_list(length=top_n)
    top_ids_obj = [r["_id"] for r in rows]
    if not top_ids_obj:
        return {"trending_deals": []}
    promos = await get_promotions_by_ids(top_ids_obj)
    pm = {str(p["_id"]): p for p in promos}
    ordered = [pm.get(str(pid)) for pid in top_ids_obj if str(pid) in pm]
    return {"trending_deals": [clean_json(p) for p in ordered if p]}


@app.get("/recommended_deals/{user_id}")
async def recommended_deals(
    user_id: str,
    top_n: int = Query(10, ge=1, le=100),
):
    # Ensure recommender is loaded
    await _try_load_recommender()
    if not _reco.get("loaded") or not _reco.get("users") or not _reco.get("items"):
        return {"recommended_deals": None, "note": "model_not_loaded"}

    users = _reco["users"]
    items = _reco["items"]
    model = _reco["model"]

    # User must exist in trained mapping
    if user_id not in users:
        return {"recommended_deals": None, "note": "cold_start_user"}

    if not items:
        return {"recommended_deals": None, "note": "no_items_in_model"}

    # Score all items for this user
    uidx = users.index(user_id)
    scores = model.predict(uidx, np.arange(len(items)))
    if scores is None or len(scores) == 0:
        return {"recommended_deals": None, "note": "no_scores"}

    # Top-N ids, in order
    top_idx = np.argsort(-scores)[:top_n]
    top_ids_str = [items[i] for i in top_idx]
    if not top_ids_str:
        return {"recommended_deals": None, "note": "no_recommendations"}

    # Fetch promos from Mongo in the same order
    promos = await get_promotions_by_ids(to_object_ids(top_ids_str))
    if not promos:
        return {"recommended_deals": None, "note": "no_promotions_found"}

    pm = {str(p["_id"]): p for p in promos}
    ordered = [pm.get(pid) for pid in top_ids_str if pid in pm]

    # If nothing matched back from DB, return null
    if not any(ordered):
        return {"recommended_deals": None, "note": "db_lookup_empty"}

    return {"recommended_deals": [clean_json(p) for p in ordered if p]}


@app.post("/admin/retrain")
async def manual_retrain():
    """
    Manually trigger the retraining job.
    Returns immediately after queuing the task.
    """
    asyncio.create_task(_retrain_now())
    return {"status": "queued", "message": "Retraining started in background"}

@app.get("/events")
async def get_events(
    limit: int = Query(50, ge=1, le=500),
    skip: int = Query(0, ge=0)
):
    """
    Get all events from MongoDB with pagination.
    """
    db = get_db()
    cursor = db[EVENTS_COL].find().skip(skip).limit(limit)
    docs = await cursor.to_list(length=limit)

    return {
        "ok": True,
        "count": len(docs),
        "skip": skip,
        "limit": limit,
        "items": [clean_json(d) for d in docs],
    }


@app.exception_handler(Exception)
async def default_handler(_, __):
    return JSONResponse(status_code=500, content={"error": "internal_error"})
