from dotenv import load_dotenv
from typing import Optional, Dict, Any
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "discount_card")
EVENTS_COL = os.getenv("EVENTS_COLLECTION", "promotionevents")

_client = MongoClient(MONGO_URI)
_db = _client[MONGO_DB]

def fetch_events(query: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    q: Dict[str, Any] = query or {}
    events = list(
        _db[EVENTS_COL].find(
            q,
            {
                "_id": 1,
                "userId": 1,
                "promotionId": 1,
                "categoryName": 1,
                "eventType": 1,
                "timestamp": 1,
                "source": 1,
            },
        )
    )

    for ev in events:
        ev["eventId"] = str(ev.pop("_id"))
        # stringify for modeling
        uid = ev.get("userId")
        pid = ev.get("promotionId")
        if isinstance(uid, ObjectId):
            ev["userId"] = str(uid)
        if isinstance(pid, ObjectId):
            ev["promotionId"] = str(pid)

    df = pd.DataFrame(events)
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df
