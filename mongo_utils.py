# mongo_utils.py
from dotenv import load_dotenv
from typing import List, Any, Dict
from bson import ObjectId
from bson.errors import InvalidId
from db import get_db
import os

load_dotenv()

PROMO_COL = os.getenv("PROMO_COLLECTION", "promotions")

def to_object_ids(ids: List[Any]) -> List[ObjectId]:
    """
    Convert a list of IDs to ObjectId, skipping invalid ones.
    Accepts strings, ObjectIds, or dicts like {"$oid": "..."}.
    """
    valid: List[ObjectId] = []
    if not isinstance(ids, (list, tuple)):
        return valid

    for x in ids:
        try:
            if isinstance(x, ObjectId):
                valid.append(x)
            elif isinstance(x, dict) and "$oid" in x:
                valid.append(ObjectId(x["$oid"]))
            else:
                valid.append(ObjectId(str(x)))
        except (InvalidId, TypeError, ValueError):
            # silently skip invalid IDs like "Product_289"
            continue
    return valid

async def get_promotions_by_ids(promotion_ids: List[Any]) -> List[Dict]:
    """
    Fetch promotions by _id (ObjectId). If no valid ids, returns [].
    """
    db = get_db()
    ids = to_object_ids(promotion_ids)
    if not ids:
        return []
    # de-duplicate to keep query small
    ids = list(dict.fromkeys(ids))
    docs = await db[PROMO_COL].find({"_id": {"$in": ids}}).to_list(length=len(ids))
    return docs
