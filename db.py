# db.py
from dotenv import load_dotenv
import os
from motor.motor_asyncio import AsyncIOMotorClient


load_dotenv()

# Prefer MONGODB_URL if present (same as your Node app), else fallback
MONGO_URI = os.getenv("MONGODB_URL") or os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB", "discount_card")

_client = AsyncIOMotorClient(MONGO_URI)
_db = _client[MONGO_DB]

def get_db():
    return _db
