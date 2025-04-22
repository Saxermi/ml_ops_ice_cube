"""
retry_missing_pushes.py

This script checks MongoDB for archived event files that have not yet been pushed to Redis.
It reprocesses those files from the archive directory, extracts event batches, and pushes them into the Redis queue (`event_queue`).
After successful push, it updates the processing status in MongoDB.

Usage:
------
Run manually or schedule periodically to ensure all event batches are delivered to Redis.

"""

import os
import pandas as pd
import redis
import pickle
import time
from pymongo import MongoClient

ARCHIVE_DIR = "../../archive"
r = redis.Redis(host='localhost', port=6379, db=0)
mongo = MongoClient("mongodb://localhost:27017/")
mdb = mongo["icecube_db"]
events_collection = mdb["events"]

docs = events_collection.find({"batches_pushed": False})

print("Not pushed files in archive:")
for doc in docs:
    print("  -", doc["archived_file"])

count = 0

for doc in docs:
    fname = doc["archived_file"]
    fpath = os.path.join(ARCHIVE_DIR, fname)

    if not os.path.exists(fpath):
        print(f"Missing file: {fpath}")
        continue

    try:
        df = pd.read_parquet(fpath)
        batches = []
        grouped = df.groupby(df.index)

        for event_id, group in grouped:
            batch = {
                "event_id": event_id,
                "data": group.reset_index(drop=True).to_dict(orient="records")
            }
            r.lpush("event_queue", pickle.dumps(batch))
            batches.append(event_id)

        events_collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {
                "batches_pushed": True,
                "pushed_at": time.time(),
                "event_ids": batches
            }}
        )
        print(f"Re-pushed and updated: {fname}")
        count += 1

    except Exception as e:
        print(f"Failed to reprocess {fname}: {e}")

print(f"Total reprocessed files: {count}")
