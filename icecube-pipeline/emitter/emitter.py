#!/usr/bin/env python3
"""
Emitter service implementation.
- Batches and serializes events from the raw IceCube dataset.
- Pushes the serialized batches to a Redis queue.
- #TODO: Sends periodic heartbeats. 

Functionality:
-------------
1. Continuously watches the `./input` directory for new `.parquet` files.
2. When a new file is detected:
   - It is archived to the `./archive` folder (timestamped copy).
   - Basic metadata is extracted and logged in MongoDB (`icecube_db.events`).
   - The data is split into event batches (grouped by event_id).
   - Each batch is serialized and pushed into a Redis queue (`event_queue`) for downstream processing.
   - Upon successful push, the MongoDB entry is updated to indicate that the batches were processed.

Robustness:
-----------
- Files are archived and logged in MongoDB even if Redis is unavailable.
- Processing status is tracked using the `batches_pushed` flag in the database.
- This enables fault-tolerant recovery via a separate retry script.

Usage:
------
Run this script continuously to process incoming event data files:

    python emitter.py
"""

import os
import shutil
import time
import pandas as pd
import redis
from pymongo import MongoClient
import pickle

INPUT_DIR = "../../input"
ARCHIVE_DIR = "../../archive"

# Setup Redis and MongoDB connections
r = redis.Redis(host='localhost', port=6379, db=0)
mongo = MongoClient("mongodb://localhost:27017/")
mdb = mongo["icecube_db"]
events_collection = mdb["events"]

def archive_file(file_path):
    ts = int(time.time())
    basename = os.path.basename(file_path)
    archived_name = f"{ts}_{basename}"
    dest_path = os.path.join(ARCHIVE_DIR, archived_name)
    shutil.copy(file_path, dest_path)
    return dest_path, ts

def extract_batches(df):
    batches = []
    grouped = df.groupby(df.index)
    for event_id, group in grouped:
        batch = {
            "event_id": event_id,
            "data": group.reset_index(drop=True).to_dict(orient="records")
        }
        batches.append(batch)
    return batches

def process_new_file(filepath):
    archived_path, ts = archive_file(filepath)
    print(f"Archived file to: {archived_path}")

    df = pd.read_parquet(filepath)

    metadata = {
        "original_file": os.path.basename(filepath),
        "archived_file": os.path.basename(archived_path),
        "num_rows": len(df),
        "timestamp": ts,
        "batches_pushed": False
    }
    result = events_collection.insert_one(metadata)
    doc_id = result.inserted_id

    print(f"Metadata saved with _id: {doc_id}")

    try:
        batches = extract_batches(df)
        event_ids = []

        for batch in batches:
            r.lpush("event_queue", pickle.dumps(batch))
            event_ids.append(batch["event_id"])
            print(f"Pushed batch for event_id: {batch['event_id']}")

        # Update MongoDB to mark as pushed
        events_collection.update_one(
            {"_id": doc_id},
            {"$set": {
                "batches_pushed": True,
                "pushed_at": time.time(),
                "event_ids": event_ids
            }}
        )
    except Exception as e:
        print(f"Error while pushing to Redis: {e}")

    os.remove(filepath)

def main_loop():
    print("Watching folder:", INPUT_DIR)
    while True:
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".parquet")]
        for fname in files:
            full_path = os.path.join(INPUT_DIR, fname)
            process_new_file(full_path)
        time.sleep(5)

if __name__ == "__main__":
    main_loop()