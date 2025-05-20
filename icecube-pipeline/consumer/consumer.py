#!/usr/bin/env python3
import os
import time
import json
import logging
import pickle
import redis
import requests
from prometheus_client import start_http_server, Counter, Gauge
from pymongo import MongoClient

# ——— Configuration via env vars ——————————————————————
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://model-server:5000/predict")
EVENT_QUEUE = os.getenv("EVENT_QUEUE", "events")
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "0.5"))  # seconds

# ——— Setup logging & Redis client ————————————————
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
r = redis.from_url(REDIS_URL)

# ——— Setup MongoDB client —————————————————————————
try:
    mongo = MongoClient("mongodb://mongo:27017/")
    db = mongo["icecube_db"]
    predictions = db["predictions"]
except:
    logging.info("mongodb failed in consumer")


# Start Prometheus metrics server on port 8001
start_http_server(8001)

# ——— Prometheus metrics ————————————————————————
heartbeat_counter = Counter("consumer_heartbeat_total", "Heartbeat from consumer")
batches_counter = Counter("consumer_batches_total", "Number of event batches processed")

azimuth_gauge = Gauge("last_prediction_azimuth", "Latest predicted azimuth")
zenith_gauge = Gauge("last_prediction_zenith", "Latest predicted zenith")


def process_event(raw):
    """
    raw is the serialized dict pushed into Redis:
      {
        "event_id": "...",
        "data": [
          {"time": ..., "charge": ..., "auxiliary": ..., "sensor_id": ...},
          ...
        ]
      }
    """
    try:
        payload = pickle.loads(raw)
    except (pickle.UnpicklingError, Exception) as e:
        logging.error("Failed to unpickle event: %s", e)
        return

    event_id = payload.get("event_id", "<no-id>")
    logging.info("Sending event %s to model server", event_id)
    try:
        resp = requests.post(MODEL_SERVER_URL, json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()

        azimuth = result.get("azimuth")
        zenith = result.get("zenith")
        model_version = result.get("model_version")

        if azimuth is not None:
            azimuth_gauge.set(azimuth)
        if zenith is not None:
            zenith_gauge.set(zenith)

        logging.info(
            "Result for %s → azimuth=%.3f, zenith=%.3f",
            event_id,
            azimuth,
            zenith,
        )

        # --- MoongoDB Logging ------------------------------------
        doc = {
            "event_id": event_id,
            "azimuth": azimuth,
            "zenith": zenith,
            "timestamp": int(time.time()),
            "model_url": MODEL_SERVER_URL,
            "model_version": model_version,
        }
        logging.info(doc)

        predictions.insert_one(doc)

    except Exception as e:
        logging.error("Error processing %s: %s", event_id, e)


def main():
    logging.info("Consumer started, connecting to %s", REDIS_URL)
    while True:
        heartbeat_counter.inc()
        item = r.blpop(EVENT_QUEUE, timeout=1)
        if item:
            _, raw = item
            process_event(raw)
            batches_counter.inc()
        else:
            logging.debug("No events, sleeping %.1fs", POLL_INTERVAL)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
