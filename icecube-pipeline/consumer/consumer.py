#!/usr/bin/env python3
import os
import time
import json
import logging

import redis
import requests

# ——— Configuration via env vars ——————————————————————
REDIS_URL         = os.getenv("REDIS_URL", "redis://redis:6379/0")
MODEL_SERVER_URL  = os.getenv("MODEL_SERVER_URL", "http://model-server:5000/predict")
EVENT_QUEUE       = os.getenv("EVENT_QUEUE", "events")
POLL_INTERVAL     = float(os.getenv("POLL_INTERVAL", "0.5"))  # seconds

# ——— Setup logging & Redis client ————————————————
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)
r = redis.from_url(REDIS_URL)


def process_event(raw):
    """
    raw is the JSON-encoded dict you originally pushed:
      {
        "event_id": "...",
        "data": [
          {"time": ..., "charge": ..., "auxiliary": ..., "sensor_id": ...},
          ...
        ]
      }
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logging.error("Malformed event, skipping: %r", raw)
        return

    event_id = payload.get("event_id", "<no-id>")
    logging.info("Sending event %s to model server", event_id)
    try:
        resp = requests.post(MODEL_SERVER_URL, json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        logging.info("Result for %s → azimuth=%.3f, zenith=%.3f",
                     event_id, result["azimuth"], result["zenith"])
    except Exception as e:
        logging.error("Error processing %s: %s", event_id, e)


def main():
    logging.info("Consumer started, connecting to %s", REDIS_URL)
    while True:
        # blocking pop: wait up to 1 second for an event
        item = r.blpop(EVENT_QUEUE, timeout=1)
        if item:
            # item is (queue_name, raw_data)
            _, raw = item
            process_event(raw)
        else:
            # optional heartbeat/log
            logging.debug("No events, sleeping %.1fs", POLL_INTERVAL)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()