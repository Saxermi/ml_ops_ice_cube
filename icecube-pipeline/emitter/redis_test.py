import redis

try:
    r = redis.Redis(host='localhost', port=6379)
    if r.ping():
        print("Redis is running and reachable.")
except Exception as e:
    print(f"Redis connection failed: {e}")


import pickle
r = redis.Redis(host="localhost", port=6379)
print("Queue length:", r.llen("event_queue"))

if r.llen("event_queue") > 0:
    batch = pickle.loads(r.rpop("event_queue"))
    print("event_id:", batch["event_id"])
    print("first rows:", batch["data"][:2])


"""r.delete("event_queue")
print("Queue length after deletion:", r.llen("event_queue"))
print("Queue cleared.")"""

