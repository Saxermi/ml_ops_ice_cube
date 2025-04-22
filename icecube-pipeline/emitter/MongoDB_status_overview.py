from pymongo import MongoClient
from pprint import pprint

client = MongoClient("mongodb://localhost:27017/")
db = client["icecube_db"]
collection = db.events

# Count stats
total = collection.count_documents({})
pushed = collection.count_documents({"batches_pushed": True})
not_pushed = collection.count_documents({"batches_pushed": False})

print("=== MongoDB Status Overview ===")
print(f"Total documents         : {total}")
print(f"Pushed to Redis         : {pushed}")
print(f"Not yet pushed          : {not_pushed}")
print()

# List of files not pushed
if not_pushed > 0:
    print("Files not yet pushed:")
    for doc in collection.find({"batches_pushed": False}, {"archived_file": 1, "_id": 0}):
        print(f"  - {doc['archived_file']}")
    print()
else:
    print("All documents marked as pushed.")
    print()
