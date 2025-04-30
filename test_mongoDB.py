from pymongo import MongoClient
cli = MongoClient("mongodb://localhost:27017")
col = cli["anxiety"]["guidelines"]

doc = col.find_one({}, {"_id": 0, "title": 1, "vector": 1})
print(doc["title"])
print("Vector length:", len(doc["vector"]))   # should be 384
print("First 5 numbers:", doc["vector"][:5])