from __future__ import annotations

from dotenv import load_dotenv
from pymongo import MongoClient
import os


def main() -> None:
    load_dotenv(".env")

    uri = os.getenv("MONGO_CONNECTION_STRING")
    print("URI_SET", bool(uri))
    if not uri:
        return

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    col = client["face_recognition_db"]["criminals"]

    print("TOTAL", col.count_documents({}))
    print("MISSING_STATUS", col.count_documents({"status": {"$exists": False}}))
    print("MISSING_IMAGEURL", col.count_documents({"imageURL": {"$exists": False}}))
    print("EMPTY_IMAGEURL", col.count_documents({"imageURL": ""}))

    print("STATUS_COUNTS")
    for row in col.aggregate(
        [
            {"$group": {"_id": "$status", "n": {"$sum": 1}}},
            {"$sort": {"n": -1}},
        ]
    ):
        print(repr(row.get("_id")), row.get("n"))

    q = {
        "status": {"$regex": r"^\s*NOT\s+ARRESTED\s*$", "$options": "i"},
        "imageURL": {"$exists": True, "$ne": ""},
    }
    print("COUNT_NOT_ARRESTED_WITH_IMAGE", col.count_documents(q))
    print(
        "SAMPLE_NOT_ARRESTED",
        col.find_one(q, {"name": 1, "status": 1, "imageURL": 1, "createdAt": 1}),
    )


if __name__ == "__main__":
    main()