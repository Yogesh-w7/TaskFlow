from flask_pymongo import PyMongo
import os

mongo = PyMongo()

def init_db(app):
    # Ensure Mongo URI exists
    mongo_uri = app.config.get("MONGO_URI") or os.getenv("MONGO_URI")

    if not mongo_uri:
        print("❌ MONGO_URI is missing. Please set it in environment variables.")
        raise ValueError("MONGO_URI not found")

    app.config["MONGO_URI"] = mongo_uri
    mongo.init_app(app)

    # Test the connection
    try:
        db = mongo.db
        if db is None:
            raise ConnectionError("MongoDB object not initialized")
        db.list_collection_names()
        print("✅ MongoDB connected successfully")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        raise e
