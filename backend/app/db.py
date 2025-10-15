from flask_pymongo import PyMongo
import os

mongo = PyMongo()

def init_db(app):
    # Get MongoDB URI from app config or environment
    mongo_uri = app.config.get("MONGO_URI") or os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ConnectionError("❌ MONGO_URI missing. Add it in Render Environment Variables.")
    
    app.config["MONGO_URI"] = mongo_uri
    mongo.init_app(app)

    # Test the connection
    try:
        if mongo.db is None:
            raise ConnectionError("MongoDB object not initialized")
        mongo.db.list_collection_names()
        print("✅ MongoDB connected successfully")
    except Exception as e:
        print("❌ MongoDB connection failed:", e)
        raise e
