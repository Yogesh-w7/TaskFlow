from flask_pymongo import PyMongo

mongo = PyMongo()

def init_db(app):
    mongo.init_app(app)
    try:
        if mongo.db is None:
            raise ConnectionError("MongoDB object not initialized")
        mongo.db.list_collection_names()
        print("✅ MongoDB connected successfully")
    except Exception as e:
        print("❌ MongoDB connection failed:", e)
        raise e
