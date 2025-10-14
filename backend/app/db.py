from flask_pymongo import PyMongo

mongo = PyMongo()

def init_db(app):
    mongo.init_app(app)
    
    # Test the connection immediately
    try:
        # This will raise an exception if connection fails
        mongo.db.list_collection_names()
        print("✅ MongoDB connected successfully")
    except Exception as e:
        print("❌ MongoDB connection failed:", e)
        # Optionally raise an error to stop app from running
        raise e
