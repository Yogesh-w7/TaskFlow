from flask_pymongo import PyMongo
from pymongo.write_concern import WriteConcern

mongo = PyMongo()

def init_db(app):
    # Strip URI to remove any trailing newlines
    mongo_uri = app.config.get("MONGO_URI")
    if not mongo_uri:
        raise ValueError("❌ MONGO_URI not found in app config.")
    app.config["MONGO_URI"] = mongo_uri.strip()

    # Initialize PyMongo
    mongo.init_app(app)

    try:
        # Force proper write concern to avoid 'majority\n' errors
        mongo.cx.get_database().write_concern = WriteConcern(w="majority", wtimeout=2500)

        # Test connection
        mongo.db.list_collection_names()
        print("✅ MongoDB connected successfully")
    except Exception as e:
        print("❌ MongoDB connection failed:", e)
        raise e
