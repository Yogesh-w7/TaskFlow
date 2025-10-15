from pymongo import WriteConcern
from flask_pymongo import PyMongo

mongo = PyMongo()

def init_db(app):
    mongo.init_app(app)
    try:
        db = mongo.cx.get_database("taskmanager").with_options(
            write_concern=WriteConcern(w="majority", wtimeout=2500)
        )
        # test connection
        db.list_collection_names()
        print("✅ MongoDB connected successfully")
    except Exception as e:
        print("❌ MongoDB connection failed:", e)
        raise e
