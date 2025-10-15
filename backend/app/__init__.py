import os
from flask import Flask, send_from_directory
from flask_cors import CORS
from .db import init_db
from .routes.tasks import task_bp
from .routes.comments import comment_bp

def create_app():
    # Flask app
    app = Flask(__name__, static_folder="../frontend/build", static_url_path="/")

    # -----------------------
    # CORS setup
    # -----------------------
    frontend_origin = os.getenv("FRONTEND_URL")
    if not frontend_origin:
        raise ValueError("❌ FRONTEND_URL is missing in environment variables.")
    CORS(app, resources={r"/api/*": {"origins": frontend_origin}})

    # -----------------------
    # MongoDB setup
    # -----------------------
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("❌ MONGO_URI is missing in environment variables.")
    app.config["MONGO_URI"] = mongo_uri.strip()  # Remove trailing newlines

    # Initialize MongoDB
    init_db(app)

    # -----------------------
    # Register Blueprints
    # -----------------------
    app.register_blueprint(task_bp, url_prefix="/api/tasks")
    app.register_blueprint(comment_bp, url_prefix="/api")

    # -----------------------
    # Health check route
    # -----------------------
    @app.route("/api/health")
    def health():
        return {"status": "ok"}, 200

    # -----------------------
    # Serve React frontend
    # -----------------------
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, "index.html")

    return app
