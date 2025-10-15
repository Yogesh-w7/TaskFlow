import os
from flask import Flask, send_from_directory
from flask_cors import CORS
from .db import init_db
from .routes.tasks import task_bp
from .routes.comments import comment_bp

def create_app():
    app = Flask(__name__, static_folder="../frontend/build", static_url_path="/")

    # CORS setup
    frontend_origin = os.getenv("FRONTEND_URL", "*")
    CORS(app, resources={r"/api/*": {"origins": frontend_origin}})

    # Load Mongo URI from environment variable
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("❌ MONGO_URI is missing. Add it in .env or Render Environment Variables.")
    app.config["MONGO_URI"] = mongo_uri

    # Initialize MongoDB
    init_db(app)

    # Register blueprints
    app.register_blueprint(task_bp, url_prefix="/api/tasks")
    app.register_blueprint(comment_bp, url_prefix="/api")

    # Health route
    @app.route("/api/health")
    def health():
        return {"status": "ok"}, 200

    # Serve React build
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, "index.html")

    return app
