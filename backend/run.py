from dotenv import load_dotenv
import os

load_dotenv()  # <-- load .env before imports

from app import create_app

print("MONGO_URI =", os.getenv("MONGO_URI"))  # debug: must print your URI

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
