from flask import Flask, jsonify
from app.api.routes import api
from flask_cors import CORS
from app.utils.config import load_config
import os

def create_app():
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False
    cfg_path = os.environ.get("ASTRO_CONFIG","config/defaults.yaml")
    app.cfg = load_config(cfg_path)
    app.register_blueprint(api)
    return app

app = create_app()
CORS(app, resources={r"*": {"origins": "*"}})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
