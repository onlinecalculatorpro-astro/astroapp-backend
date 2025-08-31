# tests/conftest.py
import pytest
from flask import Flask
from app.api.predictions import predictions_bp

@pytest.fixture
def app():
    app = Flask(__name__)
    app.config.update(TESTING=True)
    app.register_blueprint(predictions_bp, url_prefix="/api")
    return app

@pytest.fixture
def client(app):
    return app.test_client()
