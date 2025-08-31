# tests/conftest.py
import pytest
from flask import Flask
from app.api.predictions import predictions_bp

@pytest.fixture
def app():
    """Minimal Flask app with only /api/predictions registered for tests."""
    app = Flask(__name__)
    app.config.update(TESTING=True)
    app.register_blueprint(predictions_bp, url_prefix="/api")
    return app

@pytest.fixture
def client(app):
    """Test client fixture for hitting the API."""
    return app.test_client()
