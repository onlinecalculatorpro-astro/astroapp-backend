from flask import Flask

def create_app():
    app = Flask(__name__)
    # load config here…

    # Western routes
    from app.api.routes import api as western_api
    app.register_blueprint(western_api, url_prefix="/api")

    # Optional Vedic routes
    if app.config.get("ENABLE_VEDIC_API", True):  # or env flag
        from app.api.vedic_routes import vedic_api
        app.register_blueprint(vedic_api, url_prefix="/api/vedic")

    return app

