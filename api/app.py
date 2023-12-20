from demo import api, glog, error
from flask import Flask, g
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

import os
import logging

def create_app() -> ProxyFix:
    # If LOG_FORMAT is "google:json" emit log message as JSON in a format Google Cloud can parse.
    fmt = os.getenv("LOG_FORMAT")
    handlers = [glog.Handler()] if fmt == "google:json" else []
    level = os.environ.get("LOG_LEVEL", default=logging.INFO)
    logging.basicConfig(level=level, handlers=handlers)

    app = Flask("demo")

    @app.teardown_appcontext
    def close_connection(exception):
        db = getattr(g, '_database', None)
        if db is not None:
            db.close()

    app.register_blueprint(api.create(), url_prefix="/")
    app.register_error_handler(HTTPException, error.handle)

    # Use the X-Forwarded-* headers to set the request IP, host and port. Technically there
    # are two reverse proxies in deployed environments, but we "hide" the reverse proxy deployed
    # as a sibling of the API by forwarding the X-Forwarded-* headers rather than chaining them.
    return ProxyFix(app, x_for=1, x_proto=1, x_host=1, x_port=1)

