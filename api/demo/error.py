from flask import jsonify
from flask.typing import ResponseReturnValue
from werkzeug.exceptions import HTTPException

def handle(e: HTTPException) -> ResponseReturnValue:
    code = e.code if e.code is not None else 500
    r = jsonify({ "error": { "code": code, "message": e.description }})
    r.status = code
    return r

