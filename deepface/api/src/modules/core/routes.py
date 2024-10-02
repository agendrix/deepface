from flask import Blueprint, request

from deepface import DeepFace
from deepface.api.src.modules.core import service
from deepface.commons.agendrix.constants import DEFAULT_DETECTOR_BACKEND, DEFAULT_DISTANCE_METRIC, DEFAULT_MODEL
from deepface.commons.logger import Logger

logger = Logger()

blueprint = Blueprint("routes", __name__)


@blueprint.route("/")
def home():
    return f"<h1>Welcome to DeepFace API v{DeepFace.__version__}!</h1>"


@blueprint.route("/detect", methods=["POST"])
def detect():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    output = service.detect(
        img_path=img_path,
        model=input_args.get("model", DEFAULT_MODEL),
        detector_backend=input_args.get("detector_backend", DEFAULT_DETECTOR_BACKEND),
        redis_key=input_args.get("redis_key"),
    )

    return output


@blueprint.route("/verify", methods=["POST"])
def verify():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img1_path = input_args.get("img1_path")
    img2_path = input_args.get("img2_path")

    if img1_path is None:
        return {"message": "you must pass img1_path input"}

    if img2_path is None:
        return {"message": "you must pass img2_path input"}

    output = service.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model=input_args.get("model", DEFAULT_MODEL),
        detector_backend=input_args.get("detector_backend", DEFAULT_DETECTOR_BACKEND),
        distance_metric=input_args.get("distance_metric", DEFAULT_DISTANCE_METRIC),
        threshold=input_args.get("threshold", 0.3),
        redis_key=input_args.get("redis_key"),
    )

    return output
