# built-in dependencies
import json
import logging
import traceback
from typing import Optional

# project dependencies
from deepface import DeepFace
from deepface.commons.agendrix.image_processing import get_faces_embeddings
from deepface.commons.agendrix.redis import initialize_redis
from deepface.commons.image_utils import load_image

# pylint: disable=broad-except


def represent(
    img_path: str,
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
    max_faces: Optional[int] = None,
):
    try:
        result = {}
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
        result["results"] = embedding_objs
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while representing: {str(err)} - {tb_str}"}, 400


def analyze(
    img_path: str,
    actions: list,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        result = {}
        demographies = DeepFace.analyze(
            img_path=img_path,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            silent=True,
            anti_spoofing=anti_spoofing,
        )
        result["results"] = demographies
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400


def detect(
    img_path: str,
    model: str,
    detector_backend: str,
    redis_key: Optional[str],
):
    try:
        img, _ = load_image(img_path)
        embeddings = get_faces_embeddings(img, model, detector_backend)
        faces_count = len(embeddings)

        raw_output = {"faces_count": faces_count}
        logging.info(json.dumps(raw_output | {"redis_key": redis_key}))

        if redis_key is not None:
            r = initialize_redis()
            r.setex(redis_key, 12 * 60 * 60, json.dumps({"faces_count": faces_count}))  # 12h expiration

        return raw_output

    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400


def verify(
    img1_path: str,
    img2_path: str,
    model: str,
    detector_backend: str,
    distance_metric: str,
    threshold: float,
    redis_key: Optional[str],
):
    try:
        img1, _ = load_image(img1_path)
        img2, _ = load_image(img2_path)

        embeddings_1 = get_faces_embeddings(img1, model, detector_backend)
        embeddings_2 = get_faces_embeddings(img2, model, detector_backend)
        faces_count = len(embeddings_2)

        result = DeepFace.verify(
            img1_path=embeddings_1,
            img2_path=embeddings_2,
            model_name=model,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=False,
            silent=True,
            threshold=threshold,
            anti_spoofing=False,
        )

        raw_output = {
            "matches": result["verified"],
            "distance": result["distance"],
            "faces_count": faces_count,
        }
        logging.info(json.dumps(raw_output | {"redis_key": redis_key}))

        if redis_key is not None:
            r = initialize_redis()
            r.setex(redis_key, 12 * 60 * 60, json.dumps(raw_output))  # 12h expiration

        return raw_output

    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400
