from typing import Union

import numpy

from deepface import DeepFace


def get_faces_embeddings(
    img: Union[str, numpy.ndarray],
    model: str,
    detector_backend: str,
) -> list[list[float]]:
    detect_result = DeepFace.represent(
        img,
        model_name=model,
        detector_backend=detector_backend,
        enforce_detection=False,
        max_faces=3,
    )
    return [result["embedding"] for result in detect_result]
