import argparse
import json

from deepface import DeepFace
from deepface.commons.agendrix.constants import DETECTOR_BACKENDS, DISTANCE_METRICS, MODELS
from deepface.commons.agendrix.image_processing import get_faces_embeddings
from deepface.commons.image_utils import load_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("img1_path", type=str, help="Path to the first image")
    parser.add_argument("img2_path", type=str, help="Path to the second image")
    parser.add_argument("--model", type=str, default="Facenet512", choices=MODELS, help="Face recognition model")
    parser.add_argument("--detector_backend", type=str, default="mediapipe", choices=DETECTOR_BACKENDS, help="Face detector and alignment backend")
    parser.add_argument("--distance_metric", type=str, default="cosine", choices=DISTANCE_METRICS, help="Type of distance metric to use")
    return parser.parse_args()


def main():
    args = parse_args()
    img1, _ = load_image(args.img1_path)
    img2, _ = load_image(args.img2_path)

    embeddings_1 = get_faces_embeddings(img1, args.model, args.detector_backend)
    embeddings_2 = get_faces_embeddings(img2, args.model, args.detector_backend)
    faces_count = len(embeddings_2)

    result = DeepFace.verify(
        img1_path=embeddings_1,
        img2_path=embeddings_2,
        model_name=args.model,
        detector_backend=args.detector_backend,
        distance_metric=args.distance_metric,
        enforce_detection=False,
        silent=True,
        threshold=None,
        anti_spoofing=False,
    )

    output = {
        "matches": result["verified"],
        "distance": result["distance"],
        "faces_count": faces_count,
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
