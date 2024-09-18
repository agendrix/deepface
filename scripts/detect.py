import argparse
import json

from deepface.commons.agendrix.constants import DETECTOR_BACKENDS, MODELS
from deepface.commons.agendrix.image_processing import get_faces_embeddings
from deepface.commons.image_utils import load_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str, help="Path to the image")
    parser.add_argument("--model", type=str, default="Facenet512", choices=MODELS, help="Face recognition model")
    parser.add_argument("--detector_backend", type=str, default="mediapipe", choices=DETECTOR_BACKENDS, help="Face detector and alignment backend")
    return parser.parse_args()


def main():
    args = parse_args()
    img, _ = load_image(args.img_path)
    embeddings = get_faces_embeddings(img, args.model, args.detector_backend)
    faces_count = len(embeddings)

    output = {
        "faces_count": faces_count,
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
