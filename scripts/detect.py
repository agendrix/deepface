import argparse
import json

from deepface.commons.agendrix.argparser import add_detector_backend_arg, add_model_arg
from deepface.commons.agendrix.image_processing import get_faces_embeddings
from deepface.commons.image_utils import load_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str, help="Path to the image")

    parser = add_model_arg(parser)
    parser = add_detector_backend_arg(parser)

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
