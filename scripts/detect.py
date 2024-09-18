import argparse
import json
import logging

from dotenv import load_dotenv

from deepface.commons.agendrix.argparser import add_detector_backend_arg, add_log_level_arg, add_model_arg, add_redis_key_arg
from deepface.commons.agendrix.image_processing import get_faces_embeddings
from deepface.commons.agendrix.redis import initialize_redis
from deepface.commons.image_utils import load_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str, help="Path to the image")

    parser = add_model_arg(parser)
    parser = add_detector_backend_arg(parser)

    parser = add_redis_key_arg(parser)

    parser = add_log_level_arg(parser)

    return parser.parse_args()


def main():
    # Load dotenv
    load_dotenv()

    args = parse_args()
    logging.basicConfig(level=args.log)

    img, _ = load_image(args.img_path)
    embeddings = get_faces_embeddings(img, args.model, args.detector_backend)
    faces_count = len(embeddings)

    output = {
        "faces_count": faces_count,
    }

    logging.info(json.dumps(output))

    if args.redis_key is not None:
        r = initialize_redis()
        r.set(args.redis_key, json.dumps(output))


if __name__ == "__main__":
    main()
