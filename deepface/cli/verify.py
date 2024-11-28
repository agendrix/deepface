import argparse
import json
import logging
import os

from deepface import DeepFace
from deepface.commons.agendrix.argparser import add_detector_backend_arg, add_distance_metric_arg, add_log_level_arg, add_model_arg, add_redis_key_arg, add_threshold_arg
from deepface.commons.agendrix.image_processing import get_faces_embeddings
from deepface.commons.agendrix.redis import initialize_redis
from deepface.commons.image_utils import load_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("img1_path", type=str, help="Path to the first image")
    parser.add_argument("img2_path", type=str, help="Path to the second image")

    parser = add_model_arg(parser)
    parser = add_detector_backend_arg(parser)
    parser = add_distance_metric_arg(parser)
    parser = add_threshold_arg(parser)

    parser = add_redis_key_arg(parser)

    parser = add_log_level_arg(parser)

    return parser.parse_args()


def main():
    if os.getenv("RAILS_ENV", "development") == "development":
        from dotenv import load_dotenv

        load_dotenv()

    args = parse_args()
    logging.basicConfig(level=args.log)

    img1, _ = load_image(args.img1_path)
    img2, _ = load_image(args.img2_path)

    logging.debug("Calculating embeddings for the first image...")
    embeddings_1 = get_faces_embeddings(img1, args.model, args.detector_backend)
    logging.debug("Calculating embeddings for the second image...")
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
        threshold=args.threshold,
        anti_spoofing=False,
    )

    json_output = json.dumps(
        {
            "matches": result["verified"],
            "distance": result["distance"],
            "faces_count": faces_count,
        }
    )

    logging.info(json_output)

    if args.redis_key is not None:
        r = initialize_redis()
        r.setex(args.redis_key, 12 * 60 * 60, json_output)  # 12h expiration


if __name__ == "__main__":
    main()
