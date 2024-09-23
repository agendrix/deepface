from argparse import ArgumentParser

from deepface.commons.agendrix.constants import DETECTOR_BACKENDS, DISTANCE_METRICS, MODELS


def add_model_arg(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--model", type=str, default="Facenet512", choices=MODELS, help="Face recognition model")
    return parser


def add_detector_backend_arg(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--detector_backend", type=str, default="ssd", choices=DETECTOR_BACKENDS, help="Face detector and alignment backend")
    return parser


def add_distance_metric_arg(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--distance_metric", type=str, default="cosine", choices=DISTANCE_METRICS, help="Type of distance metric to use")
    return parser


def add_threshold_arg(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--threshold", type=float, help="Model threshold to use when matching faces")
    return parser


def add_redis_key_arg(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--redis_key", type=str, help="Key to use to store results in Redis")
    return parser


def add_log_level_arg(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--log", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return parser
