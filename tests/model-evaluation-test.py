import argparse
import csv
import itertools
import logging
import os
import random
import time
import typing
from typing import Any, Optional

import matplotlib.pyplot as plt
import PIL
import PIL.Image
import PIL.ImageFile
import tqdm
from PIL import Image

from deepface import DeepFace

# MODELS = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace', 'SFace', 'GhostFaceNet']
MODELS = ["VGG-Face", "Facenet", "Facenet512", "OpenFace"]
# DETECTOR_BACKENDS = ['opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface', 'skip']
DETECTOR_BACKENDS = [
    "opencv",
    "retinaface",
    "mtcnn",
    "ssd",
    "dlib",
    "mediapipe",
    "yolov8",
    "centerface",
]
# DISTANCE_METRICS = ['cosine', 'euclidean', 'euclidean_l2']
DISTANCE_METRICS = ["cosine"]

# IMAGES = [
#     ["/Users/philippe/Pictures/Photo on 2024-09-16 at 09.31.jpg", "/Users/philippe/Pictures/Photo on 2024-09-16 at 09.31 #2.jpg"],
#     ["/Users/philippe/CleanShot/CleanShot 2024-09-12 at 08.17.04.png"]
# ]


def parse_args():
    parser = argparse.ArgumentParser(description="Model evaluation test")
    parser.add_argument("directory", type=str, help="Directory containing images")
    parser.add_argument("-o", "--output", type=str, default="output.csv", help="Output file to write results to")
    parser.add_argument("--n", type=int, help="Number of people directories to include")
    parser.add_argument("--log", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.log)

    images = get_data(args.directory, args.n)

    logging.info("=== Testing model configurations ===")
    for model, detector_backend, distance_metric in itertools.product(MODELS, DETECTOR_BACKENDS, DISTANCE_METRICS):
        eval([images[0][:2], images[1][:2]], model, detector_backend, distance_metric, silent=True)
    logging.info("=== Finished testing model configurations ===")

    logging.info("=== Evaluating models ===")
    final_results: list[tuple[tuple[str, str, str], dict[str, float]]] = []
    for model, detector_backend, distance_metric in itertools.product(MODELS, DETECTOR_BACKENDS, DISTANCE_METRICS):
        results = eval(images, model, detector_backend, distance_metric)
        final_results.append(((model, detector_backend, distance_metric), results))
        logging.info(f"Precision: {results['precision']}, Recall: {results['recall']}, F1 Score: {results['f1']}, Avg Time: {results['avg_time']}s")

    # Write results to CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Detector", "Distance Metric", "Precision", "Recall", "F1 Score", "Avg Time"])
        for (model, detector_backend, distance_metric), results in final_results:
            writer.writerow([model, detector_backend, distance_metric, results["precision"], results["recall"], results["f1"], results["avg_time"]])

    logging.info("=== Final Results ===")
    for (model, detector_backend, distance_metric), results in final_results:
        logging.info(f"Model: {model}, Detector: {detector_backend}, Distance Metric: {distance_metric}")
        logging.info(f"Precision: {results['precision']}")
        logging.info(f"Recall: {results['recall']}")
        logging.info(f"F1 Score: {results['f1']}")
        logging.info(f"Avg Time: {results['avg_time']}s")
        logging.info("")

    logging.info("=== Top-3 Best Models ===")
    best_models = sorted(final_results, key=lambda x: x[1]["f1"], reverse=True)[:3]
    for (model, detector_backend, distance_metric), results in best_models:
        logging.info(f"Model: {model}, Detector: {detector_backend}, Distance Metric: {distance_metric}")
        logging.info(f"Precision: {results['precision']}")
        logging.info(f"Recall: {results['recall']}")
        logging.info(f"F1 Score: {results['f1']}")
        logging.info(f"Avg Time: {results['avg_time']}s")
        logging.info("")

    logging.info("=== Top-3 Fastest Model ===")
    fastest_models = sorted(final_results, key=lambda x: x[1]["avg_time"])[:3]
    for (model, detector_backend, distance_metric), results in fastest_models:
        logging.info(f"Model: {model}, Detector: {detector_backend}, Distance Metric: {distance_metric}")
        logging.info(f"Precision: {results['precision']}")
        logging.info(f"Recall: {results['recall']}")
        logging.info(f"F1 Score: {results['f1']}")
        logging.info(f"Avg Time: {results['avg_time']}s")
        logging.info("")


def get_data(directory: str, n_dirs: Optional[int]) -> list[list[str]]:
    subdirs = [f"{directory}/{person}" for person in os.listdir(directory) if os.path.isdir(f"{directory}/{person}")]
    random.shuffle(subdirs)
    if n_dirs is not None:
        subdirs = subdirs[:n_dirs]
    return [[f"{subdir}/{img}" for img in os.listdir(subdir)] for subdir in subdirs]


def show_images(images: list[PIL.ImageFile.ImageFile]):
    _, axs = plt.subplots(1, len(images))
    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis("off")
    plt.show()


def process_image(
    img_path: str,
    model: str = MODELS[0],
    detector_backend: str = DETECTOR_BACKENDS[2],
) -> list[list[float]]:
    detect_result = DeepFace.represent(
        img_path,
        model_name=model,
        detector_backend=detector_backend,
        enforce_detection=False,
        max_faces=3,
    )
    return [result["embedding"] for result in detect_result]


def verify(
    img1_path: str,
    img2_path: str,
    model: str = MODELS[0],
    detector_backend: str = DETECTOR_BACKENDS[2],
    distance_metric: str = DISTANCE_METRICS[0],
    silent: bool = False,
) -> dict[str, typing.Union[bool, float]]:
    if silent == False:
        show_images([Image.open(img1_path), Image.open(img2_path)])

    start = time.perf_counter()
    embeddings_1 = process_image(img1_path, model=model, detector_backend=detector_backend)
    embeddings_2 = process_image(img2_path, model=model, detector_backend=detector_backend)
    faces_count = len(embeddings_2)

    result = DeepFace.verify(
        img1_path=embeddings_1,
        img2_path=embeddings_2,
        model_name=model,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=False,
        silent=silent,
        threshold=None,
        anti_spoofing=False,
    )
    end = time.perf_counter()
    duration = end - start

    if silent == False:
        logging.debug(f"Matches: {result['verified']}")
        logging.debug(f"Faces count: {faces_count}")
        logging.debug(f"Model: {result['model']}")
        logging.debug(f"Detector: {result['detector_backend']}")
        logging.debug(f"Distance metric: {result['similarity_metric']}")
        logging.debug(f"Distance: {result['distance']}")
        logging.debug(f"Threshold: {result['threshold']}")
        logging.debug(f"Time: {duration}")

    return {
        "matches": result["verified"],
        "distance": result["distance"],
        "time": duration,
        "faces_count": faces_count,
    }


def eval(
    images: list[list[str]],
    model: str,
    detector_backend: str,
    distance_metric: str,
    silent: bool = True,
) -> dict[str, float]:
    prediction_results: dict[str, float] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "avg_time": 0.0}
    total_time = 0.0

    total_verifications = sum([len(person_images) for person_images in images]) * (sum([len(person_images) for person_images in images]) - 1) // 2

    with TqdmLoggingContextManager(
        total=total_verifications,
        desc=f"Model: {model}, Detector: {detector_backend}, Distance Metric: {distance_metric}",
    ) as pbar:
        for i, person_images in enumerate(images):
            same_person_combination = itertools.combinations(person_images, 2)
            # Compare all images of the same person
            for img_pair in same_person_combination:
                img1_path, img2_path = img_pair
                result = verify(
                    img1_path,
                    img2_path,
                    model=model,
                    detector_backend=detector_backend,
                    distance_metric=distance_metric,
                    silent=silent,
                )
                pbar.update(1)
                total_time += typing.cast(float, result["time"])
                if result["matches"]:
                    prediction_results["TP"] += 1
                else:
                    prediction_results["FN"] += 1

            for other_images in images[i + 1 :]:
                # Compare all images of different people
                different_person_combination = itertools.product(person_images, other_images)
                for img_pair in different_person_combination:
                    img1_path, img2_path = img_pair
                    result = verify(
                        img1_path,
                        img2_path,
                        model=model,
                        detector_backend=detector_backend,
                        distance_metric=distance_metric,
                        silent=silent,
                    )
                    pbar.update(1)
                    total_time += typing.cast(float, result["time"])
                    if result["matches"]:
                        prediction_results["FP"] += 1
                    else:
                        prediction_results["TN"] += 1

    results = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    epsilon = 1e-6
    results["precision"] = round(
        prediction_results["TP"] / (prediction_results["TP"] + prediction_results["FP"] + epsilon),
        2,
    )
    results["recall"] = round(
        prediction_results["TP"] / (prediction_results["TP"] + prediction_results["FN"] + epsilon),
        2,
    )
    results["f1"] = round(
        2 * (results["precision"] * results["recall"]) / (results["precision"] + results["recall"] + epsilon),
        2,
    )
    results["avg_time"] = round(
        total_time / (prediction_results["TP"] + prediction_results["FP"] + prediction_results["TN"] + prediction_results["FN"]),
        2,
    )
    return results


class TqdmLoggingContextManager:
    def __init__(self, total: int, desc: str, **kwargs: Any):
        self.total = total
        self.desc = desc
        self.kwargs = kwargs

    def __enter__(self):
        if logging.getLogger().getEffectiveLevel() <= logging.INFO:
            self.progressbar = tqdm.tqdm(total=self.total, desc=self.desc, **self.kwargs)
        else:
            self.progressbar = TqdmDummy()

        return self.progressbar

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any):
        self.progressbar.close()


class TqdmDummy:
    def __init__(self):
        pass

    def update(self, n: int):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    main()
