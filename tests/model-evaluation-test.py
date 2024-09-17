import argparse
import itertools
import os
import time
import typing

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
    return parser.parse_args()


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
    distance_metric: str = DISTANCE_METRICS[0],
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
    embeddings_1 = process_image(img1_path, model=model, detector_backend=detector_backend, distance_metric=distance_metric)
    embeddings_2 = process_image(img2_path, model=model, detector_backend=detector_backend, distance_metric=distance_metric)
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
        print(f"Matches: {result['verified']}")
        print(f"Faces count: {faces_count}")
        print(f"Model: {result['model']}")
        print(f"Detector: {result['detector_backend']}")
        print(f"Distance metric: {result['similarity_metric']}")
        print(f"Distance: {result['distance']}")
        print(f"Threshold: {result['threshold']}")
        print(f"Time: {duration}")

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

    with tqdm.tqdm(
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


if __name__ == "__main__":
    args = parse_args()

    people_directories = [f"{args.directory}/{person}" for person in os.listdir(args.directory) if os.path.isdir(f"{args.directory}/{person}")]
    images = [[f"{person_directory}/{img}" for img in os.listdir(person_directory)] for person_directory in people_directories]

    print("=== Testing model configurations ===")
    for model, detector_backend, distance_metric in itertools.product(MODELS, DETECTOR_BACKENDS, DISTANCE_METRICS):
        eval([images[0][:2], images[1][:2]], model, detector_backend, distance_metric, silent=True)

    final_results: list[tuple[tuple[str, str, str], dict[str, float]]] = []
    for model, detector_backend, distance_metric in itertools.product(MODELS, DETECTOR_BACKENDS, DISTANCE_METRICS):
        results = eval(images, model, detector_backend, distance_metric)
        final_results.append(((model, detector_backend, distance_metric), results))
        print(f"Precision: {results['precision']}, Recall: {results['recall']}, F1 Score: {results['f1']}, Avg Time: {results['avg_time']}s")

    print("=== Final Results ===")
    for (model, detector_backend, distance_metric), results in final_results:
        print(f"Model: {model}, Detector: {detector_backend}, Distance Metric: {distance_metric}")
        print(f"Precision: {results['precision']}")
        print(f"Recall: {results['recall']}")
        print(f"F1 Score: {results['f1']}")
        print(f"Avg Time: {results['avg_time']}s")
        print("")

    print("=== Top-3 Best Models ===")
    best_models = sorted(final_results, key=lambda x: x[1]["f1"], reverse=True)[:3]
    for (model, detector_backend, distance_metric), results in best_models:
        print(f"Model: {model}, Detector: {detector_backend}, Distance Metric: {distance_metric}")
        print(f"Precision: {results['precision']}")
        print(f"Recall: {results['recall']}")
        print(f"F1 Score: {results['f1']}")
        print(f"Avg Time: {results['avg_time']}s")
        print("")

    print("=== Top-3 Fastest Model ===")
    fastest_models = sorted(final_results, key=lambda x: x[1]["avg_time"])[:3]
    for (model, detector_backend, distance_metric), results in fastest_models:
        print(f"Model: {model}, Detector: {detector_backend}, Distance Metric: {distance_metric}")
        print(f"Precision: {results['precision']}")
        print(f"Recall: {results['recall']}")
        print(f"F1 Score: {results['f1']}")
        print(f"Avg Time: {results['avg_time']}s")
        print("")
