import argparse

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("img1_path", type=str, help="Path to the first image")
    parser.add_argument("img2_path", type=str, help="Path to the second image")
    parser.add_argument("--model", type=str, default="VGG-Face", choices=MODELS, help="Face recognition model")
    parser.add_argument("--detector_backend", type=str, default="opencv", choices=DETECTOR_BACKENDS, help="Face detector and alignment backend")
    parser.add_argument("--distance_metric", type=str, default="cosine", choices=DISTANCE_METRICS, help="Type of distance metric to use")
    return parser.parse_args()


def main():
    args = parse_args()
    embeddings_1 = process_image(args.img1_path, args.model, args.detector_backend)
    embeddings_2 = process_image(args.img2_path, args.model, args.detector_backend)
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

    print(output)


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


if __name__ == "__main__":
    main()
