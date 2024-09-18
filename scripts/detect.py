import argparse
import json

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
    parser.add_argument("img_path", type=str, help="Path to the image")
    parser.add_argument("--model", type=str, default="Facenet512", choices=MODELS, help="Face recognition model")
    parser.add_argument("--detector_backend", type=str, default="mediapipe", choices=DETECTOR_BACKENDS, help="Face detector and alignment backend")
    return parser.parse_args()


def main():
    args = parse_args()
    embeddings = process_image(args.img_path, args.model, args.detector_backend)
    faces_count = len(embeddings)

    output = {
        "faces_count": faces_count,
    }

    print(json.dumps(output))


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
