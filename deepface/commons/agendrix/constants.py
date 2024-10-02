# MODELS = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace', 'SFace', 'GhostFaceNet']
MODELS = ["VGG-Face", "Facenet", "Facenet512", "OpenFace"]
DEFAULT_MODEL = "Facenet512"

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
DEFAULT_DETECTOR_BACKEND = "ssd"

# DISTANCE_METRICS = ['cosine', 'euclidean', 'euclidean_l2']
DISTANCE_METRICS = ["cosine"]
DEFAULT_DISTANCE_METRIC = "cosine"
