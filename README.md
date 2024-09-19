## Installation

We use PDM for dependency management.

```
brew install pdm
```

Install dependencies

```
pdm install
```

## Running scripts

Detecting the number of faces :

```
usage: detect.py [-h] [--model {VGG-Face,Facenet,Facenet512,OpenFace}] [--detector_backend {opencv,retinaface,mtcnn,ssd,dlib,mediapipe,yolov8,centerface}] [--redis_key REDIS_KEY] img_path

positional arguments:
  img_path              Path to the image

options:
  -h, --help            show this help message and exit
  --model {VGG-Face,Facenet,Facenet512,OpenFace}
                        Face recognition model
  --detector_backend {opencv,retinaface,mtcnn,ssd,dlib,mediapipe,yolov8,centerface}
                        Face detector and alignment backend
  --redis_key REDIS_KEY
                        Key to use to store results in Redis
```

Verifying the presence of a person on 2 images :

```
usage: verify.py [-h] [--model {VGG-Face,Facenet,Facenet512,OpenFace}] [--detector_backend {opencv,retinaface,mtcnn,ssd,dlib,mediapipe,yolov8,centerface}] [--distance_metric {cosine}] [--redis_key REDIS_KEY] img1_path img2_path

positional arguments:
  img1_path             Path to the first image
  img2_path             Path to the second image

options:
  -h, --help            show this help message and exit
  --model {VGG-Face,Facenet,Facenet512,OpenFace}
                        Face recognition model
  --detector_backend {opencv,retinaface,mtcnn,ssd,dlib,mediapipe,yolov8,centerface}
                        Face detector and alignment backend
  --distance_metric {cosine}
                        Type of distance metric to use
  --redis_key REDIS_KEY
                        Key to use to store results in Redis
```

## Notes

- Images can be passed as paths, URLs, or base60 strings.

# Building the package for release

```
pdm build
```
