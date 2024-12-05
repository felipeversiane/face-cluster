# Face Cluster

This project uses [Docker](https://www.docker.com) to manage dependencies and containerize the environment, enabling easy deployment and execution of facial recognition and clustering tasks.

## Prerequisites:

Ensure you have the following installed:

- Docker installed on your system.
- Docker Compose for managing multi-container applications.

## Libs:

The system leverages Python libraries such as:

- OpenCV: Utilized for image processing and face detection.
- Scikit-learn: Used for clustering the facial encodings using algorithms like DBSCAN.
- Imutils: A utility library to assist with common image-processing tasks.
- Numpy: Essential for numerical operations, especially when handling large sets of encodings.
- Face Recognition: A deep learning-based library for detecting and encoding faces.

## Installation:

Follow these steps to get the project up and running:

### 1. Clone repository

Use this to copy the source code in your machine:

```bash
git clone https://github.com/felipeversiane/face-cluster.git
```

### 2. Build and start the docker containers:

Use this to build, download the dependences and start containers : 

```bash
docker-compose up --build
```

## Directory structure:

- `/src` : Contains the Python source code for face encoding `encode_faces.py` and clustering `cluster_faces.py`.

- `/database` : Folder for storing the input images used for generating face encodings and clusters.

- `encodings/` : Directory where the serialized facial encodings will be saved after processing.

- `face_cluster/` : The output folder where clustered images and results will be saved.

## Environment Variables:

- ENCODINGS_PATH: Path to the serialized encodings file.

- FACE_DATA_PATH: Path to store face data during processing.

- CLUSTERING_RESULT_PATH: Path where clustering results will be saved.

## Usage in Docker:

- `encode-faces` service processes images and generates face encodings.

- `cluster-faces` service uses DBSCAN to group faces into clusters, storing them in the designated result path.

## Notes:

- The project is designed to be extensible, allowing additional services like a web interface for image upload or real-time face recognition.

- The current focus is on the core functionality of encoding and clustering faces.

