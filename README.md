# Image Processing API

This project provides an API to process image data stored in a CSV file. The application is built using FastAPI, Dask, and MongoDB, and is Dockerized for easy deployment.

## Features

- Resize image width from 200 to 150 pixels.
- Store resized image in MongoDB.
- Request image frames based on `depth_min` and `depth_max`.
- Apply a custom color map to the generated frames.
- Fully Dockerized for easy deployment.

## Requirements Containerization

- Docker
- Docker Compose

## Setup Via Docker

### 1. Clone the repository
### 2. Nav to the project dir
### 3. Up the docker
### 4. Run the test


```bash
git clone https://github.com/abdelrhman-elsbagh/Depth-Image-Processor
cd "Depth-Image-Processor"
docker-compose up --build
docker-compose run test
```


## Requirements For Setup Locally

- Python 3.10.12
- MongoDB

## Setup Via Docker

### 1. Clone the repository
### 2. Nav to the project dir
### 3. Up the docker
### 4. Install requirements.txt
### 5. Run the test


```bash
git clone https://github.com/abdelrhman-elsbagh/Depth-Image-Processor
cd "Depth-Image-Processor"
pip install -r requirements.txt
python3 main.py
pytest
```

## The App Already Deployed on 
### 13.60.8.235

### Curl Command for Get Image Frame

```bash
curl -X POST "http://13.60.8.235:8000/get-image-frame/" -H "Content-Type: application/json" -d '{
  "depth_min": 9000.0,
  "depth_max": 9500.0
}'
```

### Curl Command for Get Colored Image Frame

```bash
curl -X POST "http://13.60.8.235:8000/get-colored-image-frame/" -H "Content-Type: application/json" -d '{
  "depth_min": 9000.0,
  "depth_max": 9500.0
}'

```



