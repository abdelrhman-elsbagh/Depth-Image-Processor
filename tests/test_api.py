import sys
import os
import pytest
import time
from fastapi.testclient import TestClient

# Add the project directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app, image_processor

client = TestClient(app)

# Define constants for acceptable response times
IMAGE_FRAME_RESPONSE_TIME_THRESHOLD = 2  # in seconds
COLORED_IMAGE_FRAME_RESPONSE_TIME_THRESHOLD = 3  # in seconds


@pytest.fixture(scope="module")
def setup_data():
    # Analyze and process data
    depth_range = image_processor.analyze_csv()
    image_processor.load_and_process_data()

    # Log depth range for debugging
    print(f"Depth range: {depth_range}")

    yield


def test_get_image_frame_boundary_conditions(setup_data):
    response = client.post("/get-image-frame/", json={"depth_min": 9000.1, "depth_max": 9000.1})
    assert response.status_code == 200
    assert "frame_image_path" in response.json()
    assert os.path.exists(response.json()["frame_image_path"])

    response = client.post("/get-image-frame/", json={"depth_min": 9546.0, "depth_max": 9546.0})
    assert response.status_code == 200
    assert "frame_image_path" in response.json()
    assert os.path.exists(response.json()["frame_image_path"])


def test_get_image_frame_invalid_input(setup_data):
    response = client.post("/get-image-frame/", json={"depth_min": "invalid", "depth_max": 9546.0})
    assert response.status_code == 422

    response = client.post("/get-image-frame/", json={"depth_min": 9000.0, "depth_max": "invalid"})
    assert response.status_code == 422


def test_get_colored_image_frame_boundary_conditions(setup_data):
    response = client.post("/get-colored-image-frame/", json={"depth_min": 9000.1, "depth_max": 9000.1})
    assert response.status_code == 200
    assert "colored_image_path" in response.json()
    assert os.path.exists(response.json()["colored_image_path"])

    response = client.post("/get-colored-image-frame/", json={"depth_min": 9546.0, "depth_max": 9546.0})
    assert response.status_code == 200
    assert "colored_image_path" in response.json()
    assert os.path.exists(response.json()["colored_image_path"])


def test_get_colored_image_frame_invalid_input(setup_data):
    response = client.post("/get-colored-image-frame/", json={"depth_min": "invalid", "depth_max": 9546.0})
    assert response.status_code == 422

    response = client.post("/get-colored-image-frame/", json={"depth_min": 9000.0, "depth_max": "invalid"})
    assert response.status_code == 422


def test_get_image_frame_data_verification(setup_data):
    response = client.post("/get-image-frame/", json={"depth_min": 9000.1, "depth_max": 9546.0})
    assert response.status_code == 200
    frame_image_path = response.json()["frame_image_path"]
    assert os.path.exists(frame_image_path)
    assert os.path.getsize(frame_image_path) > 0  # Ensure the file is not empty


def test_performance(setup_data):
    start_time = time.time()
    response = client.post("/get-image-frame/", json={"depth_min": 9000.1, "depth_max": 9546.0})
    assert response.status_code == 200
    assert time.time() - start_time < IMAGE_FRAME_RESPONSE_TIME_THRESHOLD  # Ensure the response is within threshold

    start_time = time.time()
    response = client.post("/get-colored-image-frame/", json={"depth_min": 9000.1, "depth_max": 9546.0})
    assert response.status_code == 200
    assert time.time() - start_time < COLORED_IMAGE_FRAME_RESPONSE_TIME_THRESHOLD  # Ensure the response is within threshold
