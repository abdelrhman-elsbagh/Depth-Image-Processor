import sys
import os
import pytest
import time
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app, image_processor

client = TestClient(app)

IMAGE_FRAME_RESPONSE_TIME_THRESHOLD = 2
COLORED_IMAGE_FRAME_RESPONSE_TIME_THRESHOLD = 3

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
    assert "frame_image_url" in response.json()
    assert client.get(response.json()["frame_image_url"]).status_code == 200

    response = client.post("/get-image-frame/", json={"depth_min": 9546.0, "depth_max": 9546.0})
    assert response.status_code == 200
    assert "frame_image_url" in response.json()
    assert client.get(response.json()["frame_image_url"]).status_code == 200

def test_get_image_frame_invalid_input(setup_data):
    response = client.post("/get-image-frame/", json={"depth_min": "invalid", "depth_max": 9546.0})
    assert response.status_code == 422

    response = client.post("/get-image-frame/", json={"depth_min": 9000.0, "depth_max": "invalid"})
    assert response.status_code == 422

def test_get_colored_image_frame_boundary_conditions(setup_data):
    response = client.post("/get-colored-image-frame/", json={"depth_min": 9000.1, "depth_max": 9000.1, "color_map": "jet"})
    assert response.status_code == 200
    assert "colored_image_url" in response.json()
    assert client.get(response.json()["colored_image_url"]).status_code == 200

    response = client.post("/get-colored-image-frame/", json={"depth_min": 9546.0, "depth_max": 9546.0, "color_map": "jet"})
    assert response.status_code == 200
    assert "colored_image_url" in response.json()
    assert client.get(response.json()["colored_image_url"]).status_code == 200

def test_get_colored_image_frame_invalid_input(setup_data):
    response = client.post("/get-colored-image-frame/", json={"depth_min": "invalid", "depth_max": 9546.0, "color_map": "jet"})
    assert response.status_code == 422

    response = client.post("/get-colored-image-frame/", json={"depth_min": 9000.0, "depth_max": "invalid", "color_map": "jet"})
    assert response.status_code == 422

def test_get_image_frame_data_verification(setup_data):
    response = client.post("/get-image-frame/", json={"depth_min": 9000.1, "depth_max": 9546.0})
    assert response.status_code == 200
    frame_image_url = response.json()["frame_image_url"]
    assert client.get(frame_image_url).status_code == 200

def test_performance(setup_data):
    start_time = time.time()
    response = client.post("/get-image-frame/", json={"depth_min": 9000.1, "depth_max": 9546.0})
    assert response.status_code == 200
    assert time.time() - start_time < IMAGE_FRAME_RESPONSE_TIME_THRESHOLD  # Ensure the response is within threshold

    start_time = time.time()
    response = client.post("/get-colored-image-frame/", json={"depth_min": 9000.1, "depth_max": 9546.0, "color_map": "jet"})
    assert response.status_code == 200
    assert time.time() - start_time < COLORED_IMAGE_FRAME_RESPONSE_TIME_THRESHOLD
