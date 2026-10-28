from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_parameters_endpoint():
    response = client.get("/parameters")
    assert response.status_code == 200
    assert "model_parameters" in response.json()

def test_prediction():
    # Test using Row 2 from your screenshot: African Elephant
    sample_payload = {
        "body_weight": 6654.0,
        "brain_weight": 5712.0,
        "max_life_span": 38.6,
        "gestation_time": 645.0,
        "predation_index": 3,
        "sleep_exposure_index": 5,
        "danger_index": 3
    }
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    result = response.json()
    assert "predicted_total_sleep_hours" in result
    assert isinstance(result["predicted_total_sleep_hours"], float)