
import pytest
import json
from unittest.mock import patch
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        # Mock the heavy model inference to return a safe legitimate prediction by default
        with patch('app.model_predict') as mock_pred:
            mock_pred.return_value = (0, 0.1)  # (label, confidence)
            yield client

def test_health_check(client):
    """Verify the API health endpoint is alive."""
    rv = client.get('/api/health')
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert data['status'] == 'ok'

def test_prediction_hybrid_logic(client):
    """
    Verify the hybrid detection logic.
    Even if the ML model is 'unsure' (mocked to 0.1), 
    the rule engine should catch high-risk signals.
    """
    # A message with a suspicious .biz URL
    test_message = {"text": "URGENT: Verify at http://secure-portal.biz"}
    
    # We don't need to patch model_predict here because the fixture already does it!
    # But for this specific test, let's ensure it returns a low score to prove the rule override works.
    rv = client.post('/api/predict', 
                     data=json.dumps(test_message),
                     content_type='application/json')
    
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert data['risk_level'] == 'HIGH'
    assert data['prediction'] == 'Phishing'
    assert data['override_reason'] == 'suspicious_url'

def test_keywords_lexicon(client):
    """Verify the keywords endpoint returns the detection categories."""
    rv = client.get('/api/keywords')
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert 'categories' in data
    # Check for some of our 9 categories
    assert 'urgency' in data['categories']
    assert 'aviation_sector' in data['categories']

def test_demo_endpoint(client):
    """Verify the demo endpoint provides pre-built scenarios."""
    rv = client.get('/api/demo')
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert 'examples' in data
    assert len(data['examples']) > 0
    assert 'text' in data['examples'][0]
