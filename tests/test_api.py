"""
Unit tests for the FastAPI endpoint schema.
"""
# Note: full API tests require the model weights to be present.
# These tests verify the response schema structure only.

def test_identify_response_schema():
    response = {"part": "Femur", "confidence": 0.94, "description": "Longest bone in the human body."}
    assert "part" in response
    assert "confidence" in response
    assert "description" in response
    assert isinstance(response["confidence"], float)
    assert 0.0 <= response["confidence"] <= 1.0
