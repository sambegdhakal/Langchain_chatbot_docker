from unittest.mock import MagicMock
from main import run_agents_based_on_keywords,statistics_agent_instance,prediction_agent_instance
from types import SimpleNamespace


def test_run_agents_statistics(monkeypatch):
    # Mock agent response
    monkeypatch.setattr(
        statistics_agent_instance, 
        "stream", 
        MagicMock(return_value=[{"messages": [SimpleNamespace(content="Stats result")]}])
    )

    response = run_agents_based_on_keywords("show me player stats")
    assert "Stats" in response

def test_run_agents_predictions(monkeypatch):
    monkeypatch.setattr(
        statistics_agent_instance, 
        "stream", 
        MagicMock(return_value=[{"messages": [SimpleNamespace(content="Stats result")]}])
    )

    monkeypatch.setattr(
        prediction_agent_instance, 
        "stream", 
        MagicMock(return_value=[{"messages": [SimpleNamespace(content="Prediction result")]}])
    )

    response = run_agents_based_on_keywords("Predict who will likely score?")
    assert "Prediction" in response