from types import SimpleNamespace
from main import run_agents_based_on_keywords
from unittest.mock import MagicMock


def test_run_agents_statistics(monkeypatch):
    # Replace the global statistics_agent_instance with a MagicMock
    mock_stats_agent = MagicMock()
    mock_stats_agent.stream.return_value = [{"messages": [SimpleNamespace(content="Stats result")]}]
    monkeypatch.setattr("main.statistics_agent_instance", mock_stats_agent)

    response = run_agents_based_on_keywords("show me player stats")
    assert "Stats" in response

def test_run_agents_predictions(monkeypatch):
    mock_stats_agent = MagicMock()
    mock_stats_agent.stream.return_value = [{"messages": [SimpleNamespace(content="Stats result")]}]
    monkeypatch.setattr("main.statistics_agent_instance", mock_stats_agent)

    mock_pred_agent = MagicMock()
    mock_pred_agent.stream.return_value = [{"messages": [SimpleNamespace(content="Prediction result")]}]
    monkeypatch.setattr("main.prediction_agent_instance", mock_pred_agent)

    response = run_agents_based_on_keywords("Predict who will likely score?")
    assert "Prediction" in response
