# tests/test_agents_units.py
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

# Patch top-level code in main.py BEFORE importing
with patch("main.OllamaEmbeddings") as MockEmbeddings, \
     patch("main.InMemoryVectorStore") as MockVectorStore, \
     patch("main.statistics_agent_instance.stream") as mock_stats, \
     patch("main.prediction_agent_instance.stream") as mock_prediction:

    from main import run_agents_based_on_keywords  # import after patching

    # Setup mocks
    mock_stats.return_value = [{"messages": [SimpleNamespace(content="Stats result")]}]
    mock_prediction.return_value = [{"messages": [SimpleNamespace(content="Prediction result")]}]
    MockVectorStore.return_value = MagicMock()
    MockEmbeddings.return_value = MagicMock()

def test_run_agents_statistics():
    response = run_agents_based_on_keywords("show me player stats")
    assert "Stats result" in response

def test_run_agents_predictions():
    response = run_agents_based_on_keywords("Predict who will likely score?")
    assert "Prediction result" in response
