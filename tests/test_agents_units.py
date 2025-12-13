# tests/test_agents_units.py
import sys
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

# Patch embeddings, vector store, and agent streams before importing main
with patch("main.OllamaEmbeddings") as MockEmbeddings, \
     patch("main.InMemoryVectorStore") as MockVectorStore, \
     patch("main.statistics_agent_instance.stream") as mock_stats, \
     patch("main.prediction_agent_instance.stream") as mock_prediction:

    # Setup mocks
    MockEmbeddings.return_value = MagicMock()
    MockVectorStore.return_value = MagicMock()
    mock_stats.return_value = [{"messages": [SimpleNamespace(content="Stats result")]}]
    mock_prediction.return_value = [{"messages": [SimpleNamespace(content="Prediction result")]}]

    # Import after patching to prevent top-level code from executing real API calls
    from main import run_agents_based_on_keywords

def test_run_agents_statistics():
    """Test that statistics agent returns mocked stats."""
    response = run_agents_based_on_keywords("show me player stats")
    assert "Stats result" in response

def test_run_agents_predictions():
    """Test that prediction agent returns mocked prediction."""
    response = run_agents_based_on_keywords("Predict who will likely score?")
    assert "Prediction result" in response
