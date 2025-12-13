from unittest.mock import patch, MagicMock
from main import run_agents_based_on_keywords
from types import SimpleNamespace

@patch("main.statistics_agent_instance.stream")
@patch("main.prediction_agent_instance.stream")
@patch("main.OllamaEmbeddings")  # Mock embeddings so it doesn't call Ollama
@patch("main.InMemoryVectorStore")  # Mock vector store
def test_smoke_agents(mock_vectorstore, mock_embeddings, mock_prediction, mock_stats):
    # Mock return values so it doesn't call real APIs
    mock_stats.return_value = [{"messages": [SimpleNamespace(content="Stats result")]}]
    mock_prediction.return_value = [{"messages": [SimpleNamespace(content="Prediction result")]}]

    # Mock vector store methods
    mock_vectorstore_instance = MagicMock()
    mock_vectorstore.return_value = mock_vectorstore_instance

    # Mock embeddings instance
    mock_embeddings_instance = MagicMock()
    mock_embeddings.return_value = mock_embeddings_instance

    # Run agents — just ensure they don't raise exceptions
    result1 = run_agents_based_on_keywords("show me player stats")
    result2 = run_agents_based_on_keywords("who will likely score?")

    # Basic assertions
    assert result1 == "Stats result"
    assert result2 == "Prediction result"
