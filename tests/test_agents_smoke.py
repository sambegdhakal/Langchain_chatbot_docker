from unittest.mock import patch
from lang_chain_project_docker.main import run_agents_based_on_keywords
from types import SimpleNamespace

@patch("lang_chain_project_docker.main.statistics_agent_instance.stream")
@patch("lang_chain_project_docker.main.prediction_agent_instance.stream")
def test_smoke_agents(mock_prediction, mock_stats):
    # Mock return values so it doesn't call real APIs
    mock_stats.return_value=[{"messages": [SimpleNamespace(content="Stats result")]}]
    mock_prediction.return_value = [{"messages": [SimpleNamespace(content="Stats result")]}]

    # Run agents — just ensure they don't raise exceptions
    run_agents_based_on_keywords("show me player stats")
    run_agents_based_on_keywords("who will likely score?")