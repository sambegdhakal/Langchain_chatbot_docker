# tests/test_agents_smoke.py
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

# Patch agent instances BEFORE importing main
with patch("main.statistics_agent_instance") as MockStatsAgent, \
     patch("main.prediction_agent_instance") as MockPredictionAgent:

    # Mock the .stream method for both agents
    MockStatsAgent.stream.return_value = [{"messages": [SimpleNamespace(content="Stats result")]}]
    MockPredictionAgent.stream.return_value = [{"messages": [SimpleNamespace(content="Prediction result")]}]

    # Import the function after patching
    from main import run_agents_based_on_keywords

def test_smoke_statistics_agent():
    """Smoke test: run statistics agent only"""
    run_agents_based_on_keywords("show me player stats")

def test_smoke_prediction_agent():
    """Smoke test: run prediction agent with keywords"""
    run_agents_based_on_keywords("who will likely score?")
