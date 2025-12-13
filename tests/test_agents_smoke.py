import streamlit as stlit
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from main import run_agents_based_on_keywords

# Initialize session_state for testing
stlit.session_state.chat_history = []

# Patch the agents so no real API calls happen
with patch("main.statistics_agent_instance") as MockStatsAgent, \
     patch("main.prediction_agent_instance") as MockPredictionAgent:

    MockStatsAgent.stream.return_value = [{"messages": [SimpleNamespace(content="Stats result")]}]
    MockPredictionAgent.stream.return_value = [{"messages": [SimpleNamespace(content="Prediction result")]}]

    def test_smoke_statistics_agent():
        """Smoke test: run statistics agent only"""
        run_agents_based_on_keywords("show me player stats")

    def test_smoke_prediction_agent():
        """Smoke test: run prediction agent with keywords"""
        run_agents_based_on_keywords("who will likely score?")
