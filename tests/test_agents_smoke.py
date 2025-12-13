import streamlit as stlit
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from main import run_agents_based_on_keywords

# Initialize session_state for testing
stlit.session_state.chat_history = []

# Patch the agent instances in main
@patch("main.statistics_agent_instance")
@patch("main.prediction_agent_instance")
def test_smoke_agents(mock_prediction_agent, mock_stats_agent):
    # Mock the .stream method to return fake messages
    mock_stats_agent.stream.return_value = [{"messages": [SimpleNamespace(content="Stats result")]}]
    mock_prediction_agent.stream.return_value = [{"messages": [SimpleNamespace(content="Prediction result")]}]

    # Call the function, should not raise errors
    run_agents_based_on_keywords("show me player stats")
    run_agents_based_on_keywords("who will likely score?")
