from types import SimpleNamespace
from main import run_agents_based_on_keywords
from unittest.mock import MagicMock
import streamlit as st

def test_run_agents_statistics(monkeypatch):
    # Mock Streamlit session_state
    st.session_state.statistics_agent_instance = MagicMock()
    st.session_state.statistics_agent_instance.stream.return_value = [
        {"messages": [SimpleNamespace(content="Stats result")]}
    ]
    st.session_state.prediction_agent_instance = None
    st.session_state.chat_history = []

    response = run_agents_based_on_keywords("show me player stats")
    assert "Stats" in response

def test_run_agents_predictions(monkeypatch):
    st.session_state.statistics_agent_instance = MagicMock()
    st.session_state.statistics_agent_instance.stream.return_value = [
        {"messages": [SimpleNamespace(content="Stats result")]}
    ]

    st.session_state.prediction_agent_instance = MagicMock()
    st.session_state.prediction_agent_instance.stream.return_value = [
        {"messages": [SimpleNamespace(content="Prediction result")]}
    ]

    st.session_state.chat_history = []

    response = run_agents_based_on_keywords("Predict who will likely score?")
    assert "Prediction" in response
