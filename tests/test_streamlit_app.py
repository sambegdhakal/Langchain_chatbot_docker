import streamlit as stlit

def test_streamlit_session_state():
    stlit.session_state.chat_history = []
    stlit.session_state.chat_history.append({"role": "user", "content": "Hi"})
    stlit.session_state.chat_history.append({"role": "user", "content": "Tell me about Nepal"})
    assert len(stlit.session_state.chat_history) == 2, "stream lit app not working"