import os
import sys
import pandas as pd
from types import SimpleNamespace
from dotenv import load_dotenv
import streamlit as stlit

# LangChain imports
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM

# Tools & agents
from tools.commentary_tool import get_commentary_tool
from tools.player_tool import get_player_stat_tool
from tools.top_scorer_tool import get_top_scorer
from agents.stats_agent import statistics_agent
from agents.prediction_agent import prediction_agent

# Globals to be initialized
statistics_agent_instance = None
prediction_agent_instance = None
vector_store = None


def initialize_agents(run_checks=True, nrows_commentary=50):
    global statistics_agent_instance, prediction_agent_instance, vector_store

    # Only run these checks if Streamlit is running
    if run_checks:
        # Check if .env exists
        if not os.path.exists(".env"):
            stlit.error("❌ Missing .env file. Some features may not work.")
            stlit.stop()

        # Check required dataset folders
        required_dataset_paths = [
            "datasets/commentary_data",
            "datasets/playerStats_data",
            "datasets/base_data"
        ]
        for path in required_dataset_paths:
            if not os.path.exists(path):
                stlit.error(f"❌ Error: Required dataset folder missing: {path}\nPlease provide your datasets.")
                stlit.stop()

    # Load environment variables
    load_dotenv()

    # GROQ API key
    groq_key = os.getenv("GROQ_API_KEY")

    # Initialize LLMs
    llm_statistics = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key, max_tokens=512)
    llm_predictions = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key, max_tokens=512)
    # Alternative: llm_predictions = OllamaLLM(model="qwen3:4b")

    # Embeddings & vector store
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = InMemoryVectorStore(embeddings)

    # Load CSVs
    df_commentary = pd.read_csv(
        "datasets/commentary_data/commentary_2025_ENG.1.csv",
        nrows=nrows_commentary,
        usecols=["commentaryText"]
    )
    df_playerstats = pd.read_csv("datasets/playerStats_data/playerStats_2025_ENG.1.csv")
    df_players = pd.read_csv("datasets/base_data/players.csv", low_memory=False)
    df_teams = pd.read_csv("datasets/base_data/teams.csv")

    # Merge player stats with player info
    merged_df = df_playerstats.merge(df_players, on="athleteId", how="left")
    merged_df = merged_df.merge(df_teams, on="teamId", how="left")

    # Select columns
    player_info_cols = [col for col in ['fullName','name', 'abbreviation', 'displayName'] if col in merged_df.columns]
    stats_columns = [col for col in df_playerstats.columns if col not in ['athleteId','teamId','timestamp']]
    final_stats_df = merged_df[player_info_cols + stats_columns]

    # Convert commentary to LangChain Documents
    docs_commentary = [
        Document(page_content=text, metadata={"row": idx, "source": "commentary_2025_ENG.1.csv"})
        for idx, text in enumerate(df_commentary["commentaryText"].dropna().tolist())
    ]

    # Convert player stats to Documents
    docs_stats = [
        Document(
            page_content="\n".join(f"{col}: {row[col]}" for col in final_stats_df.columns),
            metadata={"row": idx, "source": "playerStats_2025_ENG.1.csv"}
        )
        for idx, row in final_stats_df.iterrows()
    ]

    # Add documents to vector store
    vector_store.add_documents(docs_commentary)
    vector_store.add_documents(docs_stats)

    # Initialize tools
    commentary_tool = get_commentary_tool(vector_store, docs_commentary)
    player_stats_tool = get_player_stat_tool(vector_store, docs_stats)
    top_scorer_tool = get_top_scorer(final_stats_df)

    tools_stats = [commentary_tool, player_stats_tool, top_scorer_tool]
    tools_predict = []

    # Initialize agents
    statistics_agent_instance = statistics_agent(tools=tools_stats, llm=llm_statistics)
    prediction_agent_instance = prediction_agent(tools=tools_predict, llm=llm_predictions)


def run_agents_based_on_keywords(query: str):
    """
    Run statistics agent always. Run prediction agent only if query contains
    certain keywords indicating forecasts.
    """
    config = {"max_retries": 0}
    prediction_keywords = ["expect", "predict", "future", "forecast", "likely", "projection", "next"]

    # Determine if prediction agent should run
    run_prediction = any(keyword.lower() in query.lower() for keyword in prediction_keywords)

    # Combine previous chat history with current query
    messages = stlit.session_state.chat_history + [{"role": "user", "content": query}]
    stats_messages = []

    # Run statistics agent
    for event in statistics_agent_instance.stream({"messages": messages}, config=config, stream_mode="values"):
        stats_messages.extend(event.get("messages", []))

    if not stats_messages:
        return "No statistics available."

    # Run prediction agent if needed
    if run_prediction:
        prediction_messages = []
        for event in prediction_agent_instance.stream({"messages": stats_messages}, config=config, stream_mode="values"):
            prediction_messages.extend(event.get("messages", []))

        return prediction_messages[-1].content if prediction_messages else "No prediction available."
    else:
        return stats_messages[-1].content


def run_streamlit():
    """Run the Streamlit chat interface."""
    stlit.set_page_config(page_title="Fotbot", page_icon="⚽", layout="centered")
    stlit.title("💬⚽ Soccer stats and prediction")

    # Initiate chat history
    if "chat_history" not in stlit.session_state:
        stlit.session_state.chat_history = []

    # Show previous messages
    for message in stlit.session_state.chat_history:
        with stlit.chat_message(message["role"]):
            stlit.markdown(message["content"])

    user_prompt = stlit.chat_input("Ask Fotbot ⚽⚽⚽")
    if user_prompt:
        # Display user message
        with stlit.chat_message("user"):
            stlit.markdown(user_prompt)

        # Save user message
        stlit.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Get bot response
        ai_response = run_agents_based_on_keywords(user_prompt)

        # Display bot message
        with stlit.chat_message("assistant"):
            stlit.markdown(ai_response)

        # Save bot response
        stlit.session_state.chat_history.append({"role": "assistant", "content": ai_response})


if __name__ == "__main__":
    # Only run checks when executing Streamlit
    os.environ["RUNNING_STREAMLIT"] = "1"
    initialize_agents(run_checks=True)
    run_streamlit()
