import os
import sys
import pandas as pd
from dotenv import load_dotenv
import streamlit as stlit
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from tools.commentary_tool import get_commentary_tool
from tools.player_tool import get_player_stat_tool
from tools.team_standings_tool import get_standings_tool
from tools.top_scorer_tool import get_top_scorer
from tools.form_tool import get_form_tool
from tools.team_fixture_result_tool import get_fixture_result_tool
from agents.stats_agent import statistics_agent
from agents.prediction_agent import prediction_agent

# Load environment variables
load_dotenv()

# Globals to be initialized
statistics_agent_instance = None
prediction_agent_instance = None
vector_store = None

def running_in_docker() -> bool:
    return os.path.exists("/.dockerenv")

OLLAMA_BASE_URL = (
    "http://host.docker.internal:11434"
    if running_in_docker()
    else "http://localhost:11434"
)

def initialize_agents(run_checks=True,nrows_commentary=50): # if only limited needed
    global statistics_agent_instance, prediction_agent_instance, vector_store
    
    # Set base dataset path from environment variable or default
    DATA_PATH = os.getenv("DATA_PATH", "datasets")

    # Only run these checks if Streamlit is running
    if run_checks:
        # Check if .env exists
        # GROQ API key
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            stlit.error("❌ GROQ_API_KEY not set. Please provide your key in a .env file or via environment variable.")
            stlit.stop()

        # Check required dataset folders
        required_dataset_paths = [
            os.path.join(DATA_PATH,"commentary_data"),
            os.path.join(DATA_PATH,"playerStats_data"),
            os.path.join(DATA_PATH,"base_data")
        ]
        for path in required_dataset_paths:
            if not os.path.exists(path):
                stlit.error(f"❌ Error: Required dataset folder missing: {path}\nPlease provide your datasets.")
                return


    OLLAMA_HOST = "host.docker.internal" if running_in_docker() else "localhost"

    # Initialize LLMs
    llm_statistics = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key, max_tokens=512)
    llm_predictions = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key, max_tokens=512)
    # Alternative: llm_predictions = OllamaLLM(model="qwen3:4b")

    # Embeddings & vector store
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_BASE_URL)
    vector_store = InMemoryVectorStore(embeddings)

    # Load CSVs
    df_commentary = pd.read_csv(
        os.path.join(DATA_PATH,"commentary_data/commentary_2025_ENG.1.csv"),
        nrows=nrows_commentary,
        usecols=["commentaryText"]
    )
    df_playerstats = pd.read_csv(os.path.join(DATA_PATH,"playerStats_data/playerStats_2025_ENG.1.csv"))
    df_players = pd.read_csv(os.path.join(DATA_PATH,"base_data/players.csv"),low_memory=False)
    df_teams = pd.read_csv(os.path.join(DATA_PATH,"base_data/teams.csv"))
    df_team_standings = pd.read_csv(os.path.join(DATA_PATH,"base_data/standings.csv"))
    df_venue= pd.read_csv(os.path.join(DATA_PATH,"base_data/venues.csv"))
    df_leagues= pd.read_csv(os.path.join(DATA_PATH,"base_data/leagues.csv"))


    # Merge player stats with player info
    merged_df = df_playerstats.merge(df_players, on="athleteId", how="left")
    merged_df = merged_df.merge(df_teams, on="teamId", how="left")


    # Keep only the first record per teamId (highest timeStamp)
    # Sort team standings by teamId and timeStamp descending
    df_sorted = df_team_standings.sort_values(["teamId", "timeStamp"], ascending=[True, False])
    df_standings_selected = df_sorted.drop_duplicates(subset=["teamId"], keep="first")

    # Select only needed columns
    df_standings_selected = df_standings_selected[["form", "next_opponent", "teamId", "next_homeAway", "next_matchDateTime", "timeStamp","seasonType", "leagueId"]]


    df_teams_selected = df_teams[["name", "abbreviation","teamId","venueId"]].rename(
    columns={
        "name": "team_name",
        "abbreviation": "team_abbreviation"
    }
    )


    df_venue_selected = df_venue[["fullName","venueId"]].rename(
    columns={
        "fullName": "stadium_name",
    }
    )
    
    df_leagues_selected= df_leagues[["seasonType", "leagueId", "seasonName"]]

    # Select columns
    player_info_cols = [col for col in ['fullName','name', 'abbreviation', 'displayName'] if col in merged_df.columns]
    stats_columns = [col for col in df_playerstats.columns if col not in ['athleteId','teamId','timestamp']]
    final_stats_df = merged_df[player_info_cols + stats_columns]

    #Merge team standings with team and venue info
    merged_df_standings = (
    df_teams_selected
    .merge(df_standings_selected, on="teamId", how="left")
    .merge(df_venue_selected, on="venueId", how="left")
    )

    # Add next opponent name
    team_id_to_name = df_teams_selected.set_index("teamId")["team_name"].to_dict()
    merged_df_standings["next_opponent_name"] = merged_df_standings["next_opponent"].map(team_id_to_name)

    # Drop unnecessary columns
    merged_df_standings = merged_df_standings.drop(columns=["venueId", "teamId", "next_opponent", "timeStamp"])

    merged_df_standings_final= merged_df_standings.merge(df_leagues_selected, on=["seasonType", "leagueId"], how="inner")

    # Drop the seasonType column
    merged_df_standings_final = merged_df_standings_final.drop(columns=["seasonType", "leagueId"])

    # Define the leagues of interest
    leagues_pattern = "English Premier League|UEFA Champions League|UEFA Europa League|UEFA Conference League"

    # Filter rows
    merged_df_standings_final = merged_df_standings_final[
        merged_df_standings_final["seasonName"].str.contains(leagues_pattern, case=False) &
        merged_df_standings_final["seasonName"].str.contains("25-26", case=False)
    ]
    
    # Convert commentary to LangChain Documents
    docs_commentary = [
    Document(page_content=text, metadata={"row": idx, "source": "commentary_2025_ENG.1.csv"})
    for idx, text in df_commentary["commentaryText"].dropna().items()
    ]

    stats_texts = final_stats_df.astype(str).agg(
    lambda x: "\n".join(f"{col}: {val}" for col, val in x.items()),
    axis=1
    )

    # Convert player stats to Documents
    docs_stats = [
        Document(
            page_content=text,
            metadata={"row": idx, "source": "playerStats_2025_ENG.1.csv"}
        )
        for idx, text in stats_texts.items()
    ]

    standings_texts = merged_df_standings_final.astype(str).agg(
    lambda x: "\n".join(f"{col}: {val}" for col, val in x.items()),
    axis=1
    )

    # Convert team standings to Documents
    docs_standings = [
        Document(
            page_content=text,
            metadata={"row": idx, "source": "standings.csv"}
        )
        for idx, text in standings_texts.items()
    ]

    # Add documents to vector store
    vector_store.add_documents(docs_commentary, batch_size=50)
    vector_store.add_documents(docs_stats, batch_size=128)
    vector_store.add_documents(docs_standings, batch_size=40)

    # Initialize tools
    commentary_tool = get_commentary_tool(vector_store)
    player_stats_tool = get_player_stat_tool(vector_store)
    top_scorer_tool = get_top_scorer(final_stats_df)
    form_tool = get_form_tool(vector_store)

    tools_stats = [commentary_tool, player_stats_tool, top_scorer_tool,form_tool]
    tools_predict = []

    # Initialize agents
    statistics_agent_instance = statistics_agent(tools=tools_stats, llm=llm_statistics)
    prediction_agent_instance = prediction_agent(tools=tools_predict, llm=llm_predictions)


def run_agents_based_on_keywords(query: str):
    # Ensure agents are initialized
    if "statistics_agent_instance" not in stlit.session_state or \
       stlit.session_state.statistics_agent_instance is None:
        stlit.error("Agents are not initialized yet.")
        return "Error: Agents not initialized."
    
    statistics_agent_instance = stlit.session_state.statistics_agent_instance
    prediction_agent_instance = stlit.session_state.prediction_agent_instance
    config = {"max_retries": 0}
    
    # Keywords to decide if prediction is needed
    prediction_keywords = [
    "expect", "predict", "forecast", "projection", "estimate", "likely", "probable",
    "future", "next", "will", "anticipate", "chance", "possibility", "potential",
    "outcome", "trend", "prediction", "projected", "expected", "suppose", "guess",
    "forecasted", "may", "might", "plan", "predicting", "predictable"]

    run_prediction = any(keyword.lower() in query.lower() for keyword in prediction_keywords)

    #Run statistics agent
    messages = stlit.session_state.chat_history + [{"role": "user", "content": query}]
    stats_messages = []
    for event in statistics_agent_instance.stream({"messages": messages}, config=config, stream_mode="values"):
        stats_messages.extend(event.get("messages", []))

    if not stats_messages:
        return "No statistics available."

    stats_content = stats_messages[-1].content

    #if prediction keywords seen
    if run_prediction:
        prediction_messages = []
        
        # Combine stats output with user query
        prediction_input = [
            {"role": "user", "content": stats_content},
            {"role": "user", "content": query}
        ]

        for event in prediction_agent_instance.stream({"messages": prediction_input}, 
                                                      config=config, stream_mode="values"):
            prediction_messages.extend(event.get("messages", []))

        result = prediction_messages[-1].content if prediction_messages else "No prediction available."
        return result
    else:
        # If no prediction, just return stats
        return stats_content


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
    # Initialize agents only once
    if "agents_initialized" not in stlit.session_state:
        initialize_agents(run_checks=True)
        
        # Save the initialized agents and vector store to session_state
        stlit.session_state.statistics_agent_instance = statistics_agent_instance
        stlit.session_state.prediction_agent_instance = prediction_agent_instance
        stlit.session_state.vector_store = vector_store
        
        stlit.session_state.agents_initialized = True

    run_streamlit()
