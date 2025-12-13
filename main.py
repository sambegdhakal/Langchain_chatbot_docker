import os
import sys
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.vectorstores import InMemoryVectorStore
import pandas as pd 
from langchain_core.documents import Document
from lang_chain_project_docker.tools.commentary_tool import get_commentary_tool
from lang_chain_project_docker.tools.player_tool import get_player_stat_tool
from lang_chain_project_docker.tools.top_scorer_tool import get_top_scorer
from lang_chain_project_docker.agents.stats_agent import statistics_agent
from lang_chain_project_docker.agents.prediction_agent import prediction_agent
from lang_chain_project_docker.download_data import download
from langchain_ollama.llms import OllamaLLM
import streamlit as stlit

# Check if .env exists
if not os.path.exists(".env"):
    stlit.error("❌ Missing .env file. Some features may not work.")
    stlit.stop() 

# Check if datasets folder exists
required_dataset_paths = [
    "datasets/commentary_data",
    "datasets/playerStats_data",
    "datasets/base_data"
]

for path in required_dataset_paths:
    if not os.path.exists(path):
        stlit.error(f"❌ Error: Required dataset folder missing: {path}\nPlease provide your datasets.")
        stlit.stop()

#download data
#download()

#loading environment variables
load_dotenv()

# page setup
stlit.set_page_config(
    page_title="Fotbot",
    page_icon="⚽",
    layout="centered",
)
stlit.title("💬⚽ Soccer stats and prediction")

# initiate chat history
if "chat_history" not in stlit.session_state:
    stlit.session_state.chat_history = []

# show chat history
for message in stlit.session_state.chat_history:
    with stlit.chat_message(message["role"]):
        stlit.markdown(message["content"])

#storing groq_key
groq_key = os.getenv("GROQ_API_KEY")

#defining first model from GROQ for statistics
llm_statistics = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key, max_tokens=512)

#defining second model either from GROQ or Ollama for predictions, I can switch between models without worrying about underlying framework
#llm_predictions =  OllamaLLM(model="qwen3:4b")
llm_predictions = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key, max_tokens=512)


# Embeddings; using Ollama
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# creating a vector store
vector_store = InMemoryVectorStore(embeddings)

# Loading commentry CSV file using pandas
df = pd.read_csv("datasets/commentary_data/commentary_2025_ENG.1.csv",nrows=50,
    usecols=["commentaryText"])

# df = pd.read_csv("datasets/commentary_data/commentary_2025_ENG.1.csv",
#     usecols=["commentaryText"])

#load files for player stats
playerstats_df = pd.read_csv("datasets/playerStats_data/playerStats_2025_ENG.1.csv")
player_df = pd.read_csv("datasets/base_data/players.csv", low_memory=False)
team_df = pd.read_csv("datasets/base_data/teams.csv")

# Merge player stats with player info
merged_df = playerstats_df.merge(player_df, on="athleteId", how="left")

# Merge with team info
merged_df = merged_df.merge(team_df, on="teamId", how="left")

# selected columns from player and the team
player_info_cols = []
for col in ['fullName','name', 'abbreviation', 'displayName']:
    if col in merged_df.columns:
        player_info_cols.append(col)

# Keep only desired columns: player & team info + stats from playerstats_df
stats_columns = [col for col in playerstats_df.columns if col not in ['athleteId','teamId', 'timestamp']]
final_stats_df = merged_df[player_info_cols + stats_columns]


# selecting only column from the commentary csv
selected_column = "commentaryText"
texts = df[selected_column].dropna().tolist()

# Converting each row of commentry into a LangChain Document
docs_commentary = [
    Document(
        page_content=text,
        metadata={"row": idx, "source": "commentary_2025_ENG.1.csv"}
    )
    for idx, text in enumerate(texts)
]

# Converting each row of palyer stats into a langchain Document
docs_stats = [
    Document(
        page_content="\n".join(f"{col}: {row[col]}" for col in final_stats_df.columns),
        metadata={
            "row": idx,
            "source": "playerStats_2025_ENG.1.csv"
        }
    )
    for idx, row in final_stats_df.iterrows()
]

#add documents to vector store
vector_store.add_documents(docs_commentary)
vector_store.add_documents(docs_stats)

#intializing tools
commentary_tool=get_commentary_tool(vector_store,docs_commentary)
player_stats_tool=get_player_stat_tool(vector_store,docs_stats)
top_scorer_tool=get_top_scorer(final_stats_df)

tools_stats = [commentary_tool,player_stats_tool,top_scorer_tool]
tools_predict=[]


#Initializing Agents
statistics_agent_instance = statistics_agent(tools=tools_stats, llm=llm_statistics)
prediction_agent_instance=prediction_agent(tools=tools_predict, llm=llm_predictions)



def run_agents_based_on_keywords(query: str):
    """
    Run statistics agent always. Run prediction agent only if query contains
    certain keywords indicating forecasts.
    """
    config = {"max_retries": 0}

    # Keywords indicating prediction/future analysis
    prediction_keywords = ["expect", "predict", "future", "forecast", "likely", "projection", "next"]

    # Check if any keyword exists in the query (case-insensitive)
    run_prediction = any(keyword.lower() in query.lower() for keyword in prediction_keywords)

    # Combine previous chat history with the current user query
    messages = stlit.session_state.chat_history + [{"role": "user", "content": query}]
    stats_messages = []

    #running statistics_agent with query
    for event in statistics_agent_instance.stream(
        {"messages": messages},
        config=config,
        stream_mode="values",
    ):
        # accumulate messages
        stats_messages.extend(event.get("messages", []))

    if not stats_messages:
        return "No statistics available."
    
    # Running prediction agent with statistics output
    if run_prediction:
        prediction_messages = []
        for event in prediction_agent_instance.stream(
            {"messages": stats_messages},
            config=config,
            stream_mode="values",
        ):
            prediction_messages.extend(event.get("messages", []))

        if prediction_messages:
            return prediction_messages[-1].content
        else:
            return "No prediction available."

    else:
        #if prediction is not needed only print stats message
        return stats_messages[-1].content

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
