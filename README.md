⚽ Fotbot – Soccer Stats & Predictions Chatbot

Fotbot is an interactive AI-powered chatbot that allows users to query soccer statistics, player performance, and predictions. It leverages LangChain, GROQ, Ollama, and Streamlit to provide fast, accurate insights using real-world soccer data.

📝 Features

Live chat interface using Streamlit.

Player statistics retrieval using historical data.

Match commentary insights from CSV datasets.

Top scorer analysis and predictions.

Automatic prediction agent for queries involving forecasts, future performance, or projections.

Flexible backend models using GROQ or Ollama.

⚡ Setup Instructions
1. Clone the repository
git clone <repository-url>
cd <repository-folder>

2. Install dependencies

Make sure you have Python 3.12+ installed, then run:

pip install -r requirements.txt


The requirements.txt file includes packages such as langchain, streamlit, pandas, dotenv, langchain_ollama, langchain_groq, etc.

3. Set up environment variables

Create a .env file in the root directory and add your API keys:

GROQ_API_KEY=<your_groq_api_key>


Store this file securely on your local drive.

4. Download datasets

The project uses the ESPN Soccer Data dataset from Kaggle:
https://www.kaggle.com/datasets/excel4soccer/espn-soccer-data

Go to ESPN Soccer Data on Kaggle

Download the datasets and save them under the datasets/ folder in your local project directory:

datasets/
├── base_data/
├── commentary_data/
└── playerStats_data/


Alternatively, you can run the included script to download datasets automatically:

python download_data.py


Ensure all CSV files are correctly saved in the datasets/ folder on your local drive.

🛠️ Running the Application
Option 1: Run with Streamlit locally
streamlit run main.py


The app will open in your browser.

Chat with Fotbot using the input box.

Ask for player stats, match commentary, or predictions.

The prediction agent triggers automatically when queries contain keywords like:
expect, predict, future, forecast, likely, projection, next.

Option 2: Run from Docker

Create a Dockerfile (if not already present):

# Use official Python 3.12 image
FROM python:3.12-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]


Build the Docker image

docker build -t fotbot-app .


Run the Docker container, mounting your local datasets and .env file:

docker run -it --rm \
  -p 8501:8501 \
  -v /path/to/your/local/project/datasets:/app/datasets \
  -v /path/to/your/local/project/.env:/app/.env \
  fotbot-app


Replace /path/to/your/local/project/ with your actual project path.

-p 8501:8501 maps Streamlit’s port to your host.

-v mounts the local datasets folder and .env file into the container.

Open your browser and go to:

http://localhost:8501


Chat with Fotbot as usual. The container uses your local datasets and GROQ API key.

Stop the container with CTRL+C or by closing the terminal.

📂 Project Structure
.
├── agents/
│   ├── prediction_agent.py
│   └── stats_agent.py
├── datasets/
│   ├── base_data/
│   ├── commentary_data/
│   └── playerStats_data/
├── tools/
│   ├── commentary_tool.py
│   ├── player_tool.py
│   └── top_scorer_tool.py
├── download_data.py
├── main.py
├── requirements.txt
└── .env


agents/ – LLM-based agents for stats and predictions.

tools/ – Utilities for retrieving commentary, player stats, and top scorers.

datasets/ – CSV datasets for commentary and player information.

main.py – Streamlit application.

download_data.py – Script to download required datasets.

.env – Stores GROQ API key.

⚙️ How It Works

Load CSV datasets for commentary, player stats, and teams.

Merge player stats with player and team info to create a unified dataset.

Convert each row of commentary and stats into LangChain Documents.

Add documents to a vector store for semantic search.

Initialize agents:

Statistics Agent: Always fetches player and match stats.

Prediction Agent: Runs only for queries containing prediction-related keywords.

Streamlit interface handles user input, displays chat history, and returns AI responses.

🚀 Example Queries

"Who scored the most goals in the 2025 season?"

"Show me Bruno Fernandes stats."

"Who is likely to be the top scorer next season?"

"Provide commentary insights from the latest matches."

📌 Notes

Supports both GROQ and Ollama LLMs; switch easily in main.py.

Agent responses are streamed for better interactivity.

Make sure datasets are downloaded locally and .env is configured with your GROQ API key.

Docker users must mount local datasets and .env for the container to run properly.