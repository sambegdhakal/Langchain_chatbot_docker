Run LangChain Fotbot App Using Docker (Windows)
Step 1: Pull the Docker image
docker pull sambegdhakal/langchain-fotbot-app:latest

Step 2: Dataset folder location

Your datasets must exist locally at:

<PROJECT_ROOT>\datasets


Example:

C:\Users\YourName\OneDrive\Desktop\lang_chain_project_docker\datasets

Step 3: Run the Docker container
docker run -p 8501:8501 `
-e GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE" `
-v "<PROJECT_ROOT>\datasets:/fotbot_app/datasets" `
sambegdhakal/langchain-fotbot-app:latest


✅ Notes

<PROJECT_ROOT> = the folder that contains your project

Left side of -v → local Windows path

Right side of -v → container path (Linux)

Always use /fotbot_app/datasets inside the container

Step 4: Access the app
http://localhost:8501

Step 5: Stop the container
Using terminal
docker ps
docker stop <CONTAINER_ID>




Run LangChain Fotbot App Using Docker (macOS)
Step 1: Pull the Docker image
docker pull sambegdhakal/langchain-fotbot-app:latest

Step 2: Dataset folder location

Your datasets must exist locally at:

<PROJECT_ROOT>/datasets


Example:

/Users/yourname/Desktop/lang_chain_project_docker/datasets

Step 3: Run the Docker container
docker run -p 8501:8501 \
-e GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE" \
-v "<PROJECT_ROOT>/datasets:/fotbot_app/datasets" \
sambegdhakal/langchain-fotbot-app:latest

✅ Notes

<PROJECT_ROOT> = the folder that contains your project

Left side of -v → local macOS path

Right side of -v → container path (Linux)

Always use /fotbot_app/datasets inside the container

Step 4: Access the app
http://localhost:8501

Step 5: Stop the container
Using terminal
docker ps
docker stop <CONTAINER_ID>
