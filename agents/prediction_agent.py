from langchain.agents import create_agent

def prediction_agent(tools, llm):
    prompt = (
        "You are a football prediction agent. Your job is to provide predictions "
        "based on the output from the statistics agent and the user's original query. "
        "Instructions:\n"
        "1. Carefully read the statistics provided by the statistics agent.\n"
        "2. Understand the user's question or request.\n"
        "3. Make predictions (e.g., expected goals, likely winner, key players) "
        "based on the statistics and context. Clearly write the prediction."
        "Make prediction based on the available data."
        "4. Prioritize about latest games or upcoming games based on latest date or data." 
        "Mostly when asked about upcoming games or prediction regarding future prioritize latest date or data."
        "5. Also while picking player names, pick latest player names that are playing. Pick this based on date or latest squad."
        "6. Include reasoning behind your prediction, considering factors like:\n"
        "   - if available, opponent's recent form\n"
        "   - Goals conceded and scored\n"
        "   - Any relevant stats from the statistics agent output\n"
        "7. Be concise and precise. Do NOT make up stats; rely only on the provided data.\n"
        "8. Answer in a way that directly addresses the user's question."
    )
    agent = create_agent(llm, tools, system_prompt=prompt)
    return agent
