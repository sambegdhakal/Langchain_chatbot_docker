from langchain.tools import tool
import re

def get_form_tool(vector_store):
    @tool
    def get_team_form(query: str):
        """
        Return the recent form for a team based on the vector store.
        Extracts last matches' W/D/L results and summarizes wins, draws, and losses.
        """
        # search vector store for the team
        docs = vector_store.similarity_search(query, k=1)
        if not docs:
            return "No form data found for this team."

        text = docs[0].page_content

        # Extract form string (W/D/L)
        match = re.search(r"form:\s*([WDL]+)", text)
        if not match:
            return "Form data not found in the document."

        form_str = match.group(1)  # Already last 5 matches

        # Convert to statistics
        wins = form_str.count("W")
        draws = form_str.count("D")
        losses = form_str.count("L")
        total_matches = len(form_str)

        # Return human-readable summary
        return f"Recent form (last {total_matches} matches): {form_str} | Wins: {wins}, Draws: {draws}, Losses: {losses}"

    return get_team_form
