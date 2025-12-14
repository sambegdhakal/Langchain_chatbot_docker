from langchain.tools import tool

def get_top_scorer(df):
    @tool(return_direct=True)
    def top_scorer(query: str = None):
        """
        Returns the top goal scorer of the season.
        Ignores the query input and computes the result from the dataset.
        """
        top = df.loc[df["totalGoals_value"].idxmax()]
        return f"Top scorer: {top['fullName']} ({top['name']}) with {int(top['totalGoals_value'])} goals."

    return top_scorer
