from langchain.tools import tool

def get_top_scorer(df):
    @tool(return_direct=True)
    def top_scorer():
        """
         Return the definitive top scorer.
        This tool gives the final answer and should terminate the agent.
        """
        top = df.loc[df["totalGoals_value"].idxmax()]
        return (
            f"Top scorer: {top['fullName']} "
            f"({top['name']}) with {int(top['totalGoals_value'])} goals."
        )
    return top_scorer
