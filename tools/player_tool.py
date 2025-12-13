from langchain.tools import tool

def get_player_stat_tool(vector_store,docs):
    @tool
    def retrieve_stats(query: str):
        """
        Retrieve relevant player statistics from the vector database
        based on a natural language query.
        """
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return {"content": serialized} 
    return retrieve_stats