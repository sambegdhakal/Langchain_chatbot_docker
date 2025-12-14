from langchain.tools import tool

def get_standings_tool(vector_store):
    @tool
    def retrieve_standings(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
    return retrieve_standings