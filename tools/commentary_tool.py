from langchain.tools import tool

def get_commentary_tool(vector_store,docs):
    @tool
    def retrieve_commentary(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
    return retrieve_commentary