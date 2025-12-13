from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

def test_vector():
    # Embeddings; using Ollama
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    doc = [Document(page_content="sample test text by sambeg", metadata={"source": "test"}),
           Document(page_content="sample test text by sambeg part 2", metadata={"source": "test2"})]

    # creating a vector store
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(doc)

    # Checking if document exists via search
    results = vector_store.similarity_search("sample test text", k=2)

    assert len(results) > 1, "vector store not successful"