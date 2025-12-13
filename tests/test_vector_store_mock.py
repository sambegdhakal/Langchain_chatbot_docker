from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

def test_vector_mocked():
    doc = [
        Document(page_content="sample test text by sambeg", metadata={"source": "test"}),
        Document(page_content="sample test text by sambeg part 2", metadata={"source": "test2"})
    ]

    with patch("langchain_ollama.OllamaEmbeddings") as mock_embed:
        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = [[0.1, 0.2, 0.3]] * len(doc)
        mock_embed.return_value = mock_instance

        # create vector store using mocked embeddings
        vector_store = InMemoryVectorStore(mock_instance)
        vector_store.add_documents(doc)

        results = vector_store.similarity_search("sample test text", k=2)

        assert len(results) > 0, "vector store not successful"
