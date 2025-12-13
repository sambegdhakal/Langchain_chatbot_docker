from langchain_core.documents import Document

def test_document_creation():
    doc = Document(page_content="sample test text by sambeg", metadata={"source": "test"})
    assert doc.page_content == "sample test text by sambeg", "content does not match"
    assert doc.metadata["source"] == "test", "metadata does not match"
