"""RAG 检索测试：常见问题应该检索到正确的文档章节。"""
from llm.retriever import PlaybookRetriever

retriever = PlaybookRetriever()


def test_playbook_is_split_into_sections():
    assert len(retriever.chunks) == 9


def test_segment_question_finds_segment_section():
    top = retriever.search("How should we market to young online customers?")[0]
    assert "Young Online Browsers" in top["section"]


def test_model_question_finds_model_section():
    top = retriever.search("What does response_probability mean?")[0]
    assert "response model" in top["section"]


def test_limitations_question_finds_limitations_section():
    top = retriever.search("data limitations recency measured before or after campaign")[0]
    assert "limitations" in top["section"].lower()
