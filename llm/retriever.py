"""RAG 的检索部分：把策略文档切成小段，按问题找出最相关的几段。

用 TF-IDF（词频-逆文档频率）衡量问题和每一段的相似度，在本地运行，不需要调用任何 API。
"""
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PLAYBOOK = Path(__file__).resolve().parent.parent / "docs" / "segment_playbook.md"


def load_chunks(path: Path = PLAYBOOK) -> list[dict]:
    """按二级标题（## ）切分文档，每个章节是一段。"""
    chunks, title, lines = [], None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if title:
                chunks.append({"section": title, "text": "\n".join(lines).strip()})
            title, lines = line[3:].strip(), []
        elif title:
            lines.append(line)
    if title:
        chunks.append({"section": title, "text": "\n".join(lines).strip()})
    return chunks


class PlaybookRetriever:
    def __init__(self, path: Path = PLAYBOOK):
        self.chunks = load_chunks(path)
        # 标题也参与匹配，并重复一次以提高它的权重
        docs = [f"{c['section']} {c['section']} {c['text']}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(docs)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = scores.argsort()[::-1][:top_k]
        return [
            {"section": self.chunks[i]["section"], "score": round(float(scores[i]), 3),
             "text": self.chunks[i]["text"]}
            for i in ranked if scores[i] > 0
        ]


if __name__ == "__main__":
    r = PlaybookRetriever()
    print(f"共 {len(r.chunks)} 段")
    for q in ["How should we market to young online customers?",
              "Which customers should get discounts?",
              "What does response_probability mean?"]:
        print(f"\n问题：{q}")
        for hit in r.search(q):
            print(f"  {hit['score']:.3f}  {hit['section']}")
