from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

INDEX_DIR = "faiss_index_3"
embedding_model = SentenceTransformerEmbeddings(model_name="jhgan/ko-sbert-sts")
db = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)

results = db.similarity_search_with_score(
    "2025학년도 서울대 법학전문대학원 모집정원은?",
    k=5
)

priority_keywords = ["모집", "정원", "인원"]

def rerank_with_keywords(results):
    scored = []
    for doc, score in results:
        keyword_score = sum(1 for kw in priority_keywords if kw in doc.page_content)
        # 점수 + 키워드 가중치 부여
        scored.append((doc, score - keyword_score * 5))  # 키워드 1개당 5점 보정
    return sorted(scored, key=lambda x: x[1])  # 낮은 점수가 더 유사

# 사용
reranked_results = rerank_with_keywords(results)

for i, (doc, score) in enumerate(results, 1):
    print(f"\n🔍 결과 {i} (유사도: {score:.4f}) - {doc.metadata['source']}")
    print(doc.page_content[:500])
