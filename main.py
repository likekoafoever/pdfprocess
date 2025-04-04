import os 
import re
import pdfplumber
import fitz  # PyMuPDF
import pickle   # chunks 캐시 저장
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings  # ✅ 최신 위치
from langchain_ollama import OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 🔧 설정
upload_dir = "upload_docs"
index_path = "faiss_index"
chunk_cache_path = os.path.join(index_path, "chunks.pkl")
faiss_file = os.path.join(index_path, "index.faiss")
pkl_file = os.path.join(index_path, "index.pkl")

# ✅ 예시 질문
question = "고려대와 연세대의 모집인원을 비교해줘"

# ✅ 시스템 프롬프트 설정
system_prompt = """너는 대한한국 법학전문대학원/로스쿨의 입시 전문가야. 표로 정리된 모집 인원도 정확히 숫자를 추출해서 요약해줘. 사용자가 질문하면, 제공된 텍스트를 바탕으로 친절하고 정확하게 답변해야 해.

- 질문에 대해 확실한 정보가 문서에 있을 경우, 요점을 간결하게 정리해서 설명해줘.
- 출처 문서의 내용 외에는 추측하지 마.
- 사용자가 특정 대학(예: 연세대, 강원대)을 언급하면 해당 대학에만 관련된 정보를 사용해.
- 모집 단위, 전형 유형, 인원 수, 일정, 지원 자격 등에 관한 질문이 자주 나올 수 있어.
- 문서가 여러 개일 경우, 대학별로 정보를 분리해서 요약하고 비교해줘.
- 문서 내용이 명확하지 않거나 관련 정보가 없으면, "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 대답해.

반드시 정확하고 신뢰할 수 있는 정보만 간략하게 답변해."""


embedding_model = OllamaEmbeddings(model="bge-m3")
#embedding_model = OllamaEmbeddings(model="gemma3:4b")
#embedding_model = OllamaEmbeddings(model="benedict/linkbricks-llama3.1-korean:8b")

#llm = OllamaLLM(model="benedict/linkbricks-llama3.1-korean:8b", system=system_prompt)
llm = OllamaLLM(model="gemma3:4b", system=system_prompt, temperature=0)


# 🧠 LLM에 전달할 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", """
        다음은 여러 대학의 입시 문서에서 추출한 정보입니다.
        각 문서는 [출처: 파일명, 페이지 n] 으로 시작합니다.
        
        출처를 확인해서 대학별로 내용을 구분해서 읽고, 다음 질문에 답해주세요.
        
        {input_documents}
        
        질문: {question}""")
])

def reduce_spaces(text: str) -> str:
    return re.sub(r' +', '', text)

def join_known_spacing_errors(text: str) -> str:
    replacements = {
        "모 집 인 원": "모집인원",
        "지 원 자 격": "지원자격",
        "입 시 요 강": "입시요강",
        "일 반 전 형": "일반전형",
        "특 별 전 형": "특별전형",
        "선 발 인 원": "선발인원",
        "자 격 요 건": "자격요건",
        "모 집 군": "모집군",
        "법 학 전 문 대 학 원": "법학전문대학원",
        "전 형": "전형",
        "유 형": "유형",
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    return text

#✅ all_docs 전체 내용 요약 출력하는 코드
def show_all_docs_summary(docs, max_len=1000):
    print(f"📄 총 문서 수: {len(docs)}\n")
    for i, doc in enumerate(docs):
        print(f"--- 문서 {i+1} ---")
        print(f"📌 출처: {doc.metadata.get('source')} (페이지 {doc.metadata.get('page')})")
        content = doc.page_content.strip()
        print("📝 내용 요약:")
        print(content[:max_len] + ("..." if len(content) > max_len else ""))
        print()
        
# 모든 가능한 대학명 리스트 (필요시 확장 가능)
UNIVERSITIES = ["강원대", "건국대", "경북대", "경희대", "고려대", "동아대","부산대", "서강대", "서울대", "서울시립대", "성균관대", "아주대", "연세대", "영남대", "원광대", "이화여대", "인하대", "전남대", "전북대", "제주대", "중앙대", "충남대", "충북대", "한국외대", "한양대"]

def normalize_univ_name(name):
    name = re.sub(r"한국외국어", "한국외대", name).strip()
    return re.sub(r"대학교", "대", name).strip()

def extract_universities(question: str) -> list[str]:
    return [univ for univ in UNIVERSITIES if normalize_univ_name(univ) in question]

# ✅ 문서 → 청크 변환
def process_pdfs_to_chunks():
    all_docs = []

    for file_name in os.listdir(upload_dir):
        if not file_name.endswith(".pdf"):
            continue

        file_path = os.path.join(upload_dir, file_name)
        print(f"📄 처리 중: {file_name}")

        # with pdfplumber.open(file_path) as pdf:
        pdf = fitz.open(file_path)
        for page_number in range(len(pdf)):
            page = pdf.load_page(page_number)
            text = page.get_text()
            tables = page.find_tables()
            table_summaries = []

            for table in tables:
                summary = table.extract()
                table_summaries.append(f"[TABLE-START]\n{summary}\n[TABLE-END]")

            combined_text = text + "\n" + "\n".join(table_summaries)
            
            doc = Document(
                page_content=combined_text,
                metadata={
                    "university": file_name.replace("2025 ", "").replace(".pdf", ""),  # 대학명
                    "source": file_name,
                    "page": page_number + 1
                }
            )

            all_docs.append(doc)

    print("PDF 페이지별 청크 생성 완료!")
    #splitter = SemanticChunker(embedding_model)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    print("✅ 캐시 저장")
    with open(chunk_cache_path, "wb") as f:
        pickle.dump(chunks, f)

    return chunks


# 💾 FAISS 벡터 저장
if os.path.exists(chunk_cache_path):
    print("📂 청크 캐시 로드 중...")
    with open(chunk_cache_path, "rb") as f:
        chunks = pickle.load(f)
else:
    print("📂 청크 생성 중...")
    chunks = process_pdfs_to_chunks()

# ✅ FAISS 인덱스 생성/불러오기
if os.path.exists(faiss_file) and os.path.exists(pkl_file):
    print("📂 FAISS 인덱스 로드 중...")
    vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
else:
    print("💾 FAISS 인덱스 생성 중...")
    if chunks:
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local(index_path)
    else:
        raise ValueError("❌ 유효한 문서 청크가 없습니다. PDF를 확인하세요.")

# 필터링 설정
univ_names = extract_universities(question)
if univ_names:
    all_docs = []

    for univ in univ_names:
        print(f"\n🎯 {univ} 정보 검색 중...")
        #source_pdf = f"2025 {univ}.pdf"
        # 리트리버 생성
        docs = vectorstore.similarity_search(question, k=5, filter={"university": univ})
        #retriever = vectorstore.as_retriever(
        #    search_kwargs={"k": 5, "filter": {"university": univ}}
        #)
        ## docs = retriever.get_relevant_documents(question)  # 최신 문법 사용
        #docs = retriever.invoke(question)
        all_docs.extend(docs)  # ✅ 모든 대학 관련 문서 하나로 모음
        
        # Ollama 활용 Prompt 구성
    # ✅ combine_documents_chain 생성 (RetrievalQA 대신!)
    combine_chain = create_stuff_documents_chain(llm, prompt, document_variable_name="input_documents")
    
    # ✅ 문서가 없으면 오류 방지
    if not all_docs:
        raise ValueError("❌ 관련 문서를 찾지 못했습니다. 대학명이나 PDF 데이터를 확인해 주세요.")
    if not question.strip():
        raise ValueError("❌ 질문이 비어있습니다. 질문을 입력해주세요.")
    
    # ✅ 문서 병합 + 질문 전달
    response = combine_chain.invoke({
        "input_documents": all_docs,
        "question": question
    })
    
    print("\n✅ 비교 결과:")
    print(response)

    # 📄 문서 출처 출력 (옵션)
    print("\n📁 사용된 문서:")
    seen = set()
    for doc in all_docs:
        key = (doc.metadata["source"], doc.metadata["page"])
        if key not in seen:
            seen.add(key)
            print(f"- {doc.metadata['source']} (페이지 {doc.metadata['page']})")
            
    # show_all_docs_summary(all_docs)
else:
    print("❗ 대학명을 인식하지 못했습니다. 전체 문서에서 검색하거나 질문을 구체화해 주세요.")



