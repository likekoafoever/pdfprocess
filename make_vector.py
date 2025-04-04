import os
import re
import pandas as pd
import fitz  # PyMuPDF
import pdfplumber
import pickle   # chunks 캐시 저장
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings   # Ollama Embeddings
from langchain_community.vectorstores import FAISS

# 🔧 설정
upload_dir = "upload_docs"
index_path = "faiss_index3"
chunk_cache_path = os.path.join(index_path, "chunks.pkl")
faiss_file = os.path.join(index_path, "index.faiss")
pkl_file = os.path.join(index_path, "index.pkl")

# embedding_model = OllamaEmbeddings(model="bge-m3")
embedding_model = OllamaEmbeddings(model="llama3.2")

# 대학별 입시요강 chunk 대상 페이지 정의
chunk_pages = {
    "2025 강원대.pdf": (3, 18),
    #"2025 강원대.txt": (1, None),
    "2025 건국대.pdf": (3, None),
    "2025 경북대.pdf": (2, 8),
    "2025 경희대.pdf": (11, 20),
    "2025 고려대.pdf": (3, 17),
    "2025 동아대.pdf": (3, 16),
    "2025 부산대.pdf": (3, 17),
    "2025 서강대.pdf": (3, None),
    "2025 서울대.pdf": (5, 12),
    "2025 서울시립대.pdf": (4, 17),
    "2025 성균관대.pdf": (5, 16),
    "2025 아주대.pdf": (2, None),
    "2025 연세대.pdf": (3, 14),
    "2025 영남대.pdf": (5, 15),
    "2025 원광대.pdf": (2, 15),
    "2025 이화여대.pdf": (2, 10),
    "2025 인하대.pdf": (2, 14),
    "2025 전남대.pdf": (3, 15),
    "2025 전북대.pdf": (3, 12),
    "2025 제주대.pdf": (3, 16),
    "2025 중앙대.pdf": (2, 14),
    "2025 충남대.pdf": (3, 19),
    "2025 충북대.pdf": (3, 22),
    "2025 한국외대.pdf": (3, None),
    "2025 한양대.pdf": (3, 22)
}

def reduce_spaces(text: str) -> str:
    return re.sub(r' {2,}', '', text)

def reduce_spaces_all(text: str) -> str:
    return re.sub(r' +', '', text)

def process_pdfs_to_chunks():
    all_docs = []

    for file_name in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, file_name)
        ext = os.path.splitext(file_name)[1].lower()
        print(f"📄 처리 중: {file_name}")

        try:
            if ext == ".pdf":
                all_docs.extend(load_pdf_file(file_path, file_name))
            elif ext == ".txt":
                doc = load_txt_file(file_path, file_name)
                if doc:
                    all_docs.append(doc)
            else:
                print(f"❗ 지원하지 않는 파일 형식: {file_name}")
        except Exception as e:
            print(f"❌ {file_name} 처리 중 오류 발생: {e}")

    print("🧩 페이지별 청크 생성 중...")
    splitter = SemanticChunker(embedding_model)
    #splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    print("💾 캐시 저장 중...")
    with open(chunk_cache_path, "wb") as f:
        pickle.dump(chunks, f)
    print("✅ 완료! 총 청크 수:", len(chunks))
    return chunks

def load_pdf_file(file_path, file_name):
    docs = []
    univ_name = clean_university_name(file_name)

    with fitz.open(file_path) as pdf_fitz, pdfplumber.open(file_path) as pdf_plumber:
        total_pages = len(pdf_fitz)
        start_page, end_page = chunk_pages.get(file_name, (1, None))
        start_idx = max(start_page - 1, 0)
        end_idx = min(end_page if end_page else total_pages, total_pages)

        for page_number in range(start_idx, end_idx):
            text = extract_text_from_pdf(pdf_fitz, page_number)
            tables = extract_tables_from_pdf(pdf_plumber, page_number)
            combined_text = text + "\n" + "\n".join(tables)

            doc = Document(
                page_content=combined_text,
                metadata={"university": univ_name, "source": file_name, "page": page_number + 1}
            )
            docs.append(doc)

    return docs


def load_txt_file(file_path, file_name):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return None

    combined_text = reduce_spaces(text)
    return Document(
        page_content=combined_text,
        metadata={"university": clean_university_name(file_name), "source": file_name, "page": 1}
    )


def extract_text_from_pdf(pdf_fitz, page_number):
    try:
        text = pdf_fitz.load_page(page_number).get_text()
        return reduce_spaces(text)
    except Exception as e:
        print(f"⚠️ 텍스트 추출 오류 (Page {page_number+1}): {e}")
        return ""


def extract_tables_from_pdf(pdf_plumber, page_number):
    summaries = []
    try:
        tables = pdf_plumber.pages[page_number].extract_tables()
        for table in tables:
            if not table or len(table) < 2:
                continue
            df = pd.DataFrame(table[1:], columns=table[0])
            md_table = df.to_markdown()
            md_table = reduce_spaces_all(md_table)
            summaries.append(f"[TABLE-START]\n{md_table}\n[TABLE-END]")
    except Exception as e:
        print(f"⚠️ 표 추출 오류 (Page {page_number+1}): {e}")
    return summaries


def clean_university_name(file_name):
    return file_name.replace("2025 ", "").rsplit(".", 1)[0]

# PDFs -> Chunks
chunks = process_pdfs_to_chunks()


print("💾 벡터스토어 생성 중...")
# 벡터스토어 만들기
print("💾 벡터스토어 생성 중...")
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local(index_path)
print("✅ 완료!")
