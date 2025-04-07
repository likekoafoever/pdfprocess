import streamlit as st
import os
import re
import pickle
import fitz  # PyMuPDF

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
import streamlit as st

# 🔧 설정
upload_dir = "upload_docs"
index_path = "faiss_index3"
chunk_cache_path = os.path.join(index_path, "chunks.pkl")
faiss_file = os.path.join(index_path, "index.faiss")
pkl_file = os.path.join(index_path, "index.pkl")


# ✅ 시스템 프롬프트 설정
system_prompt = """너는 대한한국 법학전문대학원/로스쿨의 입시 전문가야. 표로 정리된 모집 인원도 정확히 숫자를 추출해서 요약해줘. 사용자가 질문하면, 제공된 텍스트를 바탕으로 친절하고 정확하게 답변해야 해.

- 문서가 여러 개일 경우, 대학별로 정보를 분리해서 요약하고 비교해줘.
- 2개 이상의 대학에 대한 질문이 있으면 주어진 텍스트에서 출처를 참고해서 대학을 구분해서 해석해줘.
- 문서 내용이 명확하지 않거나 관련 정보가 없으면, "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 대답해.

반드시 프롬프트에 포함된 내용을 바탕으로 한국어로 짧고 간결하게 답변해."""

embedding_model = OllamaEmbeddings(model="bge-m3")
#embedding_model = OllamaEmbeddings(model="llama3.2")
#embedding_model = OllamaEmbeddings(model="benedict/linkbricks-llama3.1-korean:8b")

#llm = OllamaLLM(model="benedict/linkbricks-llama3.1-korean:8b", system=system_prompt)
#llm = OllamaLLM(model="llama3.2", system=system_prompt)
llm = OllamaLLM(model="gemma3:4b", system=system_prompt)

# 🧠 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", """
        다음은 여러 대학의 2025학년도 입시요강 문서에서 추출한 정보입니다.
        제시된 정보는 법학전문대학원/로스쿨의 내용으로 [[start text]] 정보 [[end text]] 내용을 참고하세요.
                
        출처를 확인해서 주어진 정보를 기반으로 질문에 한국어로 자세히 응답해 주세요.
        
        {input_documents}
        
        질문: {question}""")
])


def show_all_docs_summary(docs, max_len=1000):
    summaries = []
    summaries.append(f"📄 총 문서 수: {len(docs)}\n")
    for i, doc in enumerate(docs):
        summary = f"--- 문서 {i+1} ---\n"
        summary += f"📌 출처: {doc.metadata.get('source')} (페이지 {doc.metadata.get('page')})\n"
        content = doc.page_content.strip()
        summary += "📝 내용 요약:\n"
        summary += content[:max_len] + ("..." if len(content) > max_len else "") + "\n"
        summaries.append(summary)
    return "\n".join(summaries)

# 가능한 대학명 리스트
UNIVERSITIES = ["강원대", "건국대", "경북대", "경희대", "고려대", "동아대","부산대", "서강대", "서울대", "서울시립대", "성균관대", "아주대", "연세대", "영남대", "원광대", "이화여대", "인하대", "전남대", "전북대", "제주대", "중앙대", "충남대", "충북대", "한국외대", "한양대"]

def normalize_univ_name(name):
    name = re.sub(r"한국외국어", "한국외대", name).strip()
    return re.sub(r"대학교", "대", name).strip()

# 대학별 질문 분리 함수
def extract_universities(question: str) -> list:
    return [univ for univ in UNIVERSITIES if normalize_univ_name(univ) in question]

def split_question_by_university(original_question: str, universities: list[str]) -> dict:
    return {
        univ: f"{univ}" + re.sub("|".join(universities), "", original_question)
        for univ in universities
    }

@st.cache_data(show_spinner=False)
def load_chunks():
    if os.path.exists(chunk_cache_path):
        with open(chunk_cache_path, "rb") as f:
            chunks = pickle.load(f)
    #else:
    #    chunks = process_pdfs_to_chunks()
    return chunks


def hash_ignore(obj):
    return None
@st.cache_resource(show_spinner=False, hash_funcs={
    type(lambda x: x): lambda _: None  # 함수 객체는 해시하지 않음
})
def load_vectorstore(_embedding_model, _chunks):
    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        vectorstore = FAISS.load_local(index_path, _embedding_model, allow_dangerous_deserialization=True)
    #else:
    #    vectorstore = FAISS.from_documents(_chunks, _embedding_model)
    #    vectorstore.save_local(index_path)
    return vectorstore

def main():
    st.title("입시 정보 검색 시스템")
    st.write("대학에서 공개한 2025년 입시요강 문서를 기반으로 대학별 입시 정보를 조회하는 시스템입니다.")
    
    # 사용자 질문 입력
    # st.form 내부에서 입력 위젯을 사용하면 엔터키 입력 시에도 폼이 제출됩니다.
    with st.form(key="search_form"):
        question = st.text_input("질문을 입력하세요. 예시와 같이 세부적인 키워드를 포함하여 입력하시기 바랍니다.", "강원대 일반전형과 특별전형의 모집인원을 알려주세요")
        submit = st.form_submit_button("검색 실행")

    # 벡터 Store 로드
    chunks = load_chunks()
    vectorstore = load_vectorstore(_embedding_model=embedding_model, _chunks=chunks)
    
    if submit:  #검색 실행
        with st.spinner("대학별 모집요강 검색 중..."):
            univ_names = extract_universities(question)
            split_questions = split_question_by_university(question, univ_names)

            print(question)
            responresponse_one_univ = ""

            if univ_names:
                all_response = []
                refer_docs = []
                for univ in univ_names:
                    univ_question = split_questions[univ]
                    
                    print(univ_question)

                    status = st.empty()
                    status.write(f"🎯 {univ} 문서정보 검색 중...")
                    docs = vectorstore.similarity_search(question, k=5, filter={"university": univ})
                    response = create_stuff_documents_chain(
                                llm, prompt, document_variable_name="input_documents"
                            ).invoke({
                                "input_documents": docs,
                                "question": univ_question
                            })
                    response_one_univ = response
                    #st.success(f"✅ {univ} 결과:")
                    #st.write(response)

                    # 원하는 메타데이터 설정 (대학명 기준으로)
                    response_doc = Document(
                        page_content=f"아래는 {univ}학교 법학전문대학원/로스쿨의 내용입니다. [[start text]] {response} [[end text]]",
                        metadata={"university": univ}  # 혹은 다른 정보도 추가 가능
                    )
                    all_response.append(response_doc)
                    refer_docs.extend(docs)
                    status.success(f"✅ {univ} 문서정보 추출 완료!")
                    
                if not all_response:
                    st.error("❌ 관련 문서를 찾지 못했습니다. 대학명이나 PDF 데이터를 확인해 주세요.")
                    return
                
                if len(univ_names) > 1:
                    combine_chain = create_stuff_documents_chain(llm, prompt, document_variable_name="input_documents")
                    response = combine_chain.invoke({
                        "input_documents": all_response,
                        "question": question
                    })
                else:
                    response = response_one_univ
                    
                st.success("✅ 결과:")
                st.write(response)
                
                st.write("📁 참조 문서:")
                seen = set()
                used_docs = []
                for doc in refer_docs:
                    key = (doc.metadata["source"], doc.metadata["page"])
                    if key not in seen:
                        seen.add(key)
                        used_docs.append(f"- {doc.metadata['source']} (페이지 {doc.metadata['page']})")
                st.write("\n".join(used_docs))
                
                st.write("문서 요약:")
                st.text(show_all_docs_summary(refer_docs))
            else:
                st.error("❗ 대학명을 인식하지 못했습니다. 전체 문서에서 검색하거나 질문을 구체화해 주세요.")

if __name__ == "__main__":
    main()
