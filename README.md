입시 정보 검색 시스템 (2025학년도 로스쿨/법학전문대학원 입시요강 분석)
======================================================================

📌 프로젝트 소개
---------------
이 시스템은 국내 법학전문대학원(로스쿨)의 2025학년도 입시요강 PDF 문서를 기반으로,
대학별 모집정원 및 전형 정보를 질문-응답 형태로 검색할 수 있도록 구성되었습니다.

사용자는 Streamlit 기반 웹 인터페이스를 통해 자연어로 질문을 입력하고,
백엔드에서는 LangChain과 FAISS 벡터 검색을 통해 관련 정보를 추출하여 응답합니다.


🔧 주요 기능
-----------
- PDF 문서 파싱 및 청크 분할
- FAISS 기반 벡터 검색
- 대학명 자동 인식 및 질문 분리
- 대학별 응답과 요약 비교
- Ollama LLM (gemma3:4b 등) 연동
- Streamlit 기반 웹 UI 제공


📂 디렉토리 구조
----------------
- `main_ui.py`: Streamlit 앱 실행 파일
- `vector_query.py`: 벡터 검색 및 rerank 예시
- `upload_docs/`: PDF 업로드 디렉토리
- `faiss_index2/`: FAISS 인덱스 및 캐시 파일 저장 위치
- `requirements.txt`: 필수 패키지 목록


⚙️ 실행 방법
-------------
1. Python 3.10 이상 설치
2. 의존성 설치:
   ```bash
   pip install -r requirements.txt


PDF 문서를 upload_docs/ 폴더에 추가

FAISS 인덱스가 없다면 벡터화 전처리 코드 추가 필요 (process_pdfs_to_chunks() 등)

앱 실행:

streamlit run main_ui.py

웹 브라우저에서 실행된 앱을 사용하여 질문 입력 (예: "서울대의 일반전형과 특별전형 모집인원 알려줘")

📌 사용 모델
LLM: gemma3:4b 또는 benedict/linkbricks-llama3.1-korean:8b (Ollama)

Embedding: bge-m3, jhgan/ko-sbert-sts 등

Ollama 설치 후 실행 중이어야 함:

ollama run gemma3:4b

