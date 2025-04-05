# app.py

import streamlit as st
import openai
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import tempfile
import os
import io
import json

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ✅ Streamlit 설정
st.set_page_config(page_title="📚 PDF 기술 키워드 검색기", layout="wide")

# ✅ Google Drive 서비스 초기화
@st.cache_resource
def init_drive_service():
    credentials_info = st.secrets["google_credentials"]
    creds = service_account.Credentials.from_service_account_info(dict(credentials_info))
    service = build("drive", "v3", credentials=creds)
    return service

drive_service = init_drive_service()

# ✅ PDF 다운로드 함수
def download_pdf_from_drive(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return fh

# ✅ 벡터 로딩 함수
@st.cache_resource
def load_vector_data(vector_dir):
    index = faiss.read_index(os.path.join(vector_dir, "faiss_index.index"))
    with open(os.path.join(vector_dir, "pages.pkl"), "rb") as f:
        pages = pickle.load(f)
    model = SentenceTransformer("all-mpnet-base-v2")
    return index, pages, model

# ✅ GPT 키워드 추출 함수
def gpt_keywords(api_key, question):
    client = openai.OpenAI(api_key=api_key)
    prompt = f"""
질문: "{question}"

이 질문과 연관된 기술 키워드를 10~15개 뽑아줘.
- 키워드는 짧은 명사형으로 (예: 온디바이스 AI, 모델 압축, 엣지 컴퓨팅 등)
- 쉼표(,)로 구분해서 한 줄로만 출력해줘.
"""
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )
    return [kw.strip() for kw in res.choices[0].message.content.split(",")]

# ✅ GPT 결과 정리 함수
def gpt_select_5(api_key, question, docs):
    client = openai.OpenAI(api_key=api_key)
    titles = "\n".join([
        f"- {doc['text'].splitlines()[0][:80]} (Page {doc['page']})"
        if doc['text'].strip() else f"- 제목 없음 (Page {doc['page']})"
        for doc in docs
    ])
    prompt = f"""
질문: "{question}"

아래는 기술 문서들의 제목과 페이지 번호입니다.

{titles}

이 중에서 질문과 관련도가 높은 기술을 5개 골라 아래 형식으로 정리해줘:

추천기술1 : Page 번호 
기술설명1 : 한 줄 이내로 간단히 요약

추천기술2 : ...
기술설명2 : ...

형식 꼭 맞춰줘. 결과는 한국어로 정리해줘.
"""
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    return res.choices[0].message.content

# ✅ FAISS 검색 함수
def search_top_k(model, index, pages, keyword, k=4):
    query_vec = model.encode([keyword])
    D, I = index.search(np.array(query_vec), k)
    return [pages[i] for i in I[0]]

# ✅ UI 구성
st.title("📚 Google Drive 기반 기술 검색기")

api_key = st.text_input("🔐 OpenAI API Key", type="password")
question = st.text_input("💬 검색할 질문을 입력하세요")

if api_key and question:
    pdf_file_id = st.secrets["pdf_file_id"]
    vector_dir = st.secrets["vector_dir"]

    with st.spinner("📄 PDF 파일 다운로드 중..."):
        pdf_data = download_pdf_from_drive(drive_service, pdf_file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_data.read())
            tmp_path = tmp.name
        st.download_button("📥 PDF 다운로드", data=open(tmp_path, "rb").read(), file_name="document.pdf")
        st.subheader("📄 PDF 미리보기")
        st.pdf(tmp_path)

    index, pages, model_embed = load_vector_data(vector_dir)

    with st.spinner("🔍 GPT 키워드 추출 및 문서 검색 중..."):
        keywords = gpt_keywords(api_key, question)
        st.success("📌 추출된 키워드:")
        st.write(", ".join(keywords))

        all_docs = []
        seen = set()
        for kw in keywords:
            for doc in search_top_k(model_embed, index, pages, kw):
                key = (doc['page'], doc['text'][:50])
                if key not in seen:
                    all_docs.append(doc)
                    seen.add(key)

        result = gpt_select_5(api_key, question, all_docs)
        st.markdown("## ✅ GPT 추천 기술 5개")
        st.markdown(result)
