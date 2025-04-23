import os
import ssl
import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma  # 使用更新的 Chroma 類別
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # 使用 langchain-openai 模塊

# 修復 SSL 憑證問題
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

# 載入 API KEY
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def query_knowledge(query_text, persist_directory='db/chroma_store'):
    # 使用 HuggingFace Embeddings
    embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = vectordb.as_retriever()

    # LLM 設定
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model="meta-llama/llama-4-maverick:free"
    )

    # QA 系統使用 "stuff" 模式，能拿到原始文檔
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True  # ⭐ 加這個才會回傳來源
    )

    result = qa.invoke(query_text)

    # 從原始資料抓出 PDF 檔名與頁碼
    sources = []
    for doc in result['source_documents']:
        metadata = doc.metadata
        pdf_name = os.path.basename(metadata.get('source', '未知來源'))
        page = metadata.get('page', '?')
        sources.append({
            'pdf_name': pdf_name,
            'page': page
        })

    return {
        "answer": result['result'],  # 回答內容
        "sources": sources           # 資料來源清單
    }

