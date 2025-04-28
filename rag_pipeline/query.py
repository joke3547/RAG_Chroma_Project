import os
import ssl
import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma  # 使用更新的 Chroma 類別
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # 使用 langchain-openai 模塊
import re  # 引入正則表達式模塊

# 修復 SSL 憑證問題
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

# 載入 API KEY
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def query_knowledge(query_text, persist_directory='db/chroma_store'):
    try:
        # 使用 HuggingFace Embeddings
        embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        retriever = vectordb.as_retriever()

        # 從 prompt 中提取檔案名稱 (如 YOLO_data.pdf)
        target_pdf = None
        match = re.search(r'(\S+\.pdf)', query_text)  # 假設檔案名稱以 .pdf 結尾
        if match:
            target_pdf = match.group(1)

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

        # 取得回答
        answer = result['result'].strip()
        sources = []

        # 如果提取到檔案名稱，過濾來源文件
        if target_pdf:
            if result['source_documents']:
                for doc in result['source_documents']:
                    metadata = doc.metadata
                    pdf_name = os.path.basename(metadata.get('source', '未知來源'))
                    # 只保留來自目標檔案的資料來源
                    if target_pdf in pdf_name:
                        page = metadata.get('page', '?')
                        sources.append({
                            'pdf_name': pdf_name,
                            'page': page
                        })
        else:
            # 如果沒有指定檔案，則返回所有來源
            if result['source_documents']:
                for doc in result['source_documents']:
                    metadata = doc.metadata
                    pdf_name = os.path.basename(metadata.get('source', '未知來源'))
                    page = metadata.get('page', '?')
                    sources.append({
                        'pdf_name': pdf_name,
                        'page': page
                    })

        # 如果沒有找到結果，則清空 sources
        if "沒有找到" in answer or "查無" in answer or "相關資訊" in answer:
            answer = "抱歉，沒有找到與您的問題相關的資訊。"  # 可以根據需求調整無結果時的訊息
            sources = []  # 沒有找到結果時清空來源資料
        else:
            # 如果有找到結果，則保留來源
            pass

        # 返回的資料
        return {
            "answer": answer,
            "sources": sources  # 只有當有有效答案時，才返回來源
        }

    except Exception as e:
        print(f"Error during query processing: {e}")
        return {
            "answer": "發生錯誤，無法處理您的查詢。請稍後再試。",
            "sources": []
        }
