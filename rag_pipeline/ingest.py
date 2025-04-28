import os
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv

# 載入 .env 的 API KEY
load_dotenv()
persist_directory = 'db/chroma_store'

embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

def get_existing_metadata():
    """
    獲取現有的 PDF 元數據（source + page），避免重複嵌入。
    """
    existing = vectordb.get(include=["metadatas"])
    return {(m["source"], m["page"]) for m in existing["metadatas"]}

def ingest_pdf(pdf_path):
    """
    處理單一 PDF，逐頁解析並避免重複嵌入。
    """
    pdf_name = os.path.basename(pdf_path)
    existing_meta = get_existing_metadata()

    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            new_documents = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                
                metadata = {"source": pdf_name, "page": i + 1}
                key = (metadata["source"], metadata["page"])

                if key in existing_meta:
                    continue  # 跳過已存在的頁面

                new_documents.append(Document(
                    page_content=text,
                    metadata=metadata
                ))

        if not new_documents:
            return f"⚠️ {pdf_name} 所有頁面都已存在，跳過嵌入"
        
        vectordb.add_documents(new_documents)
        return f"✅ {pdf_name} 嵌入 {len(new_documents)} 頁成功"

    except Exception as e:
        return f"❌ 處理 PDF {pdf_name} 時發生錯誤: {str(e)}"

def ingest_all_pdfs(pdf_folder="pdfs"):
    """
    批次處理 pdf_folder 資料夾中所有 PDF。
    """
    results = []
    if not os.path.exists(pdf_folder):
        return [f"❌ 資料夾 {pdf_folder} 不存在！"]
    
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(pdf_folder, filename)
            result = ingest_pdf(full_path)
            results.append(result)
    
    return results
