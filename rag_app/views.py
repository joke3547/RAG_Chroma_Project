from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import PDFUploadForm
from .models import PDFDocument
from rag_pipeline.ingest import ingest_pdf, ingest_all_pdfs
from rag_pipeline.query import query_knowledge
import os

def home(request):
    form = PDFUploadForm()
    answer = ""
    question = "" 
    
    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'upload':
            form = PDFUploadForm(request.POST, request.FILES)
            if form.is_valid():
                # 儲存上傳的多個 PDF 檔案
                files = request.FILES.getlist('pdf_file')
                upload_dir = 'media/pdfs/'
                os.makedirs(upload_dir, exist_ok=True)

                # 儲存檔案並記錄處理結果
                for f in files:
                    file_path = os.path.join(upload_dir, f.name)
                    with open(file_path, 'wb') as f_dest:
                        for chunk in f.chunks():
                            f_dest.write(chunk)

                # 處理並將所有 PDF 檔案嵌入知識庫
                results = ingest_all_pdfs(upload_dir)
                for result in results:
                    messages.success(request, result)

                return redirect('home')
            else:
                messages.error(request, "上傳 PDF 時發生錯誤！")

        elif action == 'ask':
            question = request.POST.get('question')
            if question:
                prompt = f"""
請根據知識庫回答以下問題，並使用 HTML 語法排版，具體格式要求如下：
1. 每個主題使用 <h3> 標題標示
2. 解釋內容請以 <p> 分段
3. 有條列時使用 <ul><li>...</li></ul> 顯示
4. 請避免使用 Markdown 語法（如 **、*、#）
問題如下：
{question}
"""

                result = query_knowledge(prompt)
                answer = result.get("answer", "")
                sources = result.get("sources", [])
            else:
                messages.error(request, "請輸入問題進行查詢！")

    return render(request, 'rag_app/home.html', {
        'form': form,
        'answer': answer,
        'sources': sources if 'sources' in locals() else [],
        'question': question
    })
