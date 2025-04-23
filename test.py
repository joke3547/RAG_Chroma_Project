import openai
import os
from dotenv import load_dotenv
from openai import OpenAI
# 載入 API KEY
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key 

# 設定 OpenRouter 客戶端
client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

try:
    response = client.chat.completions.create(
        model="meta-llama/llama-4-maverick:free",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    print("✅ API key 正常工作")
except Exception as e:
    print(f"❌ 發生錯誤: {e}")
