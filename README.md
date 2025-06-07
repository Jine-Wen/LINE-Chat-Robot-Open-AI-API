# 🤖 我的 LINE Bot + OpenAI 聊天機器人專案

最近剛好有點空閒時間，就做了一個結合 LINE BOT 和 OpenAI 的聊天機器人，這是我工作無聊做的技術小應用，分享一下實作過程

## 專案簡介

這個專案是一個 LINE 聊天機器人，能夠使用 OpenAI 的 API 來回應用戶的問題，還能輸出結構化的回覆，讓訊息更清晰易讀，機器人支援私訊直接回覆，在群組則需要標記 @貼心小助理 才會回應

我用了 ngrok 作為中介，讓我可以完全利用 Google Colab 的免費運算資源，這樣我就能實現類似 RAG 的功能 
- 讓 AI 從我提供的知識庫檔案中檢索資訊，然後生成更準確的回答，這樣做不需要自己架設伺服器，節省了很多成本👍

## 🛠️ 專案架構說明

### 1. 環境設置

我使用 Google Colab 來運行這個專案，因為它方便且免費，首先需要掛載 Google Drive 來存取知識庫檔案：

```python
from google.colab import drive
drive.mount('/content/drive')
```

然後安裝必要的套件：

```python
!pip install flask line-bot-sdk pyngrok openai flask-cors pandas
```

### 2. 主要組件設定 🔑

我設定了一些環境變數，包括 LINE 和 OpenAI 的 API 金鑰：

```python
import os
os.environ['LINE_CHANNEL_SECRET'] = '你的LINE_CHANNEL_SECRET'
os.environ['LINE_CHANNEL_TOKEN'] = '你的LINE_CHANNEL_TOKEN'
os.environ['OPENAI_API_KEY'] = '你的OPENAI_API_KEY'
os.environ['NGROK_AUTHTOKEN'] = '你的NGROK_AUTHTOKEN'
```

> 提醒：實際發布時記得移除這些金鑰，不要編碼在程式中！

### 3. 知識庫整合 

我從 Google Drive 讀取一個知識庫檔案，這就是我實現 RAG 功能的關鍵部分：

```python
kb_path = '/content/drive/MyDrive/Flask/knowledge_base.txt'
if os.path.exists(kb_path):
    with open(kb_path, 'r', encoding='utf-8') as f:
        knowledge_base = f.read()
else:
    knowledge_base = '知識庫檔案未找到，使用預設內容'
```

這樣機器人就能根據我放在 Google Drive 的知識文件來回答用戶的問題，實現情境化的答覆，

### 4. 結構化回應設計 ✨

我使用 OpenAI 的 function calling 功能，讓回應更加結構化：

```python
def get_structured_response(user_input, previous=None):
    # 設定系統提示和用戶輸入
    messages = [
        {"role": "system", "content": "Knowledge Base:\n" + knowledge_base},
        {"role": "system", "content": "請根據指定的 JSON schema 回傳結構化回應，且僅輸出 JSON，不要文字"}
    ]
    if previous:
        messages.append({"role": "assistant", "content": previous})
    messages.append({"role": "user", "content": user_input})
    
    # 呼叫 OpenAI API
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",  # 使用的模型
        messages=messages,
        functions=functions,
        function_call={"name": "generate_structured"},
        max_tokens=250,
        temperature=0.7
    )
    
    args_str = resp.choices[0].message.function_call.arguments
    return json.loads(args_str)
```

### 5. LINE Bot 實現 

處理用戶的訊息，包括私訊和群組中的互動：

```python
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text
    source_type = event.source.type
    mention_tag = f"@{BOT_NAME}"
    
    # 判斷是否需要回覆
    if source_type == 'user':
        should_reply = True
        clean_text = user_text.strip()
    elif source_type in ['group', 'room'] and mention_tag.lower() in user_text.lower():
        should_reply = True
        clean_text = user_text.replace(mention_tag, "", 1).strip()
        # 還有處理其他情況的邏輯...
    
    # 如果需要回覆，則生成結構化回應
    if should_reply and clean_text:
        structured = get_structured_response(clean_text)
        # 組成漂亮的回覆格式並標準化~

```

### 6. 啟動 Web 服務與 ngrok 整合 🚀

最後，我使用 ngrok 創建一個公開的 URL，並啟動 Flask 服務：

```python
if __name__ == '__main__':
    ngrok.set_auth_token(NGROK_TOKEN)
    public_url = ngrok.connect(5000).public_url
    print(f'請將此 URL 填入 LINE Developers → Webhook URL：{public_url}/callback')
    app.run(host='0.0.0.0', port=5000)
```

這段代碼是最關鍵的一環 - ngrok 幫我把 Colab 本地的 Flask 服務暴露到公網，這樣 LINE Platform 就能把用戶訊息轉發到我的 Colab 筆記本上，我完全不用租用伺服器，只要 Colab 筆記本保持運行就行了！

## 🌟 實用功能

1. **免費運算資源**：利用 Google Colab 的免費 GPU/CPU 來處理請求
2. **RAG 知識增強**：透過知識庫檔案讓 AI 能夠回答特定領域問題 
3. **結構化回答**：回覆會自動格式化成標題、步驟清單和備註
4. **支援群組使用**：在群組中只要 @機器人 就可以使用

這運用了如何串接 LINE Bot API、OpenAI 的 function calling 功能，以及如何利用 ngrok 和 Colab 來搭建免費的服務
最讚的是實現了類似 RAG 的功能，讓 AI 能夠更加情境化地回答問題，而且完全不用花錢租伺服器！ 💪
如果你有什麼想法或建議，歡迎分享！ 
