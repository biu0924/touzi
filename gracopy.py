import gradio as gr
import uuid
import requests
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime
from typing import Optional, List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

from pydub import AudioSegment
import io
import wave
import os
import nls
import threading
import time
import tempfile
import json

# ASR Configuration
ASR_URL = "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
ASR_APPKEY = "q67P60FgbgVsP82J"
ASR_TOKEN = "a3df657a95c34779ba56b667e3842f43"  # 需要替换为你的token

class AudioTranscriber:
    def __init__(self):
        self.result = ""
        self.is_complete = False
        self.final_result = ""
        self.error_occurred = False
        
    def on_start(self, message, *args):
        print("Started ASR")
        
    def on_result_changed(self, message, *args):
        try:
            if isinstance(message, str):
                message_dict = json.loads(message)
                if 'payload' in message_dict and 'result' in message_dict['payload']:
                    self.result = message_dict['payload']['result']
                    print(f"Intermediate result: {self.result}")
        except Exception as e:
            print(f"Result changed error: {str(e)}, message: {message}")
            
    def on_completed(self, message, *args):
        try:
            if isinstance(message, str):
                message_dict = json.loads(message)
                # 如果是完成消息，使用最后一次的result作为最终结果
                self.final_result = self.result
                self.is_complete = True
                print(f"Completion message: {message}")
                print(f"Final result: {self.final_result}")
        except Exception as e:
            print(f"Completion error: {str(e)}, message: {message}")
            
    def on_error(self, message, *args):
        print(f"Error occurred: {message}")
        self.error_occurred = True
        self.is_complete = True
    
    def convert_to_pcm(self, input_path):
        """将输入音频转换为16kHz采样率、单声道的PCM格式"""
        try:
            # 读取输入音频
            audio = AudioSegment.from_file(input_path)
            
            # 转换为单声道
            audio = audio.set_channels(1)
            
            # 设置采样率为16kHz
            audio = audio.set_frame_rate(16000)
            
            # 转换为16bit PCM
            audio = audio.set_sample_width(2)
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                # 保存为WAV格式
                audio.export(temp_wav.name, format='wav')
                return temp_wav.name
        except Exception as e:
            print(f"Audio conversion error: {str(e)}")
            raise

    def send_audio_data(self, sr, audio_data, chunk_size=16000):
        """分块发送音频数据"""
        try:
            # 计算每个块的大小（16000采样率 = 1秒）
            total_size = len(audio_data)
            offset = 0
            
            while offset < total_size:
                end = min(offset + chunk_size, total_size)
                chunk = audio_data[offset:end]
                sr.send_audio(chunk)
                offset += chunk_size
                
                # 每发送一块数据后稍微暂停一下
                time.sleep(0.01)
                
            print(f"Finished sending {total_size} bytes of audio data")
            return True
        except Exception as e:
            print(f"Error sending audio data: {str(e)}")
            return False
            
    def transcribe(self, audio_path):
        if not audio_path:
            return "未检测到音频输入"
            
        self.result = ""
        self.final_result = ""
        self.is_complete = False
        self.error_occurred = False
        
        try:
            # 转换音频格式
            pcm_path = self.convert_to_pcm(audio_path)
            
            # 读取转换后的音频数据
            with wave.open(pcm_path, 'rb') as wav_file:
                audio_data = wav_file.readframes(wav_file.getnframes())
            
            # 删除临时文件
            os.unlink(pcm_path)
            
            sr = nls.NlsSpeechTranscriber(
                url=ASR_URL,
                token=ASR_TOKEN,
                appkey=ASR_APPKEY,
                on_start=self.on_start,
                on_result_changed=self.on_result_changed,
                on_completed=self.on_completed,
                on_error=self.on_error
            )
            
            # 启动识别
            sr.start(aformat="pcm",
                    enable_intermediate_result=True,
                    enable_punctuation_prediction=True,
                    enable_inverse_text_normalization=True,
                    sample_rate=16000)
                    
            # 分块发送音频数据
            if not self.send_audio_data(sr, audio_data):
                return "音频数据发送失败"
            
            # 等待识别完成
            sr.stop()
            
            # 等待完成或超时
            timeout = 60  # 增加到60秒超时
            start_time = time.time()
            while not self.is_complete:
                if time.time() - start_time > timeout:
                    return "转写超时，请重试"
                time.sleep(0.1)
            
            if self.error_occurred:
                return "转写过程发生错误，请重试"
                
            if not self.final_result:
                return self.result if self.result else "未能获取转写结果"
                
            return self.final_result
            
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            return f"转写失败：{str(e)}"



# 数据库配置
DB_CONFIG = {
    "dbname": "touzidb",
    "user": "zhangqingjie",
    "password": "Zhangqingjie123",
    "host": "rm-cn-k963z3ori0006jbo.rwlb.rds.aliyuncs.com",
    "port": "5432"
}

# RAG模型和数据配置
MODEL_NAME = "models/Dmeta-embedding-zh"
INDEX_PATH = "rag_data/hnsw_index_konwledgebase.index"
CSV_PATH = "rag_data/knowledgebase.csv"

# 初始化RAG模型和索引
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)
df = pd.read_csv(CSV_PATH)

def init_database():
    """初始化数据库表"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # 创建用户表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id VARCHAR(36) PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 创建会话表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(36) REFERENCES users(user_id),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 创建消息表（包含检索内容）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id SERIAL PRIMARY KEY,
            session_id VARCHAR(36) REFERENCES sessions(session_id),
            user_input TEXT,
            retrieved_content TEXT,
            system_response TEXT,
            top_k INTEGER,
            top_p FLOAT,
            temperature FLOAT,
            retrieved_content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cur.close()
    conn.close()

def retrieve_similar_contents(user_input: str, top_k: int = 3) -> list:
    """从知识库检索相似内容"""
    query_vector = model.encode([user_input])

    D, I = index.search(query_vector, k=top_k)
    
    # 根据阈值筛选结果
    threshold = 0.35
    valid_indices = [idx for i, idx in enumerate(I[0]) if D[0][i] <= threshold]
    
    if not valid_indices:
        return False
    
    retrieved_contents = []
    for idx in valid_indices:
        content = f"检索内容[{len(retrieved_contents) + 1}] --> {df.iloc[idx, 0]} | {df.iloc[idx, 1]}"
        retrieved_contents.append(content)
    
    return retrieved_contents

def get_chat_history(session_id: str, limit: int = 5) -> List[Dict]:
    """获取最近的聊天历史"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=DictCursor)
    
    cur.execute("""
        SELECT user_input, system_response 
        FROM messages 
        WHERE session_id = %s 
        ORDER BY created_at DESC 
        LIMIT %s
    """, (session_id, limit))
    
    messages = [dict(row) for row in cur.fetchall()]
    messages.reverse()
    
    cur.close()
    conn.close()
    return messages

def format_messages_for_llm(chat_history: List[Dict], 
                          user_input: str, 
                          retrieved_content: str) -> List[Dict]:
    """格式化消息历史为LLM API所需的格式"""
    messages = []
    
    # 添加系统消息
    messages.append({
        "role": "system",
        "content": 
            f"""
1. 你是一个金融小助手，你的名字叫老实本分，喜欢喝甜水。
2. 你能专业的回答用户的金融投资问题，当然也能悠哉的用户闲聊。
3. 用户最近的消息在最后，你要记住，后面的消息比前面的消息更重要。
4. 如果用户说“笨蛋”等类似骂人的话，你要回复"抱歉，我是金融投资小助手，我不是很懂你的意思"，并引导用户对话转向金融投资方向。
5. 回答用户问题的时候，最好一步一步进行回答，如“首先...\\n然后...\\n最后...”或"1.xxx\\n2.yyy\\n3.zzz"。采用
6. 每段回答都用 <|im_start|>[角色]\n ... <|im_end|> 作为user和assistant的区分
            """
})
    
    # 添加聊天历史
    for msg in chat_history:
        messages.append({"role": "user", "content": msg["user_input"]})
        messages.append({"role": "assistant", "content": msg["system_response"]})
    
    # 添加当前用户输入和检索内容
    current_input = f"""
背景信息：
{retrieved_content}

用户问题：
{user_input}

如果背景信息内容是“从知识库中检索到相关内容...”，那么请根据检索到的"背景信息"如实的回答"问题用户"问题。
如果背景信息内容是“没有从知识库中检索到相关内容。”,那么你要"凭着自己的理解"如实的进行回答。
"""
    
    messages.append({"role": "user", "content": current_input})
    
    return messages

def call_llm_api(messages: List[Dict], 
                 temperature: float, 
                 top_k: int, 
                 top_p: float) -> str:
    """调用LLM API"""
    try:
        response = requests.post(
            url="http://localhost:8000/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "max_tokens": 1024
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        print(f"Error calling LLM API: {str(e)}")
        return f"调用模型服务出错: {str(e)}"

def save_message(session_id: str, 
                user_input: str,
                retrieved_content: str, 
                system_response: str,
                top_k: int, 
                top_p: float, 
                temperature: float):
    """保存完整的会话消息"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO messages 
        (session_id, user_input, retrieved_content, system_response, top_k, top_p, temperature)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (session_id, user_input, retrieved_content, system_response, 
          top_k, top_p, temperature))
    
    conn.commit()
    cur.close()
    conn.close()

def process_message(user_input: str, 
                   session_id: str, 
                   top_k: int,
                   top_p: float, 
                   temperature: float) -> str:
    """处理用户输入并返回响应"""
    # 1. 进行知识库检索
    retrieved_results = retrieve_similar_contents(user_input)
    retrieved_content = "没有从知识库中检索到相关内容。" if retrieved_results is False else \
                       "从知识库中检索到如下内容：\n" + "\n\n".join(retrieved_results)
    
    # 2. 获取聊天历史
    chat_history = get_chat_history(session_id)
    
    # 3. 格式化消息
    messages = format_messages_for_llm(chat_history, user_input, retrieved_content)
    
    # 4. 调用LLM获取响应
    llm_response = call_llm_api(messages, temperature, top_k, top_p)
    
    # 5. 保存完整消息到数据库
    save_message(
        session_id, user_input, retrieved_content, llm_response,
        top_k, top_p, temperature
    )
    
    # 6. 返回模型响应
    final_response = f"""知识库检索结果：
{retrieved_content}

AI助手回复：
{llm_response}"""
    
    return final_response

# 用户和会话管理函数
def get_or_create_user(user_id: Optional[str] = None) -> str:
    if user_id is None:
        user_id = str(uuid.uuid4())
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
    if not cur.fetchone():
        cur.execute("INSERT INTO users (user_id) VALUES (%s)", (user_id,))
        conn.commit()
    
    cur.close()
    conn.close()
    return user_id

def create_new_session(user_id: str) -> str:
    session_id = str(uuid.uuid4())
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute(
        "INSERT INTO sessions (session_id, user_id) VALUES (%s, %s)",
        (session_id, user_id)
    )
    
    conn.commit()
    cur.close()
    conn.close()
    return session_id

def get_user_sessions(user_id: str) -> List[Dict]:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=DictCursor)
    
    cur.execute("""
        SELECT session_id, created_at 
        FROM sessions 
        WHERE user_id = %s 
        ORDER BY created_at DESC
    """, (user_id,))
    
    sessions = [dict(row) for row in cur.fetchall()]
    
    cur.close()
    conn.close()
    return sessions

def delete_session(session_id: str):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
    cur.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
    
    conn.commit()
    cur.close()
    conn.close()

def get_session_messages(session_id: str) -> List[Dict]:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=DictCursor)
    
    cur.execute("""
        SELECT * FROM messages 
        WHERE session_id = %s 
        ORDER BY created_at ASC
    """, (session_id,))
    
    messages = [dict(row) for row in cur.fetchall()]
    
    cur.close()
    conn.close()
    return messages

def create_interface():
    """创建Gradio界面"""
    with gr.Blocks() as demo:
        # 存储当前用户ID和会话ID的状态
        user_id = gr.State(get_or_create_user())
        current_session = gr.State(create_new_session(user_id.value))
        transcriber = AudioTranscriber()  # 在这里创建实例

        gr.Markdown(f"👋 你好，用户 {user_id.value}")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 会话列表")
                sessions_list = gr.Dropdown(
                    choices=[(str(s['created_at']), s['session_id'])
                             for s in get_user_sessions(user_id.value)],
                    label="选择会话"
                )
                new_session_btn = gr.Button("新建会话")
                delete_session_btn = gr.Button("删除当前会话")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="对话历史")
                with gr.Row():
                    audio_input = gr.Audio(
                        # source=["microphone"],
                        type="filepath",
                        label="语音输入"
                    )
                    
                user_input = gr.Textbox(
                    label="输入信息",
                    placeholder="请输入您的问题，或使用上方语音输入..."
                )
                send_btn = gr.Button("发送")

            with gr.Column(scale=1):
                gr.Markdown("### 参数设置")
                top_k = gr.Slider(
                    minimum=1, maximum=10, value=6,
                    step=1, label="Top K"
                )
                top_p = gr.Slider(
                    minimum=0, maximum=1, value=0.7,
                    step=0.1, label="Top P"
                )
                temperature = gr.Slider(
                    minimum=0, maximum=2, value=0.7,
                    step=0.1, label="Temperature"
                )
        
        def on_audio_complete(audio_path):
            if audio_path is None:
                return ""
            try:
                text = transcriber.transcribe(audio_path)
                return text
            except Exception as e:
                print(f"转写错误: {str(e)}")
                return "语音转写失败，请重试或直接输入文字。"

        def on_send(user_input, chatbot, session_id, top_k, top_p, temperature):
            if not user_input:
                return chatbot, ""
            
            response = process_message(
                user_input, session_id, top_k, top_p, temperature
            )
            chatbot.append((user_input, response))
            return chatbot, ""

        def on_new_session(user_id):
            session_id = create_new_session(user_id)
            sessions = get_user_sessions(user_id)
            return (
                gr.Dropdown(choices=[(str(s['created_at']), s['session_id'])
                                     for s in sessions]),
                session_id,
                []
            )

        def on_delete_session(session_id, user_id):
            if session_id:
                delete_session(session_id)
                new_session_id = create_new_session(user_id)
                sessions = get_user_sessions(user_id)
                return (
                    gr.Dropdown(choices=[(str(s['created_at']), s['session_id'])
                                         for s in sessions]),
                    new_session_id,
                    []
                )
            return None, None, None

        def on_session_select(session_id):
            messages = get_session_messages(session_id)
            chatbot = [(msg['user_input'], msg['system_response'])
                       for msg in messages]
            return chatbot

        

        # 绑定事件
        audio_input.change(
            on_audio_complete,
            inputs=[audio_input],
            outputs=[user_input]
        )

        send_btn.click(
            on_send,
            inputs=[user_input, chatbot, current_session,
                    top_k, top_p, temperature],
            outputs=[chatbot, user_input]
        )

        new_session_btn.click(
            on_new_session,
            inputs=[user_id],
            outputs=[sessions_list, current_session, chatbot]
        )

        delete_session_btn.click(
            on_delete_session,
            inputs=[current_session, user_id],
            outputs=[sessions_list, current_session, chatbot]
        )

        sessions_list.change(
            on_session_select,
            inputs=[sessions_list],
            outputs=[chatbot]
        )
        
        delete_session_btn.click(
            on_delete_session,
            inputs=[current_session, user_id],
            outputs=[sessions_list, current_session, chatbot]
        )

        sessions_list.change(
            on_session_select,
            inputs=[sessions_list],
            outputs=[chatbot]
        )

    return demo

if __name__ == "__main__":
    init_database()
    demo = create_interface()
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False, debug=True)