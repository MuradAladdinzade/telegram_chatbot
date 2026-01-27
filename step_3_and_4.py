import os
import re
import time
import traceback
import socket
from typing import Dict, List

import requests
import telebot
from telebot import apihelper
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser


# Telegram timeouts
apihelper.CONNECT_TIMEOUT = 30
apihelper.READ_TIMEOUT = 120


# Env

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Put it in your .env")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN is missing. Put it in your .env")


# Telegram preflight check

def telegram_preflight(token: str) -> None:
    """Fail fast if Telegram isn't reachable or token is invalid."""
    url = f"https://api.telegram.org/bot{token}/getMe"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram getMe failed: {data}")
    bot_username = data["result"].get("username")
    print(f"✅ Telegram OK. Bot username: @{bot_username}")

telegram_preflight(BOT_TOKEN)


# Telegram bot

bot = telebot.TeleBot(BOT_TOKEN)


# Vector DB 

embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

db = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


# LLM + Prompt (with history)

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4.1-2025-04-14",
    temperature=0.00
)
parser = StrOutputParser()

SYSTEM_RULES = (
    "You are a helpful assistant answering labour code related questions. You will be given relevant context everytime a question is asked.  Always refer to the context, and never go beyond that. If it is asking about the definition of a term, use only sentences from the context."
    "Use ONLY the provided context, don't refer to your knowledge. If the most probable answer is not in the context, you can ask probing question in order to clarify the question and get more context from the user. Never say 'Bilmirəm' without trying to understand the question. If you can't find the answer after 2nd attempt of clarification, say 'Bilmirəm'. "
    "You must answer ONLY in Azerbaijani. Be concise in your answers."
    "Output must use correct Azerbaijani letters."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_RULES),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
    ]
)

# Chat history memory
HISTORY = {}
MAX_HISTORY_MESSAGES = 20

def trim_history(chat_id: int) -> None:
    msgs = HISTORY.get(chat_id, [])
    if len(msgs) > MAX_HISTORY_MESSAGES:
        HISTORY[chat_id] = msgs[-MAX_HISTORY_MESSAGES:]


# Normalization for understanding
def normalize_user_text(text: str) -> str:
    text = re.sub(r"(?i)sh", "ş", text)
    text = re.sub(r"(?i)ch", "ç", text)
    text = re.sub(r"(?i)w", "ş", text)
    return text


# RAG answer function (with memory)

def answer_with_rag(chat_id: int, user_text: str) -> str:
    normalized_q = normalize_user_text(user_text.strip())

    docs = retriever.get_relevant_documents(normalized_q)
    context = format_docs(docs)

    history_msgs = HISTORY.get(chat_id, [])

    messages = prompt.format_messages(
        question=normalized_q,
        context=context,
        history=history_msgs
    )

    ai_text = parser.invoke(llm.invoke(messages))

    HISTORY.setdefault(chat_id, []).append(HumanMessage(content=normalized_q))
    HISTORY[chat_id].append(AIMessage(content=ai_text))
    trim_history(chat_id)

    return ai_text


# Telegram handlers

@bot.message_handler(commands=["start", "hello"])
def send_welcome(message):
    bot.reply_to(
        message,
        "Salam! Mən RAG əsaslı chatbotam. Əmək Məcəlləsilə bağlı sualını yaz, mən cavab verəcəyəm.\n"
        "Əmrlər: /reset (yaddaşı sil)"
    )

@bot.message_handler(commands=["reset"])
def reset_memory(message):
    chat_id = message.chat.id
    HISTORY.pop(chat_id, None)
    bot.reply_to(message, "Bu çat üçün yaddaş sıfırlandı.")

@bot.message_handler(func=lambda msg: True, content_types=["text"])
def handle_text(message):
    chat_id = message.chat.id
    user_text = message.text or ""

    if not user_text.strip():
        bot.reply_to(message, "Sualı yaz.")
        return

    if user_text.strip().startswith("/"):
        bot.reply_to(message, "Bu əmri tanımıram.")
        return

    try:
        bot.send_chat_action(chat_id, "typing")
        answer = answer_with_rag(chat_id, user_text)
        bot.send_message(chat_id, answer)
    except Exception:
        traceback.print_exc()
        bot.send_message(chat_id, "Xəta baş verdi. Zəhmət olmasa bir daha yoxla.")

if __name__ == "__main__":
    while True:
        try:
            bot.infinity_polling(timeout=20, long_polling_timeout=20, skip_pending=True)
        except Exception as e:
            print("❌ Polling error:", e)
            time.sleep(5)
