import os

import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings.base import OpenAIEmbeddings
from openai import OpenAI

load_dotenv()

target_url = "http://127.0.0.1:11435/v1"
api_key = os.getenv("ENDPOINT_API_KEY", "")
assert api_key.startswith("sk-")

source_default_model = os.getenv("OPENAI_DEFAULT_MODEL")
source_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
source_transcription_model = os.getenv("OPENAI_TRANSCRIPTION_MODEL")

# Models
target_client = OpenAI(base_url=target_url, api_key=api_key)
target_models = {model.id for model in target_client.models.list()}
assert source_default_model in target_models

# Chat
target_llm_chat = ChatOpenAI(base_url=target_url, api_key=api_key, model=source_default_model, temperature=0)
messages = [
    ("system", "You are a helpful assistant that translates English to French. Translate the user sentence."),
    ("human", "I love programming."),
]
target_ai_msg = target_llm_chat.invoke(messages)
assert isinstance(target_ai_msg.content, str)

# Embeddings
target_embeddings = OpenAIEmbeddings(base_url=target_url, api_key=api_key, model=source_embedding_model)
target_one_vector = target_embeddings.embed_query("text")
target_two_vectors = target_embeddings.embed_documents(["text", "text2"])
assert isinstance(target_one_vector, list)
assert np.array(target_two_vectors).shape[0] == 2

# Transcriptions
target_transcriptions = OpenAI(base_url=target_url, api_key=api_key)
with open("datasets/ru.mp3", "rb") as file:
    target_text = target_transcriptions.audio.transcriptions.create(model=source_transcription_model, file=file)
assert isinstance(target_text.text, str)

# Result
print("ok")
