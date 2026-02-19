import os

import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings.base import OpenAIEmbeddings
from openai import OpenAI

load_dotenv()

target_url = "http://127.0.0.1:11435/v1"
target_api_key = os.getenv("ENDPOINT_API_KEY")

source_default_model = os.getenv("OPENAI_DEFAULT_MODEL")
source_base_url = os.getenv("OPENAI_BASE_URL")
source_api_key = os.getenv("OPENAI_API_KEY")

source_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
source_embedding_url = os.getenv("OPENAI_EMBEDDING_URL")
source_embedding_key = os.getenv("OPENAI_EMBEDDING_KEY")

source_transcription_model = os.getenv("OPENAI_TRANSCRIPTION_MODEL")
source_transcription_url = os.getenv("OPENAI_TRANSCRIPTION_URL")
source_transcription_key = os.getenv("OPENAI_TRANSCRIPTION_KEY")


# Chat
target_llm_chat = ChatOpenAI(base_url=target_url, api_key=target_api_key, model=source_default_model, temperature=0)
source_llm_chat = ChatOpenAI(
    base_url=source_base_url, api_key=source_api_key, model=source_default_model, temperature=0
)

messages = [
    ("system", "You are a helpful assistant that translates English to French. Translate the user sentence."),
    ("human", "I love programming."),
]
target_ai_msg = target_llm_chat.invoke(messages)
source_ai_msg = source_llm_chat.invoke(messages)
assert target_ai_msg.content == source_ai_msg.content
assert set(target_ai_msg.model_dump().keys()) == set(source_ai_msg.model_dump().keys())

# Models
target_client = OpenAI(base_url=target_url, api_key=target_api_key)
source_client = OpenAI(base_url=source_base_url, api_key=source_api_key)
target_models = {model.id for model in target_client.models.list()}
source_models = {model.id for model in source_client.models.list()}
assert target_models == source_models
assert set(target_client.models.list().model_dump().keys()) == set(source_client.models.list().model_dump().keys())

# Embeddings
target_embeddings = OpenAIEmbeddings(base_url=target_url, api_key=target_api_key, model=source_embedding_model)
source_embeddings = OpenAIEmbeddings(
    base_url=source_embedding_url, api_key=source_embedding_key, model=source_embedding_model
)
target_one_vector = target_embeddings.embed_query("text")
source_one_vector = source_embeddings.embed_query("text")
assert target_one_vector == source_one_vector
target_two_vectors = target_embeddings.embed_documents(["text", "text2"])
source_two_vectors = source_embeddings.embed_documents(["text", "text2"])
assert np.array(target_two_vectors).shape == np.array(source_two_vectors).shape
assert target_two_vectors[0] == source_two_vectors[0]
assert target_two_vectors[1] == source_two_vectors[1]

# Transcriptions
target_transcriptions = OpenAI(base_url=target_url, api_key=target_api_key)
source_transcriptions = OpenAI(base_url=source_transcription_url, api_key=source_transcription_key)
with open(f"datasets/ru.mp3", "rb") as file:
    target_text = target_transcriptions.audio.transcriptions.create(model=source_transcription_model, file=file)
    source_text = source_transcriptions.audio.transcriptions.create(model=source_transcription_model, file=file)
assert target_text == source_text
