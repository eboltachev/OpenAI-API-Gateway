import os
import json

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI as LangchainOpenAI
from openai import OpenAI

load_dotenv()

target_url = "http://127.0.0.1:11435/v1"
target_api_key = os.getenv("ENDPOINT_API_KEY")
chat_routes = json.loads(os.getenv("OPENAI_CHAT_MODEL_ROUTES", "{}") or "{}")
default_model = next((m for m in chat_routes.keys() if m != "*"), None) or os.getenv("TEST_CHAT_MODEL")
assert default_model, "Set OPENAI_CHAT_MODEL_ROUTES or TEST_CHAT_MODEL"

# Langchain OpenAI simple
llm = LangchainOpenAI(model=default_model, base_url=target_url, api_key=target_api_key)
print("Langchain OpenAI simple:")
print(llm.invoke("Hello how are you?"))

# Langchain OpenAI stream
llm = LangchainOpenAI(model=default_model, base_url=target_url, api_key=target_api_key)
print("Langchain OpenAI response:")
for chunk in llm.stream("Write a short poem about a rainy day."):
    print(chunk, end="|", flush=True)

# ChatOpenAI stream
llm = ChatOpenAI(model=default_model, base_url=target_url, api_key=target_api_key)
messages = [HumanMessage(content="Write a short poem about a rainy day.")]
print("ChatOpenAI response:")
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)

# OpenAI strem
client = OpenAI(base_url=target_url, api_key=target_api_key)
response = client.chat.completions.create(
    model=default_model,
    messages=[
        {"role": "user", "content": "Write a short poem about a rainy day."},
    ],
    stream=True,
)
print("OpenAI response:")
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")
