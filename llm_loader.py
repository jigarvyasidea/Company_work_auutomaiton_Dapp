import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceEndpoint

# Load environment variables from .env file
load_dotenv()

# HuggingFace API Token
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Create HuggingFace endpoint-based LLM
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    huggingfacehub_api_token=token,
    task="text-generation"
)
