import os
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()


class Config:

    # Get the LLM Provider and the LLM Model form .env file 
    LLM_PROVIDER = os.getenv("LLM_PROVIDER")
    LLM_MODEL = os.getenv("LLM_MODEL")


config=Config()