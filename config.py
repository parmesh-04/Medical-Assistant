import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load variables from .env
load_dotenv()

class Config:
    def __init__(self):
        # --- GEMINI LLM (The Brain) ---
        # gemini-1.5-flash is fast and great for agent orchestration
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            disable_streaming=True
        )

        # --- GEMINI EMBEDDINGS (The Memory) ---
        # text-embedding-04 is Google's state-of-the-art embedding model
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # --- LOCAL VECTOR DB (QDRANT) ---
        # We keep the database local so your medical data stays on your machine
        self.vector_db_type = "qdrant"
        self.vector_local_path = "./data/qdrant_db"
        self.collection_name = "medical_assistance_rag"

# Instantiate for use in other files
config = Config()