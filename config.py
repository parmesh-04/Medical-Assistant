import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

class Config:
    def __init__(self):
    
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            disable_streaming=True
        )

       
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

      
        self.vector_db_type = "qdrant"
        self.vector_local_path = "./data/qdrant_db"
        self.collection_name = "medical_assistance_rag"

config = Config()