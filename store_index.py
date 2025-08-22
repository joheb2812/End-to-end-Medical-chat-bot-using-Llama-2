from pinecone import Pinecone, ServerlessSpec
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
# from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone as PineconeStore

import os


load_dotenv()  # Load environment variables from .env file
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")


extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)


embeddings = download_hugging_face_embeddings()

pc = Pinecone(PINECONE_API_KEY)

index_name = "medical-chatbot"

# If the index doesn't exist, create it 
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Now connect to the index
index = pc.Index(index_name)

# For LangChain integration
docsearch = PineconeStore.from_texts(
    [t.page_content for t in text_chunks],
    embeddings,
    index_name=index_name
)
