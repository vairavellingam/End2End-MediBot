from dotenv import load_dotenv
import os

from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone 
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


extracted_docs = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_docs)
texts_chunk = text_split(minimal_docs)
embeddings = download_embeddings()

# Initialize Pinecone client
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

# Create a new index
index_name = "medibot"

if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name = index_name,
        dimension=384,  # Dimension of the embeddings
        metric= "cosine",  # Cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# Create a Pinecone vector store from the text chunks and embeddings
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding= embeddings,
    index_name=index_name
)