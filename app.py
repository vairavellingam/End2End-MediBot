from flask import Flask, request, jsonify, render_template
from src.helper import download_embeddings
from src.adaptive_rag import build_adaptive_rag
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()


# Environment

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


# Build retriever from existing Pinecone index

embeddings = download_embeddings()
index_name = "medibot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# Build the Adaptive RAG LangGraph pipeline

adaptive_rag_app = build_adaptive_rag(retriever)


# Flask routes

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"\n[User]: {msg}")

    # Stream through the graph and collect the final generation
    # retries is initialised to 0 so GraphState.retries is always defined
    final_output = None
    for output in adaptive_rag_app.stream({"question": msg, "retries": 0, "rewrite_count": 0}):
        for node_name, node_value in output.items():
            print(f"  [Node: {node_name}]")
            final_output = node_value  

    answer = final_output.get("generation", "I'm sorry, I couldn't find an answer.")
    print(f"[MediBot]: {answer}\n")
    return str(answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
