from flask import Flask, render_template, jsonify, request
from flask import Flask, render_template
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from pinecone import Pinecone
from dotenv import load_dotenv
from src.prompt import *
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as PineconeStore
import os

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "medical-chatbot"

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Connect to existing Pinecone index (no PDF reloading here)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# Attach LangChain vector store to Pinecone
docsearch = PineconeStore(
    index, embeddings.embed_query, "text"   # "text" is the default metadata field
)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        msg = request.form.get("msg") or request.json.get("msg")
    else:
        msg = request.args.get("msg")

    if not msg:
        return jsonify({"error": "No message provided"}), 400

    print("User Query:", msg)
    result = qa.invoke({"query": msg})   
    print("Response:", result["result"])
    return str(result["result"])




if __name__ == "__main__":
    app.run(debug=True)
