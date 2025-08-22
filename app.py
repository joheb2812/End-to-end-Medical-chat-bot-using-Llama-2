from flask import Flask, render_template, jsonify, request
# from langchain import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain_community.llms import CTransformers
# from pinecone import Pinecone, ServerlessSpec
# from dotenv import load_dotenv 
# from src.prompt import *
# from src.helper import download_hugging_face_embeddings
# from langchain_community.vectorstores import Pinecone as PineconeStore
# from src.helper import load_pdf, text_split, download_hugging_face_embeddings


# import os

# app = Flask(__name__) 

# load_dotenv()

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

# extracted_data = load_pdf("data/")
# text_chunks = text_split(extracted_data)
# embeddings = download_hugging_face_embeddings()

# # Initialize Pinecone client
# pc = Pinecone(api_key=PINECONE_API_KEY)

# index_name = "medical-chatbot"

# # If the index doesn't exist, create it
# if index_name not in [index.name for index in pc.list_indexes()]:
#     pc.create_index(
#         name=index_name,
#         dimension=384,  
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )

# # Now connect to the index 
# index = pc.Index(index_name)

# # For LangChain integration
# from langchain_community.vectorstores import Pinecone as PineconeStore

# docsearch = PineconeStore.from_texts(
#     [t.page_content for t in text_chunks],
#     embeddings,
#     index_name=index_name
# )





# PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# chain_type_kwargs={"prompt": PROMPT}


# llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
#                   model_type="llama",
#                   config={'max_new_tokens':512,
#                           'temperature':0.8})


# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
#     return_source_documents=True,
#     chain_type_kwargs=chain_type_kwargs)

# app.route("/")
# def index():
#     return render_template('chat.html')


# if __name__ =='__main__':
#     app.run(debug=True)


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

# 🔹 Initialize embeddings
embeddings = download_hugging_face_embeddings()

# 🔹 Connect to existing Pinecone index (no PDF reloading here)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# 🔹 Attach LangChain vector store to Pinecone
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
