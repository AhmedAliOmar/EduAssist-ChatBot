import os
from flask import Flask, render_template, request, redirect
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from summarizer import generate_summary
from mcq_generator import generate_mcqs

# === Setup ===
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
DATA_DIR = "__data__"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

app = Flask(__name__)
vectorstore = None
conversation_chain = None
chat_history = []

# === Helper Functions ===
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        filename = os.path.join(DATA_DIR, pdf.filename)
        pdf_reader = PdfReader(pdf)
        pdf_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
                pdf_text += page_text
        with open(filename, "w", encoding="utf-8") as f:
            f.write(pdf_text)
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def get_conversation_chain(vstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vstore.as_retriever(),
        memory=memory
    )

# === Routes ===
@app.route('/')
def home():
    return render_template('new_home.html')

@app.route('/process', methods=['POST'])
def process_documents():
    global vectorstore, conversation_chain
    pdf_docs = request.files.getlist('pdf_docs')
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    return redirect('/chat')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global conversation_chain, chat_history
    if request.method == 'POST':
        user_question = request.form['user_question']
        response = conversation_chain({'question': user_question})
        chat_history = response['chat_history']
    return render_template('new_chat.html', chat_history=chat_history)

@app.route('/summary', methods=['GET'])
def summarize():
    global vectorstore
    if not vectorstore:
        return "No document uploaded."
    doc_text = "\n".join([doc.page_content for doc in vectorstore.docstore._dict.values()])
    summary = generate_summary(doc_text)
    return render_template("summary.html", summary=summary)

@app.route('/generate_mcqs', methods=['GET'])
def generate_mcq():
    global vectorstore
    if not vectorstore:
        return "No document uploaded."
    doc_text = "\n".join([doc.page_content for doc in vectorstore.docstore._dict.values()])
    mcqs = generate_mcqs(doc_text, num_questions=5)
    return render_template("mcq.html", mcqs=mcqs)

if __name__ == '__main__':
    app.run(debug=True)
