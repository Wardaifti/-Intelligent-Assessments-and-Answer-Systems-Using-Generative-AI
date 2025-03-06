import streamlit as st
import time
import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA 
from langchain_community.chat_models import ChatOpenAI
import tiktoken
import PyPDF2
import openai

load_dotenv(find_dotenv(), override=True)

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()  # Corrected method
    return text

def count_tokens(text):
    enc = tiktoken.encoding_for_model('gpt-3.5-turbo')
    return len(enc.encode(text))

def generate_mcqs(text_chunk, api_key, num_questions):
    openai.api_key = api_key
    
    prompt = (
        f"Create {num_questions} multiple-choice questions based on the following text:\n\n"
        f"{text_chunk}\n\n"
        "Each question should have one correct answer and three incorrect answers."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        n=1,
        stop=None,
        temperature=0.7
    )
    
    return response['choices'][0]['message']['content'].strip()

def generate_mcqs_main(file, api_key, num_questions):
    text = extract_text_from_pdf(file)
    st.write("Extracted text from PDF.")
    
    try:
        tokens = count_tokens(text)
        st.write(f"Text contains {tokens} tokens.")
        mcqs = generate_mcqs(text, api_key, num_questions)
        st.write(f"Generated MCQs:\n{mcqs}\n")
        
        time.sleep(1)
    except Exception as e:
        st.error(f"An error occurred while generating MCQs: {e}")
    return mcqs

def qa_main(file):
    text = extract_text_from_pdf(file)
    if text:
        st.write('Loaded Document successfully.')
        data = [Document(page_content=text)]
        enc = tiktoken.encoding_for_model('text-embedding-ada-002')
        total_tokens = sum([len(enc.encode(page.page_content)) for page in data])
        embedding_Cost = total_tokens / 1000 * 0.0004
        st.write(f"Embedding cost: ${embedding_Cost:.4f}")
        
        open_api_key = os.getenv('OPENAI_API_KEY')
        embeddings = OpenAIEmbeddings(api_key=open_api_key)  # Ensure correct usage of variable
        vector_Store = Chroma.from_documents(data, embeddings)
        
        question = st.text_input("Ask anything about the content of your file:")
        if st.button("Get answer") and question:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1, api_key=open_api_key)
            retriever = vector_Store.as_retriever(search_type="similarity", search_kwargs={'k': 3})
            chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            answer = chain.run(question)
            st.write(f'Answer: {answer}')

st.title("Content Processor")
file = st.file_uploader("Upload a PDF file", type="pdf")
action = st.selectbox("Do you want to create MCQ or ask anything about your content?", ["Select", "MCQ", "QA"])

if file and action != "Select":
    api_key = os.getenv("OPENAI_API_KEY")
    if action == "MCQ":
        num_questions = st.number_input("Enter the number of questions to be generated:", min_value=1, step=1)
        if st.button("Generate MCQ"):
            with st.spinner("Generating MCQs...."):
                mcqs = generate_mcqs_main(file, api_key, num_questions)
                st.write("Generated MCQs:")
                st.write(mcqs)
    elif action == "QA":
        qa_main(file)
