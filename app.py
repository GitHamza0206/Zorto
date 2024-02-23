import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from datetime import datetime
import json 
import sys 

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore


def conversation_chain(vectorstore):
    
    llm=ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = "chat_history", return_message = False)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def main():
    st.header("Contract Validator")
    st.markdown("This agent is able to read the contract and determine whether the validation date is expired or not")

    contract = st.file_uploader("Upload your contract here", accept_multiple_files = False)
    if contract:
        file_name = contract.name
        file_extension = file_name.split('.')[-1]
        if file_extension in ['pdf']:
            if st.button('Process'):
                with st.spinner('Processing...'): 
                    try: 
                        raw_text = get_pdf_text(contract)
                        text_chunks = get_text_chunks(raw_text)
                        vector_store = get_vector_store(text_chunks)
                        conv = conversation_chain(vector_store)
                        st.session_state.conversation = conv
                        answer = conv({"question": 'Extract the date of expiration of the contract in this format Day/Month/Year. You answer should be the date only in JSON format like this {"expiration_date":"extracted_date}"'})
                        str_answer = str(answer['answer'])
                        json_answer = json.loads(str_answer)

                        str_date = json_answer['expiration_date']
                        dt_date = datetime.strptime(str_date, "%d/%m/%Y")
                        current_date = datetime.today()
                        diff_days=abs((dt_date - current_date).days)

                        if dt_date<current_date:
                            st.write(f'The contract has expired by {diff_days} days')
                        else:
                            st.write(f'The contract is Valid. It will expire in {diff_days} days')

                    except Exception as e:
                        st.error(e)
                        st.error("Error while processing the file...")
        else:
            st.error('Uploaded files should be in PDF...')
        
if __name__=="__main__":
    main()