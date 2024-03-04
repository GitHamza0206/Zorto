import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from datetime import datetime
import json 
import sys 
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import requests 
import os 


os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
BASE_URL = "https://api.pdf.co/v1"
#SourceFile = "POA.pdf"
#DestinationFile = "result.pdf"
Password = ""
PDF_API_KEY = st.secrets["PDF_API_KEY"]


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




def create_text_overlay(text, filename="text_overlay.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100,100, text)  # Adjust the position (100, 750) as needed
    c.save()


def merge_pdfs(original_pdf, overlay_pdf, output_pdf="merged.pdf"):
    # Create a PDF reader object for the original and overlay PDFs
    original_reader = PdfReader(original_pdf)
    overlay_reader = PdfReader(overlay_pdf)
    writer = PdfWriter()

    # Iterate through the original PDF pages
    for page_number in range(len(original_reader.pages)):
        original_page = original_reader.pages[page_number]
        # Get the corresponding page from the overlay PDF
        # Assuming overlay PDF has at least as many pages as the original
        overlay_page = overlay_reader.pages[0] if page_number == 0 else overlay_reader.pages[page_number % len(overlay_reader.pages)]

        # Merge the overlay page onto the original page
        original_page.merge_page(overlay_page)

        # Add the merged page to the writer object
        writer.add_page(original_page)
    # Write the merged content to a new PDF file
    with open(output_pdf, "wb") as output_file:
        writer.write(output_file)
    
    with open(output_pdf, "rb") as pdf_file:
        PDFbyte = pdf_file.read()
        st.download_button( label="Download PDF",
                            data=PDFbyte,
                            file_name="downloaded_document.pdf",
                            mime="application/octet-stream")
    
        
def conversation_chain(vectorstore):
    
    llm=ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = "chat_history", return_message = False)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain

import tempfile

def uploadFile(fileName):
    """Uploads file to the cloud"""
    
    # 1. RETRIEVE PRESIGNED URL TO UPLOAD FILE.

    # Prepare URL for 'Get Presigned URL' API request
    url = "{}/file/upload/get-presigned-url?contenttype=application/octet-stream&name={}".format(
        BASE_URL, os.path.basename(fileName))
    
    # Execute request and get response as JSON
    response = requests.get(url, headers={ "x-api-key": PDF_API_KEY })
    if (response.status_code == 200):
        json = response.json()
        
        if json["error"] == False:
            # URL to use for file upload
            uploadUrl = json["presignedUrl"]
            # URL for future reference
            uploadedFileUrl = json["url"]

            # 2. UPLOAD FILE TO CLOUD.
            with open(fileName, 'rb') as file:
                requests.put(uploadUrl, data=file, headers={ "x-api-key": PDF_API_KEY, "content-type": "application/octet-stream" })

            return uploadedFileUrl
        else:
            # Show service reported error
            print(json["message"])    
    else:
        print(f"Request error: {response.status_code} {response.reason}")

    return None

def replaceStringFromPdf(uploadedFileUrl, destinationFile, searchString, replaceString ):
    """Replace Text FROM UPLOADED PDF FILE using PDF.co Web API"""

    # Prepare requests params as JSON
    # See documentation: https://apidocs.pdf.co
    parameters = {}
    parameters["name"] = os.path.basename(destinationFile)
    parameters["password"] = Password
    parameters["url"] = uploadedFileUrl
    parameters["searchString"] = searchString
    parameters["replaceString"] = replaceString

    # Prepare URL for 'Replace Text from PDF' API request
    url = "{}/pdf/edit/replace-text".format(BASE_URL)

    # Execute request and get response as JSON
    response = requests.post(url, data=parameters, headers={ "x-api-key": PDF_API_KEY })
    if (response.status_code == 200):
        json = response.json()

        if json["error"] == False:
            #  Get URL of result file
            resultFileUrl = json["url"]            
            # Download result file
            r = requests.get(resultFileUrl, stream=True)
            if (r.status_code == 200):
                with open(destinationFile, 'wb') as file:
                    for chunk in r:
                        file.write(chunk)

                with open(destinationFile, "rb") as pdf_file:
                    PDFbyte = pdf_file.read()
                    st.download_button( label="Download PDF",
                                        data=PDFbyte,
                                        file_name=destinationFile,
                                        mime="application/octet-stream")
                print(f"Result file saved as \"{destinationFile}\" file.")
            else:
                print(f"Request error: {response.status_code} {response.reason}")
        else:
            # Show service reported error
            print(json["message"])
    else:
        print(f"Request error: {response.status_code} {response.reason}")


def modifyPDF(SourceFile, DestinationFile, searchString,replaceString):
    uploadedFileUrl = uploadFile(SourceFile)
    if (uploadedFileUrl != None):
        replaceStringFromPdf(uploadedFileUrl, DestinationFile, searchString, replaceString)


def main():
    st.header("Contract Validator")
    st.markdown("This agent is able to read the contract and determine whether the validation date is expired or not")

    contract = st.file_uploader("Upload your contract here", accept_multiple_files = False)
    if contract:
        file_name = contract.name
        file_extension = file_name.split('.')[-1]
        
        temp_dir = tempfile.mkdtemp() 
        sourceFile = os.path.join(temp_dir, contract.name) 
        with open(sourceFile, "wb") as f: 
            f.write(contract.getvalue())

        if file_extension in ['pdf']:
            
            if st.button('Process'):
                with st.spinner('Processing...'): 
                    try: 
                        raw_text = get_pdf_text(contract)
                        text_chunks = get_text_chunks(raw_text)
                        vector_store = get_vector_store(text_chunks)
                        conv = conversation_chain(vector_store)
                        st.session_state.conversation = conv
                        answer = conv({"question": 'Extract the date of expiration of the contract in the exact format of the document PDF. I want you also to transform this extracted date in this format: day/month/year ,  You answer should be the date only in JSON format like this {"expiration_date":"extracted_date, "formated_date":"formated_date" }"'})
                        str_answer = str(answer['answer'])
                        json_answer = json.loads(str_answer)

                        str_date = json_answer['expiration_date']

                        formated_date = json_answer['formated_date']
                        dt_date = datetime.strptime(formated_date, "%d/%m/%Y")
                        current_date = datetime.today()
                        diff_days=abs((dt_date - current_date).days)


                        if dt_date<current_date:
                            new_date = str_date[:-1] + str( int(str_date[-1]) +1 )
                            st.write(f'The contract has expired by {diff_days} days')
                            out_name = file_name.split('.')[0] + '_out.pdf'
                            #bytes_data = contract.getvalue()
                            st.write(str_date)
                            st.write(new_date)
                            modifyPDF(sourceFile, out_name, searchString = str_date, replaceString=new_date)
                            
                            
                        else:
                            st.write(f'The contract is Valid. It will expire in {diff_days} days')

                    except Exception as e:
                        st.error(e)
                        st.error("Error while processing the file...")
        else:
            st.error('Uploaded files should be in PDF...')
        
if __name__=="__main__":
    main()