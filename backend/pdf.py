from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.chains import RetrievalQA
# from langchain_community.llms import GooglePalm
from pydantic import BaseModel
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
#_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Plan-and-execute"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyC6FhO2BCsCoLp9aYjMnq59mzZ-hkKImi0"

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Step 2: Use LangChain’s RecursiveCharacterTextSplitter for consistent chunks
def chunk_text_with_langchain(pdf_path, chunk_size=500, chunk_overlap=50):
    text = extract_text_from_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    # Wrap each chunk in a Document object
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

# Load and split the PDF
def get_pdf_response(pdf):
    chunks = chunk_text_with_langchain(pdf)

    # Generate embeddings for each chunk
    

    vectorstore = Chroma.from_documents(
        documents=chunks,
        collection_name="rag-chroma",
        embedding= GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" concatenates all retrieved chunks; other types may work depending on need
        retriever=retriever,
        return_source_documents=True  # Optionally return the source documents
    )

    # Step 3: Ask a question and get an answer
    question = "What is the main point of the database course material?"
    response = qa_chain({"query": question})
    return response


response = get_pdf_response(r"C:\Users\owner\Desktop\School\200L\CSC 272 - Intro to database 2.pdf")
print(response)
# # Optionally, display the source documents
# for i, doc in enumerate(response["source_documents"], 1):
#     print(f"\nSource {i}:")
#     print(doc.page_content)
