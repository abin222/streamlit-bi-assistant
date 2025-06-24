from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

def load_and_split_pdfs(pdf_folder_path):
    all_chunks = []
    for path in Path(pdf_folder_path).rglob("*.pdf"):
        loader = PyPDFLoader(str(path))
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)
    return all_chunks

def create_vectorstore(chunks, persist_directory="vectorstore/"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
    vectorstore.persist()
    return vectorstore

if __name__ == "__main__":
    chunks = load_and_split_pdfs("PDF_Folder")
    vectorstore = create_vectorstore(chunks)
    print("✅ Vectorstore created and saved.")
