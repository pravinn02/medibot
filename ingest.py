import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_FOLDER = "pdfs/"
docs = []

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        print(f"📄 Loading: {file}")
        loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
        docs.extend(loader.load())

print(f"✅ Total pages: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)
print(f"✅ Total chunks: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print("✅ FAISS index saved!")
