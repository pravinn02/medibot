import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_FOLDER = "medical_docs"

print("📄 Loading PDFs...")
docs = []
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDFs: {pdf_files}\n")

for file in pdf_files:
    path = os.path.join(PDF_FOLDER, file)
    print(f"  Loading: {file}")
    try:
        loader = PyMuPDFLoader(path)
        pages = loader.load()
        docs.extend(pages)
        print(f"  ✅ {len(pages)} pages loaded")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

print(f"\n✅ Total pages loaded: {len(docs)}")

print("\n✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=[
        "\n\n\n",
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
        "; ",
        ": ",
        " ",
    ],
    length_function=len,
    is_separator_regex=False,
)
chunks = splitter.split_documents(docs)
print(f"✅ Total chunks: {len(chunks)}")

print("\n🔢 Generating embeddings (this may take 5-10 minutes)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\n🗄️  Building FAISS index...")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

print(f"\n✅ FAISS index saved!")
print(f"📊 Total vectors: {vectorstore.index.ntotal}")
print(f"\n📚 Sources indexed:")
for f in pdf_files:
    print(f"   - {f}")