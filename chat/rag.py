import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

SOURCE_NAMES = {
    "8205oxford": "Oxford Handbook of Clinical Medicine",
    "clinical-guidelines": "MSF Clinical Guidelines",
    "who-mhp": "WHO Essential Medicines",
}

def clean_source_name(filename):
    filename_lower = filename.lower()
    for key, clean_name in SOURCE_NAMES.items():
        if key in filename_lower:
            return clean_name
    return filename

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1024,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

prompt = PromptTemplate.from_template("""
You are MediBot, a professional AI medical assistant with extensive medical knowledge.

Use the context below as your primary source. If the context does not contain enough information, use your own medical knowledge to give a complete and helpful answer.

Format your answer clearly:
- Use **bold** for key medical terms and headings
- Use numbered lists for multiple points or steps
- Add a blank line between sections
- Use simple language a non-doctor can understand
- Always end with: This is for informational purposes only.
- For serious conditions always add: Please consult a licensed doctor.

Context:
{context}

Question: {question}

MediBot:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask_medibot(question):
    docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    sources = list(set([
        clean_source_name(os.path.basename(doc.metadata.get("source", "Unknown")))
        for doc in docs
    ]))
    return answer, sources
