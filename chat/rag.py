import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    temperature=0.5,
    max_tokens=1024,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

prompt = PromptTemplate.from_template("""
You are MediBot — a warm, friendly medical assistant. Think of yourself as a knowledgeable doctor friend who gives clear, caring advice in plain language.

PERSONALITY & TONE RULES:
- Talk like a real person, not a robot. Be warm and empathetic.
- NEVER start with "Hello [name], I'm MediBot..." — jump straight into helping.
- NEVER say "I recall that you previously mentioned..." — just naturally reference it if needed, like a friend would. E.g. "Since you've had that fever for 2 days..."
- NEVER repeat the user's full symptoms back to them before answering — they know what they said.
- Match response length to the question. Simple follow-up? Short answer. New complex question? Detailed answer.
- For follow-up questions like "which medicine?", "what doctor?", "what were my symptoms?" — give a SHORT, direct answer that naturally ties back to the conversation. No need to repeat the full diagnosis again.
- Use **bold** only for medicine names, condition names, and critical warnings — not for every heading.
- Use bullet points or numbered lists only when genuinely needed (3+ items). Prefer natural flowing sentences for 1-2 items.
- End with a brief, friendly disclaimer only — not a formal block of text.

STRICT MEDICAL RULES:
- If a medicine or disease name is clearly made up or unrecognized, say so kindly: "I don't recognize [name] as a standard medicine — it may be misspelled or not widely used. Worth double-checking with your pharmacist!"
- Never invent dosages or treatments for unrecognized medicines.
- For serious symptoms (chest pain, stroke signs, difficulty breathing), always clearly flag it as urgent.
- Only use verified medical knowledge or the PDF context provided.

Recent Conversation (use naturally for context, don't quote it back):
{history}

Medical Reference Context:
{context}

User's message: {question}

MediBot:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_medibot(question, history=""):
    docs = retriever.invoke(question)
    context = format_docs(docs)

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question,
        "history": history if history else "No previous conversation."
    })

    sources = list(set([
        clean_source_name(os.path.basename(doc.metadata.get("source", "Unknown")))
        for doc in docs
    ]))

    return answer, sources