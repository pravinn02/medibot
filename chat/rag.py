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
    "hospital_care": "WHO Hospital Care for Children",
    "hospital care": "WHO Hospital Care for Children",
    "mhgap": "WHO mhGAP Mental Health Guidelines",
    "mhgap_intervention": "WHO mhGAP Mental Health Guidelines",
    "davidsonmedicine": "Davidson's Principles of Medicine 24th Edition",
    "davidson": "Davidson's Principles of Medicine 24th Edition",
    "who-cancer": "WHO Cancer Pain Guidelines",
    "who-surgical": "WHO Surgical Care Guidelines",
}

def clean_source_name(filename):
    filename_lower = filename.lower()
    for key, clean_name in SOURCE_NAMES.items():
        if key in filename_lower:
            return clean_name
    return filename

def detect_language(text):
    """Detect if message is Hindi, Marathi or English — supports Devanagari and Romanized"""
    text_lower = text.lower()

    # ── Devanagari script words ──
    marathi_devanagari = [
        'आहे', 'आणि', 'मला', 'तुम्ही', 'काय', 'करावे', 'घ्यावी',
        'दुखत', 'होत', 'माझ्या', 'त्वरित', 'नाही', 'आहात', 'कसे',
        'मराठी', 'छातीत', 'डाव्या', 'श्वास', 'घेणे', 'कठीण',
        'उपचार', 'लक्षणे', 'औषध', 'आजार', 'वेदना'
    ]
    hindi_devanagari = [
        'मुझे', 'मेरे', 'मेरा', 'मेरी', 'है', 'हैं', 'कौन', 'क्या',
        'लूं', 'खाऊं', 'बताओ', 'दर्द', 'बुखार', 'सिरदर्द', 'दवा',
        'गोली', 'नमस्ते', 'हिंदी', 'करें', 'होता', 'होती', 'कीजिए'
    ]

    # ── Romanized Marathi words ──
    marathi_roman = [
        'kay ahe', 'kay aahe', 'varti upchar', 'vr upchar',
        'upchar kay', 'lakshan kay', 'aajaar', 'aushadh',
        'marathi madhe', 'marathi mein', 'marathi me',
        'marathi madhye', 'give response in marathi',
        'respond in marathi', 'answer in marathi',
        'mala saang', 'tumhi saanga', 'kay karwe',
        'kay karaye', 'kasa ahe', 'kashe ahe',
        'aahe ka', 'nahi ka', 'sangaa', 'sanga',
        'tb varti', 'taap varti', 'dokedukhee',
        'dokhedukhee', 'taap', 'khokla',
    ]

    # ── Romanized Hindi words ──
    hindi_roman = [
        'kya hai', 'kya hota', 'mujhe', 'meri',
        'dawai', 'bukhar', 'dawa batao',
        'hindi mein', 'hindi me', 'hindi main',
        'give response in hindi', 'respond in hindi',
        'answer in hindi', 'batao', 'kya kare',
        'kaise', 'kaisa', 'ilaj', 'dawa kya',
        'tb ka ilaj', 'upchar kya',
    ]

    # ── Score Devanagari ──
    marathi_dev = sum(1 for w in marathi_devanagari if w in text)
    hindi_dev   = sum(1 for w in hindi_devanagari   if w in text)

    if marathi_dev > 0 and marathi_dev >= hindi_dev:
        return 'marathi'
    elif hindi_dev > 0:
        return 'hindi'

    # ── Score Romanized ──
    marathi_rom = sum(1 for w in marathi_roman if w in text_lower)
    hindi_rom   = sum(1 for w in hindi_roman   if w in text_lower)

    if marathi_rom > 0 and marathi_rom >= hindi_rom:
        return 'marathi'
    elif hindi_rom > 0:
        return 'hindi'

    return 'english'

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    max_tokens=1024,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

prompt = PromptTemplate.from_template("""
You are MediBot — a warm, friendly medical assistant. Think of yourself as a knowledgeable doctor friend who gives clear, caring advice in plain language.

LANGUAGE RULE — MOST IMPORTANT, FOLLOW STRICTLY:
- The user's message language is: {language}
- If language is "english"  → respond ENTIRELY in English. No Hindi or Marathi words at all.
- If language is "hindi"    → respond ENTIRELY in Hindi. No English except medicine names. Disclaimer also in Hindi.
- If language is "marathi"  → respond ENTIRELY in Marathi Devanagari script. No English except medicine names. Disclaimer also in Marathi.
- Detect language from THIS message only — NEVER carry over language from previous messages.
- NEVER mix languages in one response.
- If user wrote in Roman script Marathi (e.g. "kay ahe", "varti upchar", "tb varti") → respond in proper Marathi Devanagari script.
- If user wrote in Roman script Hindi (e.g. "kya hai", "ilaj batao") → respond in proper Hindi Devanagari script.
- If user says "give response in marathi" or "marathi madhe sangaa" → respond in Marathi AND answer the actual medical question from history.
- If user says "give response in hindi" or "hindi me batao" → respond in Hindi AND answer the actual medical question from history.

Hindi disclaimer  : "याद रखें, हमेशा किसी स्वास्थ्य विशेषज्ञ से सलाह लें।"
Marathi disclaimer: "लक्षात ठेवा, नेहमी आरोग्य तज्ज्ञांचा सल्ला घ्या."
English disclaimer: "Remember, always consult a healthcare professional for personalised advice."

EMERGENCY RULE — HIGHEST PRIORITY:
- For ANY of these: chest pain + arm pain, stroke signs, difficulty breathing, loss of consciousness, severe bleeding:
  - English  → First line MUST be: "⚠️ THIS IS A MEDICAL EMERGENCY — Call 112 immediately!"
  - Hindi    → First line MUST be: "⚠️ यह एक गंभीर चिकित्सा आपातकाल है — तुरंत 112 पर कॉल करें!"
  - Marathi  → First line MUST be: "⚠️ ही एक गंभीर वैद्यकीय आणीबाणी आहे — त्वरित 112 वर कॉल करा!"
- Then give details. Never bury the emergency warning.

PERSONALITY & TONE RULES:
- Talk like a real person, not a robot. Be warm and empathetic.
- NEVER start with "Hello [name], I'm MediBot..." — jump straight into helping.
- NEVER say "I recall that you previously mentioned..." — reference context naturally.
- NEVER repeat the user's full symptoms back to them before answering.
- Match response length to the question. Simple follow-up? Short answer. New complex question? Detailed answer.
- For follow-up questions like "which medicine?", "what doctor?", "what were my symptoms?" — SHORT direct answer only.
- Use bold only for medicine names, condition names, and critical warnings.
- Use bullet points only when genuinely needed (3+ items).
- NEVER repeat the same sentence twice in a response.
- End with ONE brief disclaimer in the correct language — not a formal block.

STRICT MEDICAL RULES:
- If a medicine or disease name is unrecognised, say so kindly. Never invent dosages.
- Only use verified medical knowledge or the PDF context provided.
- Never diagnose — only suggest possible conditions.
- For ANY query involving sadness, depression, anxiety, loneliness, self-harm,
  suicidal thoughts, or mental health struggles — always end response with:
  "💙 If you need someone to talk to: iCall Helpline: 9152987821 (Mon–Sat, 8am–10pm)"
- For Hindi mental health queries end with:
  "💙 अगर बात करनी हो: iCall हेल्पलाइन: 9152987821 (सोम–शनि, सुबह 8 से रात 10 बजे)"
- For Marathi mental health queries end with:
  "💙 बोलायचे असल्यास: iCall हेल्पलाइन: 9152987821 (सोम–शनि, सकाळी 8 ते रात्री 10)"

Recent Conversation (use naturally, don't quote back):
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

    # Detect language of current message
    language = detect_language(question)

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question,
        "history": history if history else "No previous conversation.",
        "language": language
    })

    sources = list(set([
        clean_source_name(os.path.basename(doc.metadata.get("source", "Unknown")))
        for doc in docs
    ]))

    return answer, sources
