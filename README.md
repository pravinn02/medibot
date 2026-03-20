\# 🩺 MediBot — AI-Powered Medical Assistant



<div align="center">



!\[Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge\&logo=python)

!\[Django](https://img.shields.io/badge/Django-4.2-green?style=for-the-badge\&logo=django)

!\[LangChain](https://img.shields.io/badge/LangChain-RAG-orange?style=for-the-badge)

!\[FAISS](https://img.shields.io/badge/FAISS-Vector\_Store-purple?style=for-the-badge)

!\[Groq](https://img.shields.io/badge/Groq-Llama\_3.3\_70B-red?style=for-the-badge)

!\[License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)



\*\*A domain-specific medical chatbot powered by Retrieval-Augmented Generation (RAG)\*\*  

Grounded in verified clinical sources — Oxford Handbook, MSF Guidelines \& WHO Medicines



\[Features](#-features) • \[Architecture](#-rag-pipeline) • \[Setup](#-setup) • \[Evaluation](#-evaluation-results) • \[Screenshots](#-screenshots)



</div>



\---



\## 🔥 Why MediBot?



Unlike generic AI chatbots that hallucinate medical facts, MediBot retrieves answers directly from verified clinical PDFs before generating a response. Every answer is grounded in real medical knowledge.



| Generic Chatbot | MediBot |

|---|---|

| Makes up medicine dosages | Retrieves from Oxford Handbook / MSF Guidelines |

| No source citations | Shows which medical source was used |

| Forgets context | Remembers last 5 messages naturally |

| Accepts fake medicine names | Rejects unrecognised medicines |

| English only | Supports Hindi + English |



\---



\## ✨ Features



\- 🔍 \*\*RAG Pipeline\*\* — PDF ingestion → chunking → embeddings → FAISS retrieval → LLM generation

\- 🛡️ \*\*Hallucination Guard\*\* — Detects and rejects fake/unrecognised medicine names

\- 🩺 \*\*Symptom Checker\*\* — Possible conditions, recommended specialist, home care, red flags

\- 💊 \*\*Medicine Info\*\* — Dosage, side effects, precautions, alternatives

\- 📋 \*\*Report Upload\*\* — PDF/image medical reports summarised in plain language via OCR

\- 💬 \*\*Conversation Memory\*\* — Natural multi-turn chat with context from last 5 messages

\- 🇮🇳 \*\*Hindi Support\*\* — Responds in Hindi when user writes in Hindi

\- 📊 \*\*Admin Analytics\*\* — Real-time dashboard with user activity, charts, top users

\- 🔒 \*\*Production Ready\*\* — Rate limiting, custom error pages, email auth, HTTPS-ready



\---



\## 🏗️ RAG Pipeline



```

Medical PDFs

&#x20;   │

&#x20;   ▼

PDF Ingestion (PyMuPDF)

&#x20;   │

&#x20;   ▼

Text Chunking (LangChain RecursiveCharacterTextSplitter)

&#x20;   │

&#x20;   ▼

Embeddings (sentence-transformers/all-MiniLM-L6-v2 → 384-dim vectors)

&#x20;   │

&#x20;   ▼

FAISS Vector Store (similarity search, k=3)

&#x20;   │

&#x20;   ▼

LLM Generation (Llama 3.3 70B via Groq API)

&#x20;   │

&#x20;   ▼

Grounded Medical Answer

```



\*\*Knowledge Base:\*\*

\- 📖 Oxford Handbook of Clinical Medicine (10th Edition)

\- 🏥 MSF Clinical Guidelines — Diagnosis \& Treatment

\- 💊 WHO Essential Medicines List 2023



\---



\## 📊 Evaluation Results



Evaluated using a custom retrieval evaluation notebook (`eval.ipynb`):



| Metric | Result |

|---|---|

| Overall Hit Rate | \*\*100%\*\* (10/10 questions) |

| Optimised Retriever | \*\*k=3\*\* (matches k=9 quality) |

| Source Distribution | MSF 51.7% · Oxford 45% · WHO 3.3% |



\*\*Key findings:\*\*

\- k=3 retrieval matches k=9 hit rate — optimised for speed with no quality loss

\- Clinical PDFs use abbreviations (CXR, RIF, GI) — real clinical terminology, not layman language

\- All 10 medical test questions retrieved at least one relevant chunk



\---



\## 🛠️ Tech Stack



| Layer | Technology |

|---|---|

| \*\*LLM\*\* | Llama 3.3 70B via Groq API |

| \*\*Embeddings\*\* | sentence-transformers/all-MiniLM-L6-v2 |

| \*\*Vector Store\*\* | FAISS |

| \*\*RAG Framework\*\* | LangChain |

| \*\*Backend\*\* | Django 4.2 |

| \*\*Database\*\* | MySQL (local) / PostgreSQL (production) |

| \*\*OCR\*\* | Pytesseract + PyMuPDF |

| \*\*Deployment\*\* | Railway-ready (Gunicorn + WhiteNoise) |



\---



\## ⚙️ Setup



\### Prerequisites

\- Python 3.10+

\- MySQL

\- Tesseract OCR (\[download](https://github.com/UB-Mannheim/tesseract/wiki))

\- Groq API key (\[get free key](https://console.groq.com))



\### 1. Clone the repo

```bash

git clone https://github.com/pravinn02/medibot.git

cd medibot

```



\### 2. Create virtual environment

```bash

python -m venv env

\# Windows

.\\env\\Scripts\\activate

\# Mac/Linux

source env/bin/activate

```



\### 3. Install dependencies

```bash

pip install -r requirements.txt

```



\### 4. Create `.env` file

```env

SECRET\_KEY=your-secret-key

DEBUG=True

GROQ\_API\_KEY=your-groq-api-key

EMAIL\_HOST\_USER=your-gmail@gmail.com

EMAIL\_HOST\_PASSWORD=your-gmail-app-password

DB\_PASSWORD=your-mysql-password

```



\### 5. Setup database

```bash

\# Create MySQL database named 'medibot\_db'

python manage.py migrate

python manage.py createsuperuser

```



\### 6. Build FAISS index

```bash

\# Add your medical PDFs to medical\_docs/ folder

python rebuild\_index.py

```



\### 7. Run the server

```bash

python manage.py runserver

```



Visit `http://127.0.0.1:8000` 🚀



\---



\## 📁 Project Structure



```

medibot/

├── chat/

│   ├── models.py          # ChatHistory model

│   ├── views.py           # All views + routing logic

│   ├── rag.py             # RAG pipeline (embeddings, retriever, LLM)

│   └── urls.py

├── medibot/

│   ├── settings.py        # Django settings

│   └── urls.py

├── templates/             # HTML templates

├── faiss\_index/           # FAISS vector store

├── medical\_docs/          # Source medical PDFs

├── eval.ipynb             # Retrieval evaluation notebook

├── rebuild\_index.py       # Script to rebuild FAISS index

├── Procfile               # Railway deployment

├── requirements.txt

└── .env

```



\---



\## 🧪 Testing MediBot



Try these queries to test different features:



```

\# Symptom checker

I have fever and headache since 2 days



\# Direct medicine question

which tablet should I take for headache?



\# Safety question

is paracetamol safe during pregnancy?



\# Hallucination test (should reject)

what is the dosage of Healocin 500mg?



\# Hindi support

मुझे बुखार है, कौन सी दवा लूं?



\# Report upload

Upload any PDF/image medical report

```



\---



\## 🔒 Security Features



\- ✅ Rate limiting (10 requests/minute per user)

\- ✅ Input length validation (500 char limit)

\- ✅ Custom 404 \& 500 error pages

\- ✅ HTTPS-ready security headers

\- ✅ Environment variable based config

\- ✅ CSRF protection



\---



\## 🚀 Future Improvements



\- \[ ] Semantic chunking (replace fixed-size chunking)

\- \[ ] Reranker model (cross-encoder/ms-marco-MiniLM-L-6-v2)

\- \[ ] RAGAS evaluation (end-to-end answer quality)

\- \[ ] Streaming responses (token by token)

\- \[ ] More medical PDFs (Harrison's, BNF, NICE Guidelines)



\---



\## 👨‍💻 Author



\*\*Pravin Landage\*\*  

AI/ML Portfolio Project



\[!\[GitHub](https://img.shields.io/badge/GitHub-pravinn02-black?style=flat\&logo=github)](https://github.com/pravinn02)



\---



\## ⚠️ Disclaimer



MediBot is an AI assistant for informational purposes only. It is \*\*not a substitute for professional medical advice\*\*. Always consult a qualified doctor for medical decisions.



\---



<div align="center">

⭐ Star this repo if you found it useful!

</div>

