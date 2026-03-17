import json
import os
import fitz
import pytesseract
from PIL import Image
import io
from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.core.mail import send_mail
from django.conf import settings as django_settings
from .rag import ask_medibot, llm
from .models import ChatHistory

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# ── Register Form ──────────────────────────────────────────────
class RegisterForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'placeholder': 'Enter your email'})
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user


# ── Helper Functions ───────────────────────────────────────────
def extract_text_from_pdf(file):
    text = ''
    pdf = fitz.open(stream=file.read(), filetype='pdf')
    for page in pdf:
        text += page.get_text()
    return text.strip()


def extract_text_from_image(file):
    image = Image.open(io.BytesIO(file.read()))
    text = pytesseract.image_to_string(image)
    return text.strip()


def summarize_report(text):
    from langchain_core.messages import HumanMessage
    prompt = f'''You are MediBot, a medical assistant.
A user has uploaded their medical report. Read it carefully and summarize it in very simple language that a non-medical person can understand.

Include:
1. What the report is about
2. Key findings in simple words
3. Any abnormal values and what they mean
4. What the user should do next

Medical Report:
{text[:3000]}

Simple Summary:'''
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ── Views ──────────────────────────────────────────────────────
@login_required
def index(request):
    history = ChatHistory.objects.filter(user=request.user).order_by('-created_at')[:20]
    return render(request, 'index.html', {'history': history})


@csrf_exempt
@login_required
def ask(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        question = data.get('question', '')
        if not question:
            return JsonResponse({'error': 'No question'}, status=400)
        answer, sources = ask_medibot(question)
        ChatHistory.objects.create(
            user=request.user,
            question=question,
            answer=answer,
            sources=', '.join(sources)
        )
        return JsonResponse({'answer': answer, 'sources': sources})
    return JsonResponse({'error': 'POST only'}, status=405)


@csrf_exempt
@login_required
def upload_report(request):
    if request.method == 'POST':
        file = request.FILES.get('report')
        if not file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        filename = file.name.lower()
        try:
            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(file)
            elif filename.endswith(('.jpg', '.jpeg', '.png')):
                text = extract_text_from_image(file)
            else:
                return JsonResponse({'error': 'Only PDF and images supported'}, status=400)
            if not text or len(text) < 50:
                return JsonResponse({'error': 'Could not extract text. Please try a clearer image or text-based PDF.'}, status=400)
            summary = summarize_report(text)
            ChatHistory.objects.create(
                user=request.user,
                question='[Report Upload] ' + file.name,
                answer=summary,
                sources='Uploaded Report'
            )
            return JsonResponse({'summary': summary})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'POST only'}, status=405)


@csrf_exempt
@login_required
def clear_history(request):
    if request.method == 'POST':
        ChatHistory.objects.filter(user=request.user).delete()
        return JsonResponse({'status': 'cleared'})
    return JsonResponse({'error': 'POST only'}, status=405)


def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Fix: specify backend explicitly to avoid multiple backends error
            login(request, user, backend='django.contrib.auth.backends.ModelBackend')

            # Send welcome email
            try:
                send_mail(
                    subject='Welcome to MediBot AI 💊',
                    message=f'''Hi {user.username},

Welcome to MediBot — your AI-powered medical assistant!

You can now:
- Ask any medical question
- Get symptom-based medicine suggestions
- Upload medical reports for simple summaries

Visit: http://127.0.0.1:8000

Stay healthy!
— MediBot AI Team''',
                    from_email=django_settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[user.email],
                    fail_silently=True,
                )
            except Exception:
                pass  # Don't break registration if email fails

            return redirect('/')
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})