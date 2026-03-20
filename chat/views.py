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
from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import Count
from django.db.models.functions import TruncDate
from django.core.cache import cache
import json as json_lib
from .rag import ask_medibot, llm
from .models import ChatHistory

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# ── Trigger Words ──────────────────────────────────────────────
GREETINGS = ['hi', 'hello', 'hey', 'good morning', 'good evening',
             'good afternoon', 'namaste', 'hii', 'helo', 'howdy',
             'sup', 'greetings', 'good night']

DIRECT_MEDICINE_TRIGGERS = [
    'which tablet', 'which medicine', 'which drug',
    'what tablet', 'what medicine', 'suggest tablet',
    'suggest medicine', 'recommend tablet', 'recommend medicine',
    'which painkiller', 'what should i take', 'what can i take',
    'which pill', 'what pill',
]

SYMPTOM_TRIGGERS = ['i have', 'i am having', 'i feel', 'i am feeling',
                    'suffering from', 'experiencing', 'my symptoms are',
                    'symptoms:', 'i got', 'having', 'mujhe', 'mera',
                    'i am suffering', "i've been"]

MEDICINE_TRIGGERS = ['tablet', 'medicine', 'drug', 'capsule', 'syrup',
                     'dose', 'dosage', 'injection', 'cream', 'ointment',
                     'paracetamol', 'ibuprofen', 'aspirin', 'amoxicillin',
                     'how to take', 'side effects of', 'uses of']

SAFETY_TRIGGERS = ['is safe', 'is it safe', 'safe to take', 'safe during',
                   'is dangerous', 'is it dangerous', 'can i take',
                   'can we take', 'safe for', 'okay to take', 'ok to take',
                   'safe in pregnancy', 'safe for kids', 'safe for children']


# ── Rate Limiting ──────────────────────────────────────────────
def is_rate_limited(user_id):
    key = f"rate_limit_{user_id}"
    requests = cache.get(key, 0)
    if requests >= 10:
        return True
    cache.set(key, requests + 1, timeout=60)
    return False


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
    prompt = f'''You are MediBot, a warm and friendly medical assistant.
A user has uploaded their medical report. Read it carefully and summarize it in very simple language that a non-medical person can understand.

Include:
1. What the report is about
2. Key findings in simple words
3. Any abnormal values and what they mean
4. What the user should do next

Be conversational and reassuring, not clinical.

Medical Report:
{text[:3000]}

Simple Summary:'''
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def send_welcome_email(user):
    try:
        send_mail(
            subject='Welcome to MediBot AI 💊',
            message=f'''Hi {user.username}!

Welcome to MediBot — your AI-powered medical assistant! 🎉

Your account has been created successfully.

📧 Email    : {user.email}
👤 Username : {user.username}

You can now:
✅ Ask any medical question
✅ Check symptoms and get possible conditions
✅ Get medicine dosage and side effect info
✅ Upload medical reports for a simple summary

👉 Login here: http://127.0.0.1:8000/login/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stay healthy and take care!
— MediBot AI Team 💊
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is an automated message. Please do not reply.''',
            from_email=django_settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            fail_silently=False,
        )
        return True
    except Exception as e:
        print(f"[MediBot] Email failed: {e}")
        return False


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
        question = data.get('question', '').strip()

        if not question:
            return JsonResponse({'error': 'No question'}, status=400)

        if len(question) > 500:
            return JsonResponse({'error': 'Question too long. Please keep it under 500 characters.'}, status=400)

        if is_rate_limited(request.user.id):
            return JsonResponse({'error': '⚠️ Too many requests. Please wait a minute before asking again.'}, status=429)

        q_lower = question.lower()

        # ── Build conversation history (last 5 messages) ──
        recent_history = ChatHistory.objects.filter(
            user=request.user
        ).order_by('-created_at')[:5]

        history_text = ""
        for item in reversed(recent_history):
            history_text += f"User: {item.question}\nMediBot: {item.answer[:300]}...\n\n"

        # ── Greeting ──
        if q_lower in GREETINGS:
            answer = (
                f"Hey {request.user.username}! 👋 Great to see you.\n\n"
                f"I'm here to help with anything medical — symptoms, treatments, medicines, or just making sense of a report you got. What's going on?"
            )
            sources = []

        # ── Direct Medicine Request ──
        elif any(trigger in q_lower for trigger in DIRECT_MEDICINE_TRIGGERS):
            answer, sources = ask_medibot(
                f"{question}\n\n"
                f"The user is asking directly for a medicine recommendation. "
                f"Give a SHORT, friendly, direct answer (3-5 sentences max):\n"
                f"1. Name the most common OTC medicine for this (e.g. Paracetamol, Ibuprofen)\n"
                f"2. Basic dosage for adults\n"
                f"3. One key caution\n"
                f"4. When to see a doctor instead\n"
                f"No full diagnosis. No list of possible conditions. Just answer the question asked.",
                history=history_text
            )

        # ── Safety Question ──
        elif any(trigger in q_lower for trigger in SAFETY_TRIGGERS):
            answer, sources = ask_medibot(
                f"{question}\n\n"
                f"Give a short, conversational answer (2-4 sentences max). "
                f"Is it safe or not? Any key warnings? When to consult a doctor? "
                f"No need for full medicine breakdown — just a friendly direct answer.",
                history=history_text
            )

        # ── Symptom Checker ──
        elif any(trigger in q_lower for trigger in SYMPTOM_TRIGGERS):
            answer, sources = ask_medibot(
                f"Patient reports: {question}\n\n"
                f"As MediBot, analyze these symptoms carefully and provide:\n"
                f"1. **Possible Conditions** — List 3-5 possible diseases or conditions that match these symptoms\n"
                f"2. **Why These Match** — For each condition, briefly explain why the symptoms match\n"
                f"3. **Recommended Doctor** — What type of specialist should the patient see\n"
                f"4. **Immediate Home Care** — Safe home remedies or actions to take right now\n"
                f"5. **Red Flag Warnings** — Any symptoms that would require emergency care\n\n"
                f"Use simple language. Always recommend consulting a doctor.",
                history=history_text
            )

        # ── Medicine Info ──
        elif any(trigger in q_lower for trigger in MEDICINE_TRIGGERS):
            answer, sources = ask_medibot(
                f"{question}\n\n"
                f"Provide complete medicine information including:\n"
                f"1. **What it is used for** — Main uses and conditions it treats\n"
                f"2. **Dosage** — Standard adult and child doses\n"
                f"3. **How to take** — With or without food, timing\n"
                f"4. **Side Effects** — Common and serious side effects\n"
                f"5. **Precautions** — Who should avoid it, drug interactions\n"
                f"6. **Alternatives** — Similar medicines if available",
                history=history_text
            )

        # ── General Medical Question ──
        else:
            answer, sources = ask_medibot(question, history=history_text)

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
            login(request, user, backend='django.contrib.auth.backends.ModelBackend')
            email_sent = send_welcome_email(user)
            if email_sent:
                print(f"[MediBot] Welcome email sent to {user.email}")
            else:
                print(f"[MediBot] Failed to send email to {user.email}")
            return redirect('/')
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})


@login_required
def profile(request):
    total_questions = ChatHistory.objects.filter(
        user=request.user
    ).exclude(question__startswith='[Report Upload]').count()
    total_reports = ChatHistory.objects.filter(
        user=request.user,
        question__startswith='[Report Upload]'
    ).count()
    recent = ChatHistory.objects.filter(
        user=request.user
    ).order_by('-created_at')[:5]
    return render(request, 'profile.html', {
        'total_questions': total_questions,
        'total_reports': total_reports,
        'recent': recent,
    })


@staff_member_required
def analytics(request):
    total_users = User.objects.count()
    total_questions = ChatHistory.objects.exclude(question__startswith='[Report Upload]').count()
    total_reports = ChatHistory.objects.filter(question__startswith='[Report Upload]').count()
    total_interactions = ChatHistory.objects.count()

    top_users = (
        ChatHistory.objects.values('user__username')
        .annotate(count=Count('id'))
        .order_by('-count')[:5]
    )

    from datetime import date, timedelta
    today = date.today()
    days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    daily_counts = (
        ChatHistory.objects
        .filter(created_at__date__gte=today - timedelta(days=6))
        .annotate(day=TruncDate('created_at'))
        .values('day')
        .annotate(count=Count('id'))
        .order_by('day')
    )
    day_map = {str(d['day']): d['count'] for d in daily_counts}
    chart_labels = [d.strftime('%b %d') for d in days]
    chart_data = [day_map.get(str(d), 0) for d in days]

    recent = ChatHistory.objects.select_related('user').order_by('-created_at')[:10]

    return render(request, 'analytics.html', {
        'total_users': total_users,
        'total_questions': total_questions,
        'total_reports': total_reports,
        'total_interactions': total_interactions,
        'top_users': top_users,
        'chart_labels': json_lib.dumps(chart_labels),
        'chart_data': json_lib.dumps(chart_data),
        'recent': recent,
    })


@csrf_exempt
def password_reset_request(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data.get('email', '').strip()

            if not email:
                return JsonResponse({'success': False, 'error': 'Email is required.'})

            try:
                user = User.objects.get(email=email)
            except User.DoesNotExist:
                return JsonResponse({'success': False, 'error': 'No account found with this email.'})

            from django.contrib.auth.tokens import default_token_generator
            from django.utils.http import urlsafe_base64_encode
            from django.utils.encoding import force_bytes

            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = default_token_generator.make_token(user)
            reset_link = f"http://127.0.0.1:8000/password-reset-confirm/{uid}/{token}/"

            send_mail(
                subject='MediBot — Password Reset Request 🔑',
                message=f'''Hi {user.username},

We received a request to reset your MediBot password.

Click the link below to reset your password:
👉 {reset_link}

This link expires in 24 hours.

If you did not request this, please ignore this email.

— MediBot AI Team 💊''',
                from_email=django_settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                fail_silently=False,
            )
            return JsonResponse({'success': True})

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'error': 'POST only'}, status=405)


@login_required
def contact(request):
    if request.method == 'POST':
        subject = request.POST.get('subject', '').strip()
        message = request.POST.get('message', '').strip()

        if not subject or not message:
            return render(request, 'contact.html', {'error': 'Please fill in all fields.'})

        try:
            send_mail(
                subject=f'[MediBot Contact] {subject}',
                message=f'''New message from MediBot user:

👤 Username : {request.user.username}
📧 Email    : {request.user.email}
📝 Subject  : {subject}

Message:
{message}

— Sent from MediBot Contact Form''',
                from_email=django_settings.DEFAULT_FROM_EMAIL,
                recipient_list=['landagepravin505@gmail.com'],
                fail_silently=False,
            )
            return render(request, 'contact.html', {'success': True})
        except Exception as e:
            return render(request, 'contact.html', {'error': str(e)})

    return render(request, 'contact.html')


def error_404(request, exception):
    return render(request, '404.html', status=404)


def error_500(request):
    return render(request, '500.html', status=500)