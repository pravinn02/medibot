from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from chat import views as chat_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/login/'), name='logout'),
    path('register/', chat_views.register, name='register'),
    path('', include('chat.urls')),
]