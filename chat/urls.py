from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('ask/', views.ask, name='ask'),
    path('upload-report/', views.upload_report, name='upload_report'),
    path('clear-history/', views.clear_history, name='clear_history'),
    path('profile/', views.profile, name='profile'),
    path('analytics/', views.analytics, name='analytics'),
    path('contact/', views.contact, name='contact'),
    path('password-reset/', views.password_reset_request, name='password_reset'),
    path('password-reset-confirm/<uidb64>/<token>/',
         __import__('django.contrib.auth.views', fromlist=['PasswordResetConfirmView']).PasswordResetConfirmView.as_view(
             template_name='password_reset_confirm.html',
             success_url='/login/'
         ), name='password_reset_confirm'),
]