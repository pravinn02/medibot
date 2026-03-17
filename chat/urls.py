from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('ask/', views.ask, name='ask'),
    path('upload-report/', views.upload_report, name='upload_report'),
    path('clear-history/', views.clear_history, name='clear_history'),
]
