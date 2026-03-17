from django.contrib import admin
from .models import ChatHistory

@admin.register(ChatHistory)
class ChatHistoryAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'short_question', 'sources', 'created_at']
    list_filter = ['user', 'created_at', 'sources']
    search_fields = ['question', 'answer', 'user__username']
    readonly_fields = ['created_at']
    ordering = ['-created_at']
    list_per_page = 25

    def short_question(self, obj):
        return obj.question[:70] + '...' if len(obj.question) > 70 else obj.question
    short_question.short_description = 'Question'