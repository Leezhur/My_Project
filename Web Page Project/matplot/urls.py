from django.contrib import admin
from django.urls import path, include
from . import views


urlpatterns = [
    path('post/', views.post, name="post"),
    path('board/', views.board, name='board'),
    path('edit/<int:pk>', views.boardEdit, name='edit'),
    path('delete/<int:pk>', views.boardDelete, name='delete'),
]
