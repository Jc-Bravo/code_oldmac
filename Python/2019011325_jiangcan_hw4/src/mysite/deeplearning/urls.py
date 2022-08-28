from django.urls import path

from . import views

app_name = 'deeplearning'
urlpatterns = [
    path('index/', views.index, name='index'),
    path('<int:project_id>/', views.detail, name='detail'),
    path('results/', views.results, name='results'),
]