from django.shortcuts import render
from django.http import HttpResponse
from .models import FTP_session_model

# Create your views here.

class FTP_session_view:
    def ftpp(request):
        obj = FTP_session_model()
        x = obj.FTP_On();
        return HttpResponse(x)


    def ftpp_test(request):
        obj = FTP_session_model()
        x1 = obj.FTP_On_test();
        return HttpResponse(x1)

    def test(request):
        return render(request,'home.html')
