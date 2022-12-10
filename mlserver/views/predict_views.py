from django.shortcuts import render




def get_predict_page(request):
    # generate random numbers
    if request.method == 'GET':
        return render(request,'predict.html')
    elif request.method == 'POST':

        return render(request, 'predict.html')