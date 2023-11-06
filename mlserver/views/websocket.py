


# #Django websocket 长连接使用
# #pip install dwebsocket
#
# #向前端发送数据
# import time
# import json
# from dwebsocket.decorators import accept_websocket
# @accept_websocket
# def test_websocket(request):
#     if request.is_websocket():
#         while 1:
#             time.sleep(1) ## 向前端发送时间
#             dit = {
#                 'time':time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time()))
#             }
#             request.websocket.send(json.dumps(dit))
#
# #接受前端信息
# @accept_websocket
# def test_socket(request):
#     if request.is_websocket():
#         for message in request.websocket:
#             print(message)
#             request.websocket.send(message)

from django.shortcuts import render
import json
import time
from dwebsocket.decorators import accept_websocket,require_websocket
import re
from django.http import HttpResponse
import dwebsocket.websocket
# Create your views here.


@accept_websocket
def test_websocket2(request):
    '''服务端视图'''
    print('request: ',request)
    print('request.is_websocket(): ',request.is_websocket())
    connect_num = 0
    if request.is_websocket(): # 如果请求是websocket请求：WebSocket = request.websocket
        WebSocket = request.websocket
        while True:
            # 判断是否通过websocket接收到数据
            if WebSocket.has_messages():
                # 接收Websocket客户端发送过来的消息
                client_msg = WebSocket.read().decode("utf-8")
                print(client_msg)
                # 设置返回前端的数据
                res = re.sub("吗?([？?])", "!", client_msg)
                if connect_num < 3:
                    messages = {
                        'time': time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())),
                        'status': 0,
                        # 'server_msg': res,
                        # 'client_msg': client_msg
                    }
                else:
                    messages = {
                        'time': time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())),
                        'status': 1,
                        # 'server_msg': res
                    }
                connect_num = connect_num + 1
                request.websocket.send(json.dumps(messages))
    else:
        return HttpResponse('请使用 WebSocket 连接')
        pass

        # while 1:
        #     time.sleep(1)  ## 向前端发送时间
        #     dit = {
        #         'time': time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time()))
        #     }
        #     request.websocket.send(json.dumps(dit))

def test_websocket_client(request):
    '''客户端视图'''
    return render(request,'websocket_client.html')

# def vue_socket(request):
#     return render(request, 'vue_analysis.html')

def get_vue_analysis_page(request):
    return render(request, 'vue_analysis.html')