
import math
import os
# from djangoProject import settings
from ML_WebServer import settings
from django.core.mail import send_mail
import re

# if os.path.exists(temp_cache_directory + 'result.zip'):
#     print('Task run complete!' + temp_cache_directory)
#     if re.match('^.*?@.*', to_mail):
#         subject = 'Task run complete!'
#         content = '<a style="font-size: 28px; font-weight: 700; text-align: center;" href="' + endpoint + '/timecourse/analysis/omics_data_analysis_page?analysisId=' + analysis_id + '">--》Click to see the results《--</a>'
#         from_mail = 'Task run notification<' + settings.EMAIL_HOST_USER + '>'
#         print('发送邮件' + content)
#         try:
#             send_mail(subject, message=None, from_email=from_mail, recipient_list=[to_mail], fail_silently=False,
#                       html_message=content)
#             print('Sending an email succeeded！')
#         except Exception as ee:
#             print('发送邮件异常')
#             print(ee)


subject = 'Task run complete!'
content = '<a style="font-size: 28px; font-weight: 700; text-align: center;" href="'+ '/timecourse/analysis/omics_data_analysis_page?analysisId=' + '">--》Click to see the results《--</a>'
# from_mail = 'Task run notification<' + '>'
from_mail = 'linzhewei1999@163.com'
to_mail = '1198369937@qq.com'
print('发送邮件' + content)
send_mail(subject, message=None, from_email=from_mail, recipient_list=[to_mail], fail_silently=False,
              html_message=content)
print('Sending an email succeeded！')

# try:
#
# except Exception as ee:
#     print('发送邮件异常')