import math
import os
import shutil
import pickle
import pandas as pd
import json
from django.http import FileResponse, HttpResponse, Http404, StreamingHttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from concurrent.futures.thread import ThreadPoolExecutor
import uuid
import numpy as np
import time
from django.core.mail import send_mail
import re
import chardet
from zipfile import ZipFile

from djangoProject import settings
# 定义全局线程池
global_thread_pool = ThreadPoolExecutor(100)


