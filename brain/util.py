import io
import os
import string
from http.cookiejar import Cookie, CookieJar
from io import BytesIO
from time import sleep
from urllib import request
import json
from typing import Callable

from PIL import Image

import numpy as np
from torchvision.transforms import transforms


def is_element_present(driver, by, value, plural=False):
    from selenium.common.exceptions import NoSuchElementException
    try:
        if not plural:
            element = driver.find_element(by=by, value=value)
        else:
            element = driver.find_elements(by=by, value=value)
    except NoSuchElementException:
        return None
    else:
        return element


def initial(username: str, password: str):
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    edriver = webdriver.Chrome()
    # 预先真实登录
    first_url = 'http://gradinfo.sustech.edu.cn/ssfw/login_cas.jsp'

    while True:
        edriver.get(first_url)
        # 探测输入框
        if not is_element_present(edriver, By.ID, 'cboxOverlay'):
            edriver.find_element_by_id('username').send_keys(username)
            edriver.find_element_by_id('password').send_keys(password)
            edriver.find_element_by_name('submit').click()
        sleep(5)
        # 探测[你没有访问该页面的权限.]
        ret = is_element_present(edriver, By.LINK_TEXT, '返回首页')
        if not ret:
            break

    # 抽出cookies
    cookies = edriver.get_cookies()
    edriver.close()
    jar = CookieJar()
    # Cookie(version, name, value, port, port_specified, domain,
    # domain_specified, domain_initial_dot, path, path_specified,
    # secure, discard, comment, comment_url, rest)
    for cookie in cookies:
        c = Cookie(0, cookie['name'], cookie['value'], '80', '80', cookie['domain'],
                   None, None, cookie['path'], None,
                   cookie['secure'], False, '', None, None, None)
        jar.set_cookie(c)
    return jar


def fetch_and_judge(jar, predict: Callable[[bytes, ], str]):
    import muggle_ocr
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"}

    captcha_url = 'http://gradinfo.sustech.edu.cn/ssfw/captcha.do'
    check_url = 'http://gradinfo.sustech.edu.cn/ssfw/pygl/xkgl/wsxk/chooseCourse.do?' \
                'bjdm=BJ201802_81879&lcid=9&validateCode={:s}'

    # 初始化
    handler = request.HTTPCookieProcessor(jar)
    opener = request.build_opener(handler)
    sdk = muggle_ocr.SDK(model_type=muggle_ocr.ModelType.Captcha)

    # 获得验证码
    req = request.Request(captcha_url, headers=headers, method='GET')
    rsp = opener.open(req)
    raw_stream = rsp.read()

    # 尝试预测
    ans = predict(raw_stream)

    # 检验预测结果
    req = request.Request(check_url.format(ans), headers=headers, method='GET')
    rsp = opener.open(req)
    ret_raw = rsp.read()
    dic = json.loads(ret_raw)

    if dic['errorMessages'] == '验证码不正确':
        return False
    elif dic['errorMessages'] == 'java.lang.String incompatible with com.wisedu.ssfw.security.SsfwUserDetails':
        return True


def fetch_and_judge_times(jar, predict: Callable[[bytes, ], str], times: int = 100):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"}

    captcha_url = 'http://gradinfo.sustech.edu.cn/ssfw/captcha.do'
    check_url = 'http://gradinfo.sustech.edu.cn/ssfw/pygl/xkgl/wsxk/chooseCourse.do?' \
                'bjdm=BJ201802_81879&lcid=9&validateCode={:s}'

    # 初始化
    handler = request.HTTPCookieProcessor(jar)
    opener = request.build_opener(handler)

    right_times = 0
    wrong_times = 0
    for i in range(times):
        # 获得验证码
        req = request.Request(captcha_url, headers=headers, method='GET')
        rsp = opener.open(req)
        raw_stream = rsp.read()

        # 尝试预测
        ans = predict(raw_stream)

        # 检验预测结果
        req = request.Request(check_url.format(ans), headers=headers, method='GET')
        rsp = opener.open(req)
        ret_raw = rsp.read()
        dic = json.loads(ret_raw)

        if dic['errorMessages'] == '验证码不正确':
            wrong_times += 1
        elif dic['errorMessages'] == 'java.lang.String incompatible with com.wisedu.ssfw.security.SsfwUserDetails':
            right_times += 1
    return right_times, wrong_times


def fetch_train_data(jar, size: int):
    import muggle_ocr
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"}

    captcha_url = 'http://gradinfo.sustech.edu.cn/ssfw/captcha.do'
    check_url = 'http://gradinfo.sustech.edu.cn/ssfw/pygl/xkgl/wsxk/chooseCourse.do?' \
                'bjdm=BJ201802_81879&lcid=9&validateCode={:s}'

    # 建立存储文件夹
    for dir_name in ['right', 'wrong']:
        if dir_name not in os.listdir('.'):
            os.mkdir(dir_name)

    # 初始化
    handler = request.HTTPCookieProcessor(jar)
    opener = request.build_opener(handler)
    sdk = muggle_ocr.SDK(model_type=muggle_ocr.ModelType.Captcha)

    while len(os.listdir('right')) < size:
        # 获得验证码
        req = request.Request(captcha_url, headers=headers, method='GET')
        rsp = opener.open(req)
        raw_stream = rsp.read()

        # 尝试预测
        ans = sdk.predict(raw_stream)

        # 验证码转换成PNG格式准备存储
        bytes_stream = BytesIO(raw_stream)
        captcha = Image.open(bytes_stream)
        # captcha.show()
        imgByteArr = BytesIO()
        captcha.save(imgByteArr, format('PNG'))
        imgByteArr = imgByteArr.getvalue()

        # 检验预测结果
        req = request.Request(check_url.format(ans), headers=headers, method='GET')
        rsp = opener.open(req)
        ret_raw = rsp.read()
        dic = json.loads(ret_raw)

        if dic['errorMessages'] == '验证码不正确':
            save_predix = 'wrong'
        elif dic['errorMessages'] == 'java.lang.String incompatible with com.wisedu.ssfw.security.SsfwUserDetails':
            save_predix = 'right'

        # 存储图片
        with open(os.path.join(save_predix, '{:s}.png'.format(ans)), 'wb') as f:
            f.write(imgByteArr)


# 训练过程使用的计算准确度
characters = '-' + string.ascii_lowercase


def decode(sequence):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j + 1]])
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s


def decode_target(sequence):
    return ''.join([characters[x] for x in sequence]).replace(' ', '')


def calc_acc(target, output, use_cuda):
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    if use_cuda:
        target = target.cpu().numpy()
        output_argmax = output_argmax.cpu().numpy()
    a = np.array([decode_target(true) == decode(pred) for true, pred in zip(target, output_argmax)])
    return a.mean()


# 包装一个predict接口用于实际测试
def use_predict(model):
    transformer = transforms.ToTensor()

    def predict(raw: bytes):
        stream = io.BytesIO(raw)
        img = Image.open(stream)
        img = img.resize([192, 64], Image.ANTIALIAS)
        tensor = transformer(img).unsqueeze(0)
        ans = model(tensor)
        ans_argmax = ans.permute(1, 0, 2).argmax(dim=-1)
        label = decode(ans_argmax.squeeze())
        return label

    return predict
