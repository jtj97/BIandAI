import torch

from util import initial, fetch_train_data, fetch_and_judge, fetch_and_judge_times, use_predict

# 环境需求
# Python=3.7
# pip install muggle_ocr==1.0.3
# pip install Pillow==7.2.0
# pip install selenium==3.141.0
# pip install pytorch
# pip install torchvision

# 安装chrome驱动
# 首先你需要有chrome
# 右上角三个点-帮助-关于-查看chrome版本
# http://npm.taobao.org/mirrors/chromedriver/
# 从以上链接找到对应自己版本的chrome驱动
# 下载解压
# 将该文件路径加入环境变量

# 使用时在status.py里去改成自己教务系统的用户名和密码
from status import username, password

# 初始化，不要改，就这样写，只需要使用一次
jar = initial(username, password)

# 如果需要收集数据用下面这条命令
fetch_train_data(jar, 10000)

# 如果需要测试写好的识别器用下面这条命令
# 第二个参数改为自己写的判别函数
# 以回调函数方式被使用
# import muggle_ocr
# sdk = muggle_ocr.SDK(model_type=muggle_ocr.ModelType.Captcha)
# ans = fetch_and_judge(jar, sdk.predict)
# print(ans)

# 如果需要测试写好的识别器准确率用下面这条命令
# 第一个参数不要改
# 第二个参数改为自己写的判别函数
# 以回调函数方式被使用
# 第三个参数为测试次数

# sdk = muggle_ocr.SDK(model_type=muggle_ocr.ModelType.Captcha)
# right_times, wrong_times = fetch_and_judge_times(jar, sdk.predict, 10)
# print(right_times / (right_times + wrong_times))

# 如果需要连接自己训练的识别器用下面这条命令

# use_cuda = torch.cuda.is_available()
# if use_cuda:
#     torch.cuda.set_device(1)
#     model = torch.load('ctc.pth')
# else:
#     model = torch.load('ctc.pth', map_location=torch.device('cpu'))
#
# model.eval()
# predict = use_predict(model)
# print(fetch_and_judge(jar, predict))
