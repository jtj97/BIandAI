# 这里我们试图将错误的验证码改为正确的
# 1.所有的l都被识别为i，还原
# 2. 部分y会被识别成v
# 3. 少数o被识别成d
# 4. 少数j被识别成i -- 解决不了不理会
# 5. 少数u被识别成j
import os
import shutil

files = os.listdir('wrong')
files = [file for file in files if list(file).count('i') == 1]
files = [file for file in files if not file.__contains__('d')]
files = [file for file in files if not file.__contains__('v')]
files = [file for file in files if len(file) == 8]
print(len(files))
print(files)
for file in files:
    shutil.copyfile(os.path.join('wrong', file),
                    os.path.join('new', file.replace('i', 'l')))
shutil.copytree('right', 'all', dirs_exist_ok=True)
shutil.copytree('new', 'all', dirs_exist_ok=True)
