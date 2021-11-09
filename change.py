import os
import sys
from itertools import groupby
if __name__ == "__main__":

    folder_name = "/home/lihang/ecg/data/multi_class_good/"  # 获取文件夹的名字，即路径
    #folder_name = "C:\\Users\\HP\\Desktop\\pretreatment\\pretreatment\\change\\"
    file_names = os.listdir(folder_name)  # 获取文件夹内所有文件的名字

    for name in file_names:  # 如果某个文件名在file_names内
        print(name)
        s = [''.join(list(g)) for k, g in groupby(name, key=lambda x: x.isdigit())]
        word1 = s[0]
        number = s[1]
        word2 = s[2]
        word3 = word2.strip('-')
        print(word1)
        print(number)
        print(word2)

        old_name = folder_name + '/' + name  # 获取旧文件的名字，注意名字要带路径名
        print(old_name)
        new_name = folder_name + word1 + '-' + number + ' '+ word3  # 定义新文件的名字，这里给每个文件名前加了前缀 a_
        print(new_name)
        os.rename(old_name, new_name)  # 用rename()函数重命名
        print(new_name)  # 打印新的文件名字



    # s = 'Chevrolet3986_small.jpg'
    # print(s)
    #
    # ss = [''.join(list(g)) for k, g in groupby(s, key=lambda x: x.isdigit())]
    # print(ss[0])
    # print(ss[1])
    # print(ss[2])