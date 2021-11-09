import os
from scipy import signal
import numpy as np
from tqdm import tqdm
import xlrd
import xlwt

workbook = xlrd.open_workbook(r'/home/lihang/ecg/data/CEW舒张功能名单.xlsx')
sheet_content = workbook.sheet_by_index(0) # sheet索引从0开始

#print(sheet_content.row(2)[2].value)

train_data_path='/home/lihang/ecg/data/multi_class_good/'  #全部txt
# label_path='./CABG舒张功能 名单.xlsx'
dir_path = '/home/lihang/ecg/data/multi_class/'
#dir_path='multi_class'
# if not os.path.exists(os.path.join('data',dir_path)):
#     os.mkdir(os.path.join('data',dir_path))
rows_num = sheet_content.nrows
col_num = sheet_content.ncols
print("有"+str(rows_num) + "行，"  + str(col_num) + "列！")
for i in range(1,rows_num):
    #print(i)
    patient_name = sheet_content.row(i)[1].value
    #print(patient_name)
    patient_rank = int (sheet_content.row(i)[20].value)
    #print(patient_name)
    #print(patient_rank)
    files_name = os.listdir(train_data_path)
    for file_name in files_name:
        file_path_name = os.path.join(train_data_path,file_name)
        #print(file_path_name)
        array = np.load(file_path_name)
       # print(file_name)
        patient = file_name.split()[0]
        if patient == patient_name:
            patient_path = os.path.join(dir_path,str(patient_rank) + '_'+patient_name + '.npy')
            np.save(patient_path,array)

print("sucessfully!")

