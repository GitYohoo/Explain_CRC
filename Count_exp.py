import os
import pandas as pd
import csv

def count_exp(org_path, count_path):
    """统计得到的解释"""
    
    def get_char_count(org_path):
        #统计排序
        files = os.listdir(org_path)
        char_count_positive = {}
        char_count_negative = {}

        for file in files:
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(org_path, file), encoding='gbk')
                for i in range(0, 4):
                    col = df.iloc[i, 1]
                    contribution = df.iloc[i, 2]
                    if int(contribution) > 0:
                        if col in char_count_positive:
                            char_count_positive[col] += 1
                        else:
                            char_count_positive[col] = 1
                    if int(contribution) < 0:
                        if col in char_count_negative:
                            char_count_negative[col] += 1
                        else:
                            char_count_negative[col] = 1
        return  char_count_positive.items(), char_count_negative.items()

    def write_csv(char_count_sorted, label, mode):
        #把最终结果写到csv
        #按照逗号分隔
        values = [str(i[0]) + ',' + str(i[1]) for i in char_count_sorted]
        #按照逗号分割成三列
        char_count_sorted = [i.split(',') for i in values]
        with open(count_path + 'label_{}_{}.csv'.format(label, mode), 'w', newline='', encoding='gbk') as csvfile:
            for char in char_count_sorted:
                writer = csv.writer(csvfile)
                writer.writerow(char)
    modes = ['positive', 'negative']
    #排序
    for mode in modes:
        if mode == "positive":    
            char_count_sorted_1 = sorted(get_char_count(org_path.format(1))[0], key=lambda x: x[1], reverse=True)
            char_count_sorted_2 = sorted(get_char_count(org_path.format(2))[0], key=lambda x: x[1], reverse=True)
            char_count_sorted_3 = sorted(get_char_count(org_path.format(3))[0], key=lambda x: x[1], reverse=True)
            char_count_sorted_4 = sorted(get_char_count(org_path.format(4))[0], key=lambda x: x[1], reverse=True)
        #写入csv
        for i in range(4):
            write_csv(eval('char_count_sorted_{}'.format(i+1)), i+1, mode="p")

        if mode == "negative":    
            char_count_sorted_1 = sorted(get_char_count(org_path.format(1))[1], key=lambda x: x[1], reverse=True)
            char_count_sorted_2 = sorted(get_char_count(org_path.format(2))[1], key=lambda x: x[1], reverse=True)
            char_count_sorted_3 = sorted(get_char_count(org_path.format(3))[1], key=lambda x: x[1], reverse=True)
            char_count_sorted_4 = sorted(get_char_count(org_path.format(4))[1], key=lambda x: x[1], reverse=True)
        #写入csv
        for i in range(4):
            write_csv(eval('char_count_sorted_{}'.format(i+1)), i+1, mode="n")

def sum_all(sum_path):
    # 获取SVM下所有label的csv文件路径
    #删除path——labels下的labels.csv
    if os.path.exists(sum_path + 'labels.csv'):
        os.remove(sum_path + 'labels.csv')
    csv_files = []
    csv_files = [os.path.join(sum_path, f) for f in os.listdir(sum_path) if os.path.isfile(os.path.join(sum_path, f)) and f.endswith('.csv')]
    #将0和1调换位置，2和3调换位置，4和5调换位置，4和5调换位置，6和7调换位置
    for i in range(len(csv_files)):
        if i % 2 == 0:#
            csv_files[i], csv_files[i+1] = csv_files[i+1], csv_files[i]
    # 读取所有csv文件并拼接
    labels_df = pd.DataFrame()
    for csv_file in csv_files:
        #横向拼接
        labels_df = pd.concat([labels_df, pd.read_csv(csv_file, encoding='gbk')], axis=1)

    # 将拼接后的DataFrame保存为labels.csv
    headers = ['AWNP','+','AWNP','-','AWP','+','AWP','-','DWNP','+','DWNP','-','DWNP','+','DWNP','-',]
    labels_df.to_csv(sum_path + 'labels.csv', index=False, header=headers)

def quotation_marks(sum_path):
    #检查D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\firstlayer\XGB\labels.csv这个文件的每一个单元格，在所有单元格前面加一个单引号
    with open(sum_path + 'labels.csv', 'r', encoding='gbk') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    # Modify the rows and write them to a new CSV file
    modified_rows = []
    for row in rows:
        modified_row = ["'" + cell for cell in row]
        modified_rows.append(modified_row)

    with open(sum_path + 'labels_new.csv', 'w', newline='', encoding='gbk') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(modified_rows)