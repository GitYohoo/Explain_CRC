import pandas as pd
import csv

def save_exp(exp, i, output, label, csv_path, html_path):
     
    """对每一个类别都进行解释的保存"""

    local_exp_values = exp.local_exp[label]
    #取出 local_exp_values中的第一列
    sortted_index = [i[0] for i in local_exp_values]
    #获取解释的各个特征
    list_exp_values  = exp.as_list(label=label)
    #去掉括号和引号
    for x in range(len(list_exp_values)):
        list_exp_values_str = str(list_exp_values[x])
        list_exp_values[x] = list_exp_values_str.replace('(', '').replace(')', '').replace("'", '')
    #拼接
    merged_exp_values = list(zip(local_exp_values, list_exp_values))
    #按照逗号分隔
    merged_exp_values = [str(i[0][0]) + ',' + str(i[1]) for i in merged_exp_values]
    #按照逗号分割成三列
    merged_exp_values = [i.split(',') for i in merged_exp_values]
    header = ['feature_numbers', 'feature_bins', 'contributions']
    pd.DataFrame(merged_exp_values).to_csv(csv_path, 
                                        index=False, header=header)
    #追加标签信息到csv
    with open(csv_path, 'a', newline='', encoding='gbk') as csvfile:
        for true_or_pred_label in output:
            writer = csv.writer(csvfile)
            writer.writerow([true_or_pred_label])
    exp.save_to_file(html_path.format(label+1, i))