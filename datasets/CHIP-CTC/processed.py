import pandas
import json

# 获取所有类别
label = pandas.read_excel('./category.xlsx')['Label Name'].to_list()

dict = {}
data_path = './data/'
i = 0
with open(data_path + 'class.txt', 'w', encoding = 'utf-8') as f:
    for x in label:
        dict[x] = i
        f.write(x + '\n')
        i+=1

print(dict)

def load_data(path, save_path, test = False):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if test == False:
        with open(save_path, 'w', encoding='utf-8') as f:
            for item in data:
                item = item['text'].strip() + '\t' + str(dict[item['label']])
                f.write(item + '\n')
    else:
        with open(save_path, 'w', encoding='utf-8') as f:
            for item in data:
                item = item['text'].strip() + '\t' + str(-1)
                f.write(item + '\n')

load_data('./CHIP-CTC_train.json', data_path + 'train.txt')
load_data('./CHIP-CTC_dev.json', data_path + 'dev.txt')
load_data('./CHIP-CTC_test.json', data_path + 'test.txt', test=True)