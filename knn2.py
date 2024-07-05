import pandas as pd
import numpy as np

#read csv_file
df = pd.read_csv('Data/breast+cancer+wisconsin+diagnostic/wdbc.csv')
print(df.head())

# test-data and train-data
df_new = df[df.columns[2:]] #df.columns[3: ] 提取出df的3之后都列的名字
train_data = df_new[0:468]
test_data = df_new[468:568]
train_data_labels = train_data['Diagnosis']
test_data_labels = test_data['Diagnosis']
##转化标签
label_map = {'Malignant': 0, 'Benign': 1}
train_data_labels = train_data_labels.map(label_map)
test_data_labels = test_data_labels.map(label_map)
test_data_labels_reset = test_data_labels.reset_index(drop=True)
#标准化数据集
def standardization(x):
    return (x - x.min()) / (x.max() - x.min())

train = train_data.iloc[:, 1:]
test = test_data.iloc[:, 1:]
train = train.apply(standardization)
test = test.apply(standardization)
#apply的用法是什么，列还是行应用？按照这个意思，应该是列应用每个函数
#te = test.iloc[0, :]
#tr = train.iloc[0:2, :]
#diff = tr - te
#diff_2 = diff.apply(lambda x: x**2)
#row_sum = diff_2.sum(axis=1)
#sqrt = np.sqrt(row_sum)
#距离函数
def euclidean_distance(df, v):
    diff = df - v
    diff_2 = diff.apply(lambda x: x**2)
    row_sum = diff_2.sum(axis=1)
    sqrt = np.sqrt(row_sum)
    return sqrt

def knn(train, train_labels, test, k):
    predictions = pd.DataFrame(columns = ['Diagnosis'])
    for i in range(test.shape[0]):
        v = test.iloc[i, :]
        distances = euclidean_distance(train, v)
        sorted_indices = np.argsort(distances)
        neighbors = sorted_indices[:k]
        prediction = np.argmax(np.bincount(train_labels.iloc[neighbors].values))
        new_row = pd.DataFrame({'Diagnosis':[prediction]})
        predictions = pd.concat([predictions, new_row], ignore_index = True)
    return predictions
k = 21

pre = knn(train, train_data_labels, test, k)
# 将预测结果映射回原始标签
#reverse_label_map = {0: 'Malignant', 1: 'Benign'}
#pre['Diagnosis'] = pre['Diagnosis'].map(reverse_label_map)

print(pre)
# 计算准确率
accuracy = (pre['Diagnosis'] == test_data_labels_reset).mean()
print(f"准确率: {accuracy}")
