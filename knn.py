import pandas as pd
import numpy as np
#读取数据和数据集
df = pd.read_csv('Data/breast+cancer+wisconsin+diagnostic/wdbc.csv')
print(df)
colnames = df.columns[2: ]
df = df[colnames]
train_data = df[0:468]
test_data = df[468:568]
train_labels = train_data['Diagnosis']
label_map = {'Malignant':0, 'Benign':1}
train_labels = train_labels.map(label_map) #将字符串转为数
test_labels = test_data['Diagnosis'].map(label_map)
test_labels = test_labels.reset_index(drop=True)
train = train_data.iloc[:, 1:]   #去除标签
test = test_data.iloc[:, 1:]

#标准化数据集
def min_max(x):
    return (x - x.min()) / (x.max() - x.min())

train = train.apply(min_max)
test = test.apply(min_max)

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

predictions = knn(train, train_labels, test,k = 21)

accuracy = (predictions['Diagnosis'] == test_labels).mean()
print(f"准确率：{accuracy}")
