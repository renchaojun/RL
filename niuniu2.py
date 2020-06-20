import pandas as pd
import numpy as np
import os
import pandas as pd

# 读取整个csv文件
csv_data = pd.read_csv("processed_data.csv")
columns=csv_data.columns.values
print(len(csv_data))
value=csv_data.values
# print(value)
data=[]
for line in value:
    indata=[]
    for juti in line:
        try:
            if juti>=0:
                indata.append(juti)
                if len(indata)==27:
                    data.append(indata)
            else:
                break
        except:
            break
print(len(data))
xinshuju = pd.DataFrame(data, columns=columns)
# xinshuju.dropna(axis=0, how='any', inplace=True)
# print(xinshuju.values)
xinshuju.to_csv("processed_data.csv", header=xinshuju.columns.values)
