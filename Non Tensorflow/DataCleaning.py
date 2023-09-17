import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("coworker raw.csv")
# removing the irrelevant columns
cols_to_drop = ["Name", "End Date", "Start Date"]
df = df.drop(columns=cols_to_drop,axis=1)# first five rows of dataframe after removing columns
df["Age"] = pd.cut(df["Age"],bins=[10,20,22,24,26,28,35])
"""values_list = []
test_item = "Start Month"
for current_value in list(df[test_item].unique()):

    p = round((df["Stay Length"][df[test_item]==current_value].mean()), 2)

    values_list.append(p)
    print(current_value,"(Average Stay Length) : ", p)
"""
df = pd.get_dummies(df)
df = df.drop(columns=["Education_None","English level_Basic", "Gender_Male", "Bar Experience_None"],axis=1)
df.to_csv('coworker_clean.csv')
