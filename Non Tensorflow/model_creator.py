import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from joblib import dump, load
df = pd.read_csv('coworker_clean.csv')
feat = df.iloc[: , 1:]
feat = feat.drop(columns=['Quit', 'Stay Length'],axis=1)
label_stay_length = df["Stay Length"]
label_quit = df["Quit"]
stay_length_predictor = SVC(kernel='rbf')
stay_length_predictor.fit(feat, label_stay_length)
dump(stay_length_predictor, 'stay_length_predictor.joblib')
quit_predictor = SVC(kernel='rbf')
quit_predictor.fit(feat, label_quit)
dump(quit_predictor, 'quit_predictor.joblib')
