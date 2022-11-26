# 오픈소스2차과제
scikit-learn을 이용한 automl

처음엔 필요한 것들을 import 해줍니다

import sys

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



판다스를 이용하여 csv파일을 DataFrame에 저장하고 return 해줍니다

def load_dataset(dataset_path):

	return pd.read_csv(dataset_path)


groupby 를 이용해서 size에 size series를 만들고 이후 feature의 길이는 len함수를 이용하고 feature중 target을 제외하기 위해 -1을 해준다음 size[0], size[1]
을 return 하여 원하는 값을 얻습니다

def dataset_stat(dataset_df):

    size = dataset_df.groupby("target").size()
    
    return len(dataset_df.columns) - 1, size[0], size[1]
   
이후 내용은 강의자료에 나와있는 그래도 입력하면됩니다
