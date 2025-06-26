import pandas as pd
import matplotlib.pyplot as mlt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lazypredict.Supervised import LazyClassifier
df=pd.read_csv('C:/Users/HUU DUY/Downloads/personality_dataset.csv')
df.head()
label_encoder=LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column]=label_encoder.fit_transform(df[column])
impute=SimpleImputer(strategy='median')
imputer=SimpleImputer(strategy='most_frequent')
df[['Time_spent_Alone','Social_event_attendance','Going_outside',
    'Friends_circle_size','Post_frequency']]=impute.fit_transform(df[['Time_spent_Alone','Social_event_attendance','Going_outside',
        'Friends_circle_size','Post_frequency']])
x=df.drop('Personality',axis=1)
y=df['Personality']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.naive_bayes import BernoulliNB
# Khởi tạo mô hình BernoulliNB
model = BernoulliNB()

# Huấn luyện mô hình
model.fit(x_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
report=classification_report(y_test, y_pred)
report1=confusion_matrix(y_test, y_pred)
