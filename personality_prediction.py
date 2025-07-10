#IMPORT THƯ VIỆN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lazypredict.Supervised import LazyClassifier
#EDA DỮ LIỆU
df=pd.read_csv('C:/Users/HUU DUY/Downloads/personality_dataset (1).csv')
df.head()
df.info()
df.isnull().sum()
df.shape
#XỬ LÝ NULL, CHUẨN HÓA DỮ LIỆU
label_encoder=LabelEncoder()
impute=SimpleImputer(strategy='median')
imputer=SimpleImputer(strategy='most_frequent')
df[['Time_spent_Alone','Social_event_attendance','Going_outside',
    'Friends_circle_size','Post_frequency']]=impute.fit_transform(
        df[['Time_spent_Alone','Social_event_attendance','Going_outside',
        'Friends_circle_size','Post_frequency']])
df[['Stage_fear','Drained_after_socializing']]=imputer.fit_transform(
    df[['Stage_fear','Drained_after_socializing']])
#ĐỘ PHÂN BỐ CỦA DỮ LIỆU CATEGORICAL    
categorical_cols = ['Personality', 'Stage_fear', 'Drained_after_socializing']
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
df['Personality'].value_counts()
#ĐỘ PHÂN BỐ DỮ LIỆU NUMERICAL
numerical_cols=['Time_spent_Alone','Social_event_attendance','Going_outside'
                ,'Friends_circle_size','Post_frequency']
for col in numerical_cols:
    # plt.figure(figsize=(6,4))
    sns.histplot(data=df,x=col)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('count')
    plt.show()
#TƯƠNG QUAN GIỮA STAGE_FEAR AND PERSONALITY
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Stage_fear', hue='Personality')
plt.title('Tương Quan Stage_fear và Personality')
plt.xlabel('Stage_fear')
plt.ylabel('Count')
plt.legend(title='Personality')
plt.tight_layout()
plt.show()
#TƯƠNG QUAN GIỮA DRAINED_AFTER_SOCIALIZING AND PERSONALITY
plt.figure(figsize=(6,4))
sns.countplot(data=df,x='Drained_after_socializing',hue='Personality')
plt.title('Tương Quan Drained_after_socializing And Personalizing')
plt.xlabel('Drained_after_socializing')
plt.ylabel('Count')
plt.legend(title='Personality')
plt.tight_layout()
plt.show()
#TƯƠNG QUAN GIỮA TIME_SPENT_ALONE AND PERSONALITY
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Personality', y='Time_spent_Alone')
plt.title('Tương Quan Time_spent_Alone And Personality')
plt.xlabel('Personality')
plt.ylabel('Time Spent Alone')
plt.tight_layout()
plt.show()
#Chuẩn Hóa Dữ Liệu
for column in df.select_dtypes(include=['object']).columns:
    df[column]=label_encoder.fit_transform(df[column])
#HỆ SỐ TƯƠNG QUAN CỦA DỮ LIỆU
corr=df.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr,cmap='coolwarm',annot=True)
plt.show()
#Xây Model
x=df.drop('Personality',axis=1)
y=df['Personality']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
clt=LazyClassifier()
pred_model=clt.fit(x_train,x_test,y_train,y_test)
print(pred_model[0].sort_values(by='Accuracy', ascending=False).head(5))
from sklearn.linear_model import LogisticRegression
# Khởi tạo mô hình BernoulliNB

ber=Pipeline(steps=[
    ('scaler',StandardScaler()),
    ('model',LogisticRegression())
    ])
ber.fit(x_train, y_train)
# Dự đoán trên tập kiểm tra
y_pred = ber.predict(x_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
report=classification_report(y_test, y_pred)
report1=confusion_matrix(y_test, y_pred)
print(report)
print(report1)
sns.heatmap(report1, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Extrovert', 'Introvert'],
            yticklabels=['Extrovert', 'Introvert'])
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Confusion Matrix-LogisticRegression')
plt.show()