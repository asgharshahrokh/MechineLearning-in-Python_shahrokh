import numpy as np
from collections import defaultdict
from sklearn import preprocessing

data_path = '.\cleve.mod'
n_instance = 303
n_attr = 15

def load_rating_data(data_path, n_instance, n_attr):
 
     data = []
 
     with open(data_path, 'r') as file:
        for line in file.readlines():
          record= line.strip().split(" ")
          record= [ r for r in record if r]
          data.append(record)
    
     return data; 


data= np.array(load_rating_data(data_path, n_instance, n_attr))

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['fem','male'])
data[:,1] = le_sex.transform(data[:,1]) 

le_cp = preprocessing.LabelEncoder()
le_cp.fit(['abnang','angina','asympt','notang'])
data[:,2] = le_cp.transform(data[:,2]) 

le_fbs = preprocessing.LabelEncoder()
le_fbs.fit(['fal','true'])
data[:,5] = le_fbs.transform(data[:,5]) 

le_restecg = preprocessing.LabelEncoder()
le_restecg.fit(['abn','hyp','norm'])
data[:,6] = le_restecg.transform(data[:,6]) 

le_exang = preprocessing.LabelEncoder()
le_exang.fit(['fal','true'])
data[:,8] = le_exang.transform(data[:,8]) 

le_slope = preprocessing.LabelEncoder()
le_slope.fit(['down','flat','up'])
data[:,10] = le_slope.transform(data[:,10]) 

le_ca = preprocessing.LabelEncoder()
le_ca.fit(['0.0','1.0','2.0','3.0','?'])
data[:,11] = le_ca.transform(data[:,11]) 

le_thal = preprocessing.LabelEncoder()
le_thal.fit(['?','fix','norm','rev'])
data[:,12] = le_thal.transform(data[:,12]) 
 
le_num = preprocessing.LabelEncoder()
le_num.fit(['sick','buff'])
data[:,13] = le_num.transform(data[:,13]) 
 
le_helth = preprocessing.LabelEncoder()
le_helth.fit(['S1','S2','S3','S4','H'])
data[:,14] = le_helth.transform(data[:,14]) 

print(data[0:5])
 
def display_distribution(data):
     values, counts = np.unique(data[:,14], return_counts=True)
     for value, count in zip(values, counts):
       print(f'Number of rating {value}: {count}')  


display_distribution(data) 

X_raw = data[:,0:12].astype(np.float64)

Y_raw = data[:, 13].astype(np.float64)

print('Shape of X_raw:', X_raw.shape)
print('Shape of Y_raw:', Y_raw.shape)

 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_raw, Y_raw, test_size=0.2, random_state=42)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)

prediction_prob = clf.predict_proba(X_test)
print(prediction_prob[0:20])

prediction = clf.predict(X_test)
print(prediction[:20])

accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is:{accuracy*100:.1f}%')

 

print('success')
