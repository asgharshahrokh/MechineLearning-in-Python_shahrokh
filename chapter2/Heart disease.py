import numpy as np
from collections import defaultdict
from sklearn import preprocessing

data_path = 'cleve.mod'
n_instance = 303
n_attr = 14

def load_rating_data(data_path, n_instance, n_attr):
 
     data = []
 
     with open(data_path, 'r') as file:
        for line in file.readlines():
          record= line.strip().split(" ")
          record= [ r for r in record if r]
          data.append(record)
    
     return data; 


data=np.array(load_rating_data(data_path, n_instance, n_attr))

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['fem','male'])
data[:,1] = le_sex.transform(data[:,1]) 
 
le_helth = preprocessing.LabelEncoder()
le_helth.fit(['H','S1','S2','S3','S4'])
data[:,14] = le_helth.transform(data[:,14]) 

le_helth = preprocessing.LabelEncoder()
le_helth.fit(['abnang','angina','asympt','notang'])
data[:,2] = le_helth.transform(data[:,2]) 

le_helth = preprocessing.LabelEncoder()
le_helth.fit(['fal','true'])
data[:,5] = le_helth.transform(data[:,5]) 

le_helth = preprocessing.LabelEncoder()
le_helth.fit(['abn','hyp','norm'])
data[:,6] = le_helth.transform(data[:,6]) 

le_helth = preprocessing.LabelEncoder()
le_helth.fit(['fal','true'])
data[:,8] = le_helth.transform(data[:,8]) 

le_helth = preprocessing.LabelEncoder()
le_helth.fit(['down','flat','up'])
data[:,10] = le_helth.transform(data[:,10]) 

le_helth = preprocessing.LabelEncoder()
le_helth.fit(['0.0','1.0','2.0','3.0','?'])
data[:,11] = le_helth.transform(data[:,11]) 

le_helth = preprocessing.LabelEncoder()
le_helth.fit(['?','fix','norm','rev'])
data[:,12] = le_helth.transform(data[:,12]) 
 

print(data[0:5])
 
def display_distribution(data):
     values, counts = np.unique(data[:,14], return_counts=True)
     for value, count in zip(values, counts):
       print(f'Number of rating {value}: {count}')  


display_distribution(data) 

"""movie_id_most, n_rating_most = sorted(movie_n_rating.items(), key=lambda d: d[1], reverse=True)[0]
print(f'Movie ID {movie_id_most} has {n_rating_most} ratings.')"""

print('Shape of Data :', data.shape)

X_raw = data[:,0:13].astype(np.float64)

Y_raw = data[:, 14].astype(np.float64)

print('Shape of X_raw:', X_raw.shape)
print('Shape of Y_raw:', Y_raw.shape)

print(X_raw[0:5])
 
"""recommend = 0

Y_raw[Y_raw <= recommend] = 0
Y_raw[Y_raw > recommend] = 1"""

""""
 
print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)  
 
 

"""""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_raw, Y_raw, test_size=0.2, random_state=42)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)

prediction_prob = clf.predict_proba(X_test)
print(prediction_prob[0:10])

prediction = clf.predict(X_test)
print(prediction[:10])

accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is:{accuracy*100:.1f}%')

"""

 
x=np.array([ [1,2,1,1,2,3] , [2,4,4,5,5,6] ])
x=x[x>2]
print(x)  """

print('success')
