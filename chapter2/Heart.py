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
          record= [ c for c in record if c]
          data.append(record)
    
     return data; 

data= load_rating_data(data_path, n_instance, n_attr)

data= np.array(data)

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

 
 
def display_distribution(data):
     values, counts = np.unique(data[:,14], return_counts=True)
     for value, count in zip(values, counts):
       print(f'Number of rating {value}: {count}')  


display_distribution(data) 

data = data.astype(np.float32)

X = data[:,0:12]
Y = data[:, 13]

print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)

 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=5.0, fit_prior=False)
clf.fit(X_train, Y_train)
print('\n')
prediction_prob = clf.predict_proba(X_test)
print("prediction_prob for 3 X_test:")
print(prediction_prob[0:3])
print('\n')
prediction = clf.predict(X_test)
print("prediction for 10 x_test:")
print(prediction[0:10])
print('\n')
accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is:{accuracy*100:.1f}%')

from sklearn.metrics import confusion_matrix
print('confusion_matrix:\n',confusion_matrix(Y_test, prediction, labels=[0, 1]))
print('\n')
from sklearn.metrics import precision_score, recall_score, f1_score

print('precision_score is:',precision_score(Y_test, prediction, pos_label=1))

print('recall_score is:',recall_score(Y_test, prediction, pos_label=1))

print('f1_score_pos_lable 1 is:',f1_score(Y_test, prediction, pos_label=1))

print('f1_score_pos_lable 0 is:',f1_score(Y_test, prediction, pos_label=0))
print('\n')

from sklearn.metrics import classification_report
report = classification_report(Y_test, prediction)
print(report)

pos_prob = prediction_prob[:, 1]
thresholds = np.arange(0.0, 1.1, 0.05)
true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
for pred, y in zip(pos_prob, Y_test):
 for i, threshold in enumerate(thresholds):
   if pred >= threshold:
   # if truth and prediction are both 1
     if y == 1:
      true_pos[i] += 1
   # if truth is 0 while prediction is 1
     else:
      false_pos[i] += 1
   else:
     break

n_pos_test = (Y_test == 1).sum()
n_neg_test = (Y_test == 0).sum()

true_pos_rate = [tp / n_pos_test for tp in true_pos]
false_pos_rate = [fp / n_neg_test for fp in false_pos]



from sklearn.metrics import roc_auc_score
print('roc_auc_score is:',roc_auc_score(Y_test, pos_prob))

print('\n')
 
# import matplotlib.pyplot as plt
# plt.figure()
# lw = 2
# plt.plot(false_pos_rate, true_pos_rate,color='darkorange', lw=lw)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()
from sklearn.model_selection import StratifiedKFold
k = 5
k_fold = StratifiedKFold(n_splits=k ,shuffle=True, random_state=42)

smoothing_factor_option = [1, 2, 3, 4,5,6]
fit_prior_option = [True, False]
auc_record = {}

for train_indices, test_indices in k_fold.split(X, Y):
       Xx_train, Xx_test = X[train_indices], X[test_indices]
       Yy_train, Yy_test = Y[train_indices], Y[test_indices]
       for alpha in smoothing_factor_option:
          if alpha not in auc_record:
            auc_record[alpha] = {}
          for fit_prior in fit_prior_option:
            clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            clf.fit(Xx_train, Yy_train)
            prediction_prob = clf.predict_proba(Xx_test)
            pos_prob = prediction_prob[:, 1]
            auc = roc_auc_score(Yy_test, pos_prob)
            auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior, 0.0)   
            
for smoothing, smoothing_record in auc_record.items():
    for fit_prior, auc in smoothing_record.items():
        print(f' {smoothing} {fit_prior} {auc/k:.5f}')
        
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)
pos_prob = clf.predict_proba(X_test)[:, 1]
print('AUC with the best model:', roc_auc_score(Y_test, pos_prob))
     
print('program end with by success.\n')

