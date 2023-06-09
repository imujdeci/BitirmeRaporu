# -*- coding: utf-8 -*-
# Temel Kütüphanler
import os
import sys
import os.path
import numpy as np
import pandas as pd
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.pylab import rcParams

# Toollar
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# Kullanılacak Modeller
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Model Analizi
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 





#veriyi data.csv'den okuyoruz
data_file_name = 'data.csv'
if not os.path.isfile(data_file_name):
    print(f'File {data_file_name} not found. Please make sure the file is locatable and change the name of file in variable above')
    sys.exit(-1)
    

my_dataframe = pd.read_csv('data.csv')
my_dataframe.head()


#veriden id ve teşhisi çıkarıyoruz  , veriyi iloc ile ayırıp x ve y ye bölüyoruz  
col_list = my_dataframe.columns.to_list()
my_dataFrameX = my_dataframe[col_list[2:-1]]
my_dataFrameY = my_dataframe[col_list[1]]
my_dataX = my_dataFrameX.iloc[:].values
my_dataY = my_dataFrameY.iloc[:].values


correlation_matrix = np.corrcoef(my_dataX.T)



#veriyi train_test_split ile eğitim verisi ve test verisi olarak bölüyoruz

X_train,X_test,Y_train,Y_test = train_test_split(my_dataX,my_dataY,test_size=0.20, random_state=4)
print(f'Eğitim-Test Veri Boyutu :\n-----------------\nX_train:{X_train.shape}, Y_train:{Y_train.shape}\nX_test:{X_test.shape}, Y_test:{Y_test.shape}')


#veri içinde 1000 ve 0.005 gibi veriler olduğu için veriyi standartlaştırıyoruz 

sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



#KNN
print("KNN için: \n")
accuracy_KNN_test = list()
accuracy_KNN_train = list()
for p_test in range(1,11):
    KNN_Classifier = KNeighborsClassifier(n_neighbors = int(0.05*X_train.shape[0]), p = p_test, metric='minkowski')
   
    KNN_Classifier.fit(X_train, Y_train)
   
    Y_pred_KNN = KNN_Classifier.predict(X_test)
    accuracy_KNN_test.append(len(Y_pred_KNN[Y_pred_KNN == Y_test])/len(Y_test))
   
    Y_pred_KNN = KNN_Classifier.predict(X_train)
    accuracy_KNN_train.append(len(Y_pred_KNN[Y_pred_KNN == Y_train])/len(Y_train))

#plt.plot([i for i in range(1,11)],np.asarray(accuracy_KNN_test),label='test accuracy')
#plt.plot([i for i in range(1,11)],np.asarray(accuracy_KNN_train),label='train accuracy')
#plt.xlabel('P değeri')
#plt.ylabel('Accuracy')
#plt.title('Knn için Doğruluk Oranı')
#plt.legend()

best_p = 2



#En iyi KNN Değeri
print("En iyi KNN Değeri İçin: \n")
KNN_Classifier = KNeighborsClassifier(n_neighbors = int(0.05*X_train.shape[0]), p = 2, metric='minkowski')
KNN_Classifier.fit(X_train, Y_train)
Y_pred_KNN = KNN_Classifier.predict(X_test)
print(f'{len(Y_pred_KNN[Y_pred_KNN==Y_test])} out of {len(Y_test)} values are correctly predicted.\n\tAccuracy={len(Y_pred_KNN[Y_pred_KNN==Y_test])/len(Y_test)}')
confusion_KNN = confusion_matrix(Y_test,Y_pred_KNN)
print(f'\n Karmaşıklık Matrisi :\n{confusion_KNN}')
print(classification_report(Y_test, Y_pred_KNN, target_names=['Class 0', 'Class 1']))



#Lojistik Regresyon 
print("Logistik Regression için: \n")
accuracy_train_LR = list()
accuracy_test_LR = list()
sp_list = np.arange(.05,.6,0.05)
for per in sp_list:
    X_train_lr, X_test_lr, Y_train_lr, Y_test_lr = train_test_split(my_dataX, my_dataY, test_size=per, random_state=4)
    
    scLR = StandardScaler()
    X_train_lr = scLR.fit_transform(X_train_lr)
    X_test_lr = sc.transform(X_test_lr)
    
    Logistic_Classifier = LogisticRegression(random_state=0, solver='lbfgs', max_iter=300)
    Logistic_Classifier.fit(X_train_lr,Y_train_lr)
    
    Y_pred_LC = Logistic_Classifier.predict(X_test_lr)
    accuracy_test_LR.append(len(Y_pred_LC[Y_pred_LC==Y_test_lr])/len(Y_test_lr))
    Y_pred_LC = Logistic_Classifier.predict(X_train_lr)
    accuracy_train_LR.append(len(Y_pred_LC[Y_pred_LC==Y_train_lr])/len(Y_train_lr))


#plt.plot([i for i in sp_list],np.asarray(accuracy_test_LR),label='test accuracy')
#plt.plot([i for i in sp_list],np.asarray(accuracy_train_LR),label='train accuracy')
#plt.xlabel('Test/Train Split')
#plt.ylabel('Accuracy')
#plt.title('Accuracy Plot for Logistic Regression')
#plt.legend()


X_train_lr, X_test_lr, Y_train_lr, Y_test_lr = train_test_split(my_dataX, my_dataY, test_size=0.45, random_state=4)

scLR = StandardScaler()
X_train_lr = scLR.fit_transform(X_train_lr)
X_test_lr = sc.transform(X_test_lr)

Logistic_Classifier = LogisticRegression(random_state=0, solver='lbfgs', max_iter=300)
Logistic_Classifier.fit(X_train_lr,Y_train_lr)

Y_pred_LC = Logistic_Classifier.predict(X_test_lr)

print(f'{len(Y_pred_LC[Y_pred_LC==Y_test_lr])} out of {len(Y_test_lr)} values are correctly predicted.\n\tAccuracy={len(Y_pred_LC[Y_pred_LC==Y_test_lr])/len(Y_test_lr)}')
confusion_LC = confusion_matrix(Y_test_lr, Y_pred_LC)
print(f'\n Karmaşıklık Matrisi :\n{confusion_LC}')
print(classification_report(Y_test_lr, Y_pred_LC, target_names=['Class 0','Class 1']))




#Decision Tree Algoritması

print("Decision Tree için: \n")
Entropy_Classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
Entropy_Classifier.fit(X_train,Y_train)
Y_pred_entropy = Entropy_Classifier.predict(X_test)
print(f'{len(Y_pred_entropy[Y_pred_entropy==Y_test])} out of {len(Y_test)} values are correctly predicted.\n\tAccuracy={len(Y_pred_entropy[Y_pred_entropy==Y_test])/len(Y_test)}')
confusion_entropy = confusion_matrix(Y_test,Y_pred_entropy)
print(f'\n Karmaşıklık Matrisi :\n{confusion_entropy}')


rcParams['figure.figsize'] = 80, 50

tree.plot_tree(Entropy_Classifier)
print(classification_report(Y_test, Y_pred_entropy, target_names=['Class 0','Class 1']))



Gini_Classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
Gini_Classifier.fit(X_train,Y_train)
Y_pred_gini = Gini_Classifier.predict(X_test)
print(f'{len(Y_pred_gini[Y_pred_gini==Y_test])} out of {len(Y_test)} values are correctly predicted.\n\tAccuracy={len(Y_pred_gini[Y_pred_gini==Y_test])/len(Y_test)}')
confusion_gini = confusion_matrix(Y_test, Y_pred_gini)
print(f'\n Karmaşıklık Matrisi :\n{confusion_gini}')


rcParams['figure.figsize'] = 80,50

tree.plot_tree(Gini_Classifier)
print(classification_report(Y_test, Y_pred_gini, target_names=['Class 0','Class 1']))





#Gaussian Naive Bayes
print("Niave Bayes için: \n")
classifier_gaussian = GaussianNB()
classifier_gaussian.fit(X_train, Y_train)
Y_pred_gaussian = classifier_gaussian.predict(X_test)
print(f'\n Karmaşıklık Matrisi :\n{confusion_matrix(Y_test,Y_pred_gaussian)}')
print(classification_report(Y_test, Y_pred_gaussian, target_names=['Class 0','Class 1']))


#SVC

print("SVC için: \n")
classifier_SVC_linear = svm.SVC(kernel = 'linear', random_state = 0)
classifier_SVC_linear.fit(X_train,Y_train)
Y_pred_SVM_linear = classifier_SVC_linear.predict(X_test)
print(f' Karmaşıklık Matrisi :\n {confusion_matrix(Y_test,Y_pred_SVM_linear)}')
print(classification_report(Y_test,Y_pred_SVM_linear,target_names=['Class 0','Class 1']))


classifier_SVC_radial = svm.SVC(kernel = 'rbf', random_state = 0)
classifier_SVC_radial.fit(X_train,Y_train)
Y_pred_SVM_rbf = classifier_SVC_radial.predict(X_test)
print(f' Karmaşıklık Matrisi :\n {confusion_matrix(Y_test,Y_pred_SVM_rbf)}')
print(classification_report(Y_test,Y_pred_SVM_rbf,target_names=['Class 0','Class 1']))


classifier_SVC_linear = svm.LinearSVC()
classifier_SVC_linear.fit(X_train,Y_train)
Y_pred_SVM_linear = classifier_SVC_linear.predict(X_test)
print(f' Karmaşıklık Matrisi :\n {confusion_matrix(Y_test,Y_pred_SVM_linear)}')
print(classification_report(Y_test,Y_pred_SVM_linear,target_names=['Class 0','Class 1']))


classifier_SVC_nu = svm.NuSVC(gamma='scale')
classifier_SVC_nu.fit(X_train,Y_train)
Y_pred_SVM_nu = classifier_SVC_nu.predict(X_test)
print(f' Karmaşıklık Matrisi :\n {confusion_matrix(Y_test,Y_pred_SVM_nu)}')
print(classification_report(Y_test,Y_pred_SVM_nu,target_names=['Class 0','Class 1']))



#Random Forest Gini Ve Entropy için Denemeler

print("Random Forest için: \n")
accuracy_est_test = list()
accuracy_est_train = list()
for n_est in range(5,75):
    classifier = RandomForestClassifier(n_estimators = n_est, criterion = 'gini', random_state = 0)
    classifier.fit(X_train, Y_train)
    Y_pred_rf = classifier.predict(X_test)
    Y_pred_rf_train = classifier.predict(X_train)
    accuracy_est_test.append(len(Y_pred_rf[Y_pred_rf==Y_test])/len(Y_pred_rf))
    accuracy_est_train.append(len(Y_pred_rf_train[Y_pred_rf_train==Y_train])/len(Y_pred_rf_train))
    
#plt.plot([i for i in range(5,75)],np.asarray(accuracy_est_test),label='test accuracy')
#plt.plot([i for i in range(5,75)],np.asarray(accuracy_est_train),label='train accuracy')
#plt.xlabel('Number of Estimators')
#plt.ylabel('Accuracy')
#plt.title('Accuracy Plot for Random Forest')  
#plt.legend()



best_estimator = 28
classifier = RandomForestClassifier(n_estimators = best_estimator, criterion = 'gini', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred_rf = classifier.predict(X_test)
print(f' Karmaşıklık Matrisi :\n {confusion_matrix(Y_test,Y_pred_rf)}')
print(classification_report(Y_test,Y_pred_rf,target_names=['Class 0','Class 1']))


accuracy_est_test = list()
accuracy_est_train = list()
for n_est in range(5,75):
    classifier = RandomForestClassifier(n_estimators = n_est, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, Y_train)
    Y_pred_rf = classifier.predict(X_test)
    Y_pred_rf_train = classifier.predict(X_train)
    accuracy_est_test.append(len(Y_pred_rf[Y_pred_rf==Y_test])/len(Y_pred_rf))
    accuracy_est_train.append(len(Y_pred_rf_train[Y_pred_rf_train==Y_train])/len(Y_pred_rf_train))
    
#plt.plot([i for i in range(5,75)],np.asarray(accuracy_est_test),label='test accuracy')
#plt.plot([i for i in range(5,75)],np.asarray(accuracy_est_train),label='train accuracy')
#plt.xlabel('Number of Estimators')
#plt.ylabel('Accuracy')
#plt.title('Accuracy Plot for Random Forest')  
#plt.legend()


best_estimator = 42
classifier = RandomForestClassifier(n_estimators = best_estimator, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred_rf = classifier.predict(X_test)
print(f' Karmaşıklık Matrisi :\n {confusion_matrix(Y_test,Y_pred_rf)}')
print(classification_report(Y_test,Y_pred_rf,target_names=['Class 0','Class 1']))






