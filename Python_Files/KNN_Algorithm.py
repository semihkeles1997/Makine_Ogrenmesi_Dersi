# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:41:57 2022

@author: semih
"""

# Kütüphaneler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import Calculate_Score as cs
# -----------------------------------------------------------------------------

def KNN_Algorithm_Semih(X_train, X_test, y_train, y_test,max_k):
    return_values = {}
    # Minimum Error ve K Değeri Hesaplama
    # Test Hata Payı
    test_error_rate = []
    for k in range(1,max_k):
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train,y_train)
        y_pred = knn_model.predict(X_test)
        hata = np.sqrt(mean_squared_error(y_test, y_pred))
        test_error_rate.append(np.mean(y_pred != y_test))

    Draw(max_range=max_k, y_value=test_error_rate, title="Test Error Rate vs K Value", 
         xlabel="K Value", ylabel = "Error Rate")

    """
    plt.Figure(figsize=(10,6))
    plt.plot(range(1,max_k),test_error_rate,color='blue', linestyle='dashed', 
             marker='o',markersize=1, markerfacecolor='red')
    plt.title("Test Error Rate vs K Value")
    plt.xlabel("K Value")
    plt.ylabel("Error rate")
    plt.show()
    """
    print("Minimum error : -",min(test_error_rate),
          " at K = ",test_error_rate.index(min(test_error_rate)))
    

    # Maksimum Accuracy ve K Değeri Hesaplama
    from sklearn import metrics
    acc = []
    for i in range(1,max_k):
        neigh = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
        yhat = neigh.predict(X_test)
        acc.append(metrics.accuracy_score(y_test,yhat))

    Draw(max_range = max_k, y_value = acc, title = "Accuracy vs K Value", 
         xlabel = "K Value", ylabel = "Accuracy")
    """
    plt.figure(figsize=(10,6))
    plt.plot(range(1,max_k),acc,color='blue',linestyle='dashed',marker='o',
             markerfacecolor='red',markersize='10')
    plt.title("Accuracy vs K Value")
    plt.xlabel("K Value")
    plt.ylabel("Accuracy")
    plt.show()
    """
    print("Maximum Accuracy: ",max(acc)," at K = ",acc.index(max(acc)))
    
    
    
    
    
    
    # iyileştirilmiş KNN
    best_knn = KNeighborsClassifier(n_neighbors=acc.index(max(acc))) # Accuracy değerinin maksimum olduğu K değeri
    best_knn.fit(X_train, y_train)
    best_knn_pred = best_knn.predict(X_test)
    from sklearn.metrics import accuracy_score
    ac = accuracy_score(y_test, best_knn_pred)

    from sklearn.metrics import confusion_matrix
    con = confusion_matrix(y_test, best_knn_pred)
    
    
    # Confusion matrix plot
    plot_confusion_matrix(best_knn,X_test,y_test)
    plt.show()



    ax= plt.subplot()
    sns.heatmap(con, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Tahmin Değerleri')
    ax.set_ylabel('Gerçek Değerler')
    ax.set_title('KNN Confusion Matrix') 
    ax.xaxis.set_ticklabels(['Beraberlik', 'Galibiyet', 'Mağlubiyet'])
    ax.yaxis.set_ticklabels(['Beraberlik', 'Galibiyet', 'Mağlubiyet'])
    
    knn_calculated_scores = cs.Calculate_Score_Algorithm(y_test, best_knn_pred)
    
   
    
    
    return_values['KNN Minimum Test Error Rate'] = min(test_error_rate)
    return_values['KNN Minimum Test Error Rate K Value'] = test_error_rate.index(min(test_error_rate))
    return_values['KNN Maximum Accuracy'] = max(acc)
    return_values['KNN Maximum Accuracy K Value'] = acc.index(max(acc))
    return_values['KNN Accuracy Score Best n'] = ac
    return_values['KNN Confusion Matrix'] = con
    return_values['Best KNN Pre Score'] = knn_calculated_scores['Predision Score']
    return_values['Best KNN Recall Score'] = knn_calculated_scores['Recall Score']
    return_values['Best KNN F1 Score'] = knn_calculated_scores['F1 Score'] 
    
    return return_values


def Draw(max_range,y_value,title,xlabel,ylabel):
    plt.figure(figsize=(10,6))
    plt.plot(range(1,max_range),y_value,color='blue',linestyle='dashed',marker='o',
             markerfacecolor='red',markersize='10')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    






























