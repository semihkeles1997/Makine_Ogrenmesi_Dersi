# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 20:17:59 2022

@author: semih
"""

# Kütüphaneler
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import Calculate_Score as cs
# -----------------------------------------------------------------------------


def NB_Algorithm(X_train, y_train, X_test, y_test):
    return_values = {}
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    nb_y_pred = classifier.predict(X_test)

    # Hata Matrisi Oluşturma
    nb_cm = confusion_matrix(y_test, nb_y_pred)

    #accuracy score
    nb_accuracy = accuracy_score(y_test, nb_y_pred)

    # Confusion matrix plot
    plot_confusion_matrix(classifier,X_test,y_test)
    plt.show()



    ax= plt.subplot()
    sns.heatmap(nb_cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Tahmin Değerleri')
    ax.set_ylabel('Gerçek Değerler')
    ax.set_title('Naive Bayes Confusion Matrix') 
    ax.xaxis.set_ticklabels(['Beraberlik', 'Galibiyet', 'Mağlubiyet'])
    ax.yaxis.set_ticklabels(['Beraberlik', 'Galibiyet', 'Mağlubiyet'])
    plt.show()
    
    
    nb_calculated_scores = cs.Calculate_Score_Algorithm(y_test, nb_y_pred)
    
    
    
    return_values['Naive Bayes Confusion Matrix'] = nb_cm
    return_values['Naive Bayes Accuracy Score'] = nb_accuracy
    return_values['Naive Bayes Precision Score'] = nb_calculated_scores['Predision Score']
    return_values['Naive Bayes Recall Score'] = nb_calculated_scores['Recall Score']
    return_values['Naive Bayes F1 Score'] = nb_calculated_scores['F1 Score']
    
    return return_values
    
# -----------------------------------------------------------------------------
    
    































