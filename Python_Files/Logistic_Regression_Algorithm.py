# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:46:59 2022

@author: semih
"""
# Kütüphaneler
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import Calculate_Score as cs
# -----------------------------------------------------------------------------

def Logistic_Regression_Algorithm(X_train, X_test, y_train, y_test):
    return_values = {}
    logreg = LogisticRegression(random_state=16)
    logreg.fit(X_train, y_train)
    logreg_y_pred = logreg.predict(X_test)

    # Confusion matrix
    logreg_cm = confusion_matrix(y_test, logreg_y_pred)

    # Accuracy score
    logreg_acc = accuracy_score(y_test, logreg_y_pred)


    
    # Confusion matrix plot
    plot_confusion_matrix(logreg,X_test,y_test)
    plt.show()
    
    # Visualizing Confusing Matrix using Heatmap
    ax= plt.subplot()
    sns.heatmap(logreg_cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    
    # labels, title and ticks
    ax.set_xlabel('Tahmin Değerleri')
    ax.set_ylabel('Gerçek Değerler')
    ax.set_title('Logistic Regression Confusion Matrix') 
    ax.xaxis.set_ticklabels(['Beraberlik', 'Galibiyet', 'Mağlubiyet'])
    ax.yaxis.set_ticklabels(['Beraberlik', 'Galibiyet', 'Mağlubiyet'])
    
    
    logreg_calculated_scores = cs.Calculate_Score_Algorithm(y_test, logreg_y_pred)
    
    return_values['Logistic Regression Confusion Matrix'] = logreg_cm
    return_values['Logistic Regression Accuracy Score'] = logreg_acc
    return_values['Logistic Regression Predision Score'] = logreg_calculated_scores['Predision Score']
    return_values['Logistic Regression Recall Score'] = logreg_calculated_scores['Recall Score']
    return_values['Logistic Regression F1 Score'] = logreg_calculated_scores['F1 Score']
    
    return return_values
    
    
    
    