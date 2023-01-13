# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 12:51:23 2022

@author: semih
"""

# Kütüphaneler
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# -----------------------------------------------------------------------------

def Calculate_Score_Algorithm(y_test,y_pred):
    return_values = {}
    
    """
     Precision Score
     Kesinlik (Precision) Positive olarak tahminlediğimiz değerlerin 
     gerçekten kaç adedinin Positive olduğunu göstermektedir.
     Precision = TP / (TP+FP)
   """
    pre_score = precision_score(y_test, y_pred, average='weighted')
    
    """
         Recall Score
         Duyarlılık (Recall) ise Positive olarak tahmin etmemiz gereken işlemlerin 
         ne kadarını Positive olarak tahmin ettiğimizi gösteren bir metriktir.
         Recall = TP / (TP+FN)
         TP : True Positive
         FN : False Negative
    """
    rec_score = recall_score(y_test, y_pred, average='weighted')
    
    """
         F1 Score değeri bize Kesinlik (Precision) ve Duyarlılık (Recall) 
         değerlerinin harmonik ortalamasını göstermektedir.
         F1 = 2 * (precision * recall) / (precision + recall)
    """
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    
    return_values['Predision Score'] = pre_score
    return_values['Recall Score'] = rec_score
    return_values['F1 Score'] = f1
    
    return return_values
     
     
     
     
     
     
     
     