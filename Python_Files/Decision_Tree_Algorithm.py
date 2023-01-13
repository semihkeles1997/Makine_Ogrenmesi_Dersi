# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 21:55:50 2022

@author: semih
"""
# Kütüphaneler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import cross_val_score
import Calculate_Score as cs
# -----------------------------------------------------------------------------


def Decision_Tree_Algorithm(X_train, X_test, y_train, y_test, df, mx_depth=0):
    return_values = {}
    
   
    depth = []
    for i in range(3,20):
        clf = tree.DecisionTreeClassifier(max_depth=i)
        # Perform 7-fold cross validation 
        scores = cross_val_score(estimator=clf,X=X_train, y=y_train, cv=10, n_jobs=4)
        depth.append((scores.mean()))
    print(depth)
    
    # Eğer max_depth değeri dışarıdan gelirse onu kabul edecek.
    # Ancak dışarıdan gelmezse 3-20 arasında maksimum cross_val score'u bulunup,
    # max_depth değeri olarak ayarlanacak.
    if mx_depth == 0:
        mx_depth = depth.index(max(depth))+3
    
    print("MX_DEPTH: ",mx_depth)
    decision_tree = DecisionTreeClassifier(max_depth=mx_depth, criterion="entropy")
    decision_tree.fit(X_train, y_train)
    dt_y_pred = decision_tree.predict(X_test)
    
    # Accuracy score
    dt_acc = accuracy_score(y_test, dt_y_pred)
    
    # Confusion matrix
    dt_cm = confusion_matrix(y_test, dt_y_pred)
    
    # Confusion matrix plot
    plot_confusion_matrix(decision_tree,X_test,y_test)
    plt.show()
      
    # Visualizing Confusing Matrix using Heatmap
    ax= plt.subplot()
    sns.heatmap(dt_cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
      
    # labels, title and ticks
    ax.set_xlabel('Tahmin Değerleri')
    ax.set_ylabel('Gerçek Değerler')
    ax.set_title('Decision Tree Confusion Matrix') 
    ax.xaxis.set_ticklabels(['Beraberlik', 'Galibiyet', 'Mağlubiyet'])
    ax.yaxis.set_ticklabels(['Beraberlik', 'Galibiyet', 'Mağlubiyet'])
    
    text_representation = tree.export_text(decision_tree)
    print(text_representation)
  
    
    fig = plt.figure(figsize=(50,25))
    
    ex = tree.plot_tree(decision_tree, feature_names=df.columns.drop(['Status']), 
                        class_names=df['Status'],filled=True)
    
   
    dt_calculated_scores = cs.Calculate_Score_Algorithm(y_test, dt_y_pred)
   
   
    
    return_values['Decision Tree Accuracy Score'] = dt_acc
    return_values['Decision Tree Confusion Matrix'] = dt_cm
    return_values['Decision Tree Precision Score'] = dt_calculated_scores['Predision Score']
    return_values['Decision Tree Recall Score'] = dt_calculated_scores['Recall Score']
    return_values['Decision Tree F1 Score'] = dt_calculated_scores['F1 Score']
    return_values['Decision Tree mx_depth'] = mx_depth
    
    return return_values
    
    
    
    
    
    
    
    
    
    
    