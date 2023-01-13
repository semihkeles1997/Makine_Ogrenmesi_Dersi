# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:04:50 2022

@author: semih
"""

# ---------------------------------------------------------------
# Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import KNN_Algorithm as knn_Algorithm
import NaiveBayes_Algorithm as nb_Algorithm
import Logistic_Regression_Algorithm as logrec_Algorithm
import Decision_Tree_Algorithm as dt_Algorithm
# ---------------------------------------------------------------

# ---------------------------------------------------------------


df_original = pd.read_csv("Champions_League_Match_Data_Duzenlenmis.csv",sep=';') # sep parametresi neye göre parçalayacağını belirtiyor.

df = df_original.copy()

df.rename(columns={'HOME_TEAM'  :  "Home_Team",
                   'AWAY_TEAM'  :  'Away_Team',
                   'STADIUM'    :  'Stadium',
                   'ATTENDANCE' :  'Attendance',
                   'SEASON'     :  'Season',
                   'Home'       :  'Home_Team_Market_Value',
                   'Away'       :  'Away_Team_Market_Value'}, inplace=True)




print(df.Home_Team.value_counts())





# Veri Görselleştirme

# Galibiyet - Beraberlik - Mağlubiyet Dağılım Grafiği
df.Status.value_counts().plot(kind = 'pie', autopct = '%.2f%%' , ylabel="")  # pie : pasta grafiği, autopct : yüzdelik sayı değerleri. İkisi de zorunlu değil.
plt.show()
#df.Status.value_counts().plot()   # Çizgi grafiği
#df.Status.value_counts().plot(kind='pie', autopct='.1f')

#sns.barplot(x = df.Home, y=df.Away, data=df)  # Bunun için en az bir değerin sayısal olması gerekli.
#sns.heatmap(df.Status,annot=df.Status.rank(axis='columns'))

plt.bar(df.Status.unique(),df.Status.value_counts(), color='maroon', 
        width=0.4)
plt.title("2016-2022 Yılları Arasındaki Maç Sonuçları Grafiği")
plt.show()



df.Status.value_counts().plot(kind='bar')

seasons_unique = {}
print(seasons_unique)
for i in df.Season.unique():
    gt, bt, mt = 0,0,0
    sayac = 0
    for x in df.Season:
        if x == i:
            if df.Status[sayac] == "Galibiyet":
                gt = gt+1   # Galibiyet Sayısı
            elif df.Status[sayac] == "Beraberlik":
                bt = bt+1   # Beraberlik Sayısı
            elif df.Status[sayac] == "Mağlubiyet":
                mt = mt+1   # Mağlubiyet Sayısı
        sayac = sayac+1
        seasons_unique[i] = {'Sezon' : i, 'Galibiyet': gt, 
                             'Beraberlik': bt, 'Mağlubiyet' :mt}
        
print("Season {0}: Galibiyet: {1}, Beraberlik: {2}, Mağlubiyet: {3}".
      format(seasons_unique['2016-2017']['Sezon'],
             seasons_unique['2016-2017']['Galibiyet'],
             seasons_unique['2016-2017']['Beraberlik'],
             seasons_unique['2016-2017']['Mağlubiyet']))
#

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(15, 8))
 
# set height of bar
Galibiyet = [seasons_unique['2016-2017']['Galibiyet'], 
             seasons_unique['2017-2018']['Galibiyet'], 
             seasons_unique['2018-2019']['Galibiyet'],
             seasons_unique['2019-2020']['Galibiyet'],
             seasons_unique['2020-2021']['Galibiyet'],
             seasons_unique['2021-2022']['Galibiyet']]  # IT
Beraberlik = [seasons_unique['2016-2017']['Beraberlik'], # ECE
             seasons_unique['2017-2018']['Beraberlik'], 
             seasons_unique['2018-2019']['Beraberlik'],
             seasons_unique['2019-2020']['Beraberlik'],
             seasons_unique['2020-2021']['Beraberlik'],
             seasons_unique['2021-2022']['Beraberlik']]
Maglubiyet = [seasons_unique['2016-2017']['Mağlubiyet'],
             seasons_unique['2017-2018']['Mağlubiyet'],  # CSE
             seasons_unique['2018-2019']['Mağlubiyet'],
             seasons_unique['2019-2020']['Mağlubiyet'],
             seasons_unique['2020-2021']['Mağlubiyet'],
             seasons_unique['2021-2022']['Mağlubiyet']] # CSE
 
# Set position of bar on X axis
br1 = np.arange(len(Galibiyet))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, Galibiyet, color ='r', width = barWidth,
        edgecolor ='grey', label ='Galibiyet')
plt.bar(br2, Beraberlik, color ='g', width = barWidth,
        edgecolor ='grey', label ='Beraberlik')
plt.bar(br3, Maglubiyet, color ='b', width = barWidth,
        edgecolor ='grey', label ='Mağlubiyet')
 
# Adding Xticks
plt.xlabel('Season', fontweight ='bold', fontsize = 15)
plt.ylabel('Status', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(Galibiyet))],
        ['2016-2017', '2017-2018', '2018-2019', 
         '2019-2020', '2020-2021', '2021-2022'])
 
plt.legend()
plt.show()
#

# -----------------------------------------------------------------------------




# Verileri numeric hâle getirme

#df['Status'] = df.Status.map({'Galibiyet' : 1, 'Beraberlik' : 0, 'Mağlubiyet' : 2})
le = LabelEncoder()
ohe = OneHotEncoder()

#season_ohe = ohe.fit_transform(df[['Season']]).toarray()  # OneHotEncoder kullanırken toarray() dönüşümü yapıyoruz. Bir de dizinin içinde dizi şeklinde alıyoruz. Yani df[['istenen_sutun]] gibi.

df = pd.concat([df,pd.get_dummies(df['Season'], prefix='Season'), pd.get_dummies(df['Home_Team'],prefix='Home_Team'), pd.get_dummies(df['Away_Team'],prefix='Away_Team'), pd.get_dummies(df['Stadium'],prefix='Stadium')],axis=1)
del df['Home_Team'], df['Away_Team'], df['Stadium'], df['Season'], df['HOME_TEAM_SCORE'], df['AWAY_TEAM_SCORE']


print(df.Status.value_counts())
# -----------------------------------------------------------------------------


# Train ve Test Veri Setini Ayarlama
X = df.drop(['Status'], axis = 1)
y = df['Status']
#y = ohe.fit_transform(df[['Status']]).toarray()
y = le.fit_transform(y.ravel())

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=0)

# Eğitim ve Test Veri Setlerinin İstatistikleri
test_veri_seti_degerler = {'Galibiyet' : 0, 'Beraberlik' : 0, 'Mağlubiyet' : 0}
for test in y_test:
    if test == 0:
        test_veri_seti_degerler['Beraberlik'] = test_veri_seti_degerler['Beraberlik'] + 1
    elif test == 1:
        test_veri_seti_degerler['Galibiyet'] = test_veri_seti_degerler['Galibiyet'] + 1
    else:
        test_veri_seti_degerler['Mağlubiyet'] = test_veri_seti_degerler['Mağlubiyet'] + 1

    

egitim_veri_seti_degerler = {'Galibiyet' : 0, 'Beraberlik' : 0, 'Mağlubiyet' : 0}
for egitim in y_train:
    if egitim == 0:
        egitim_veri_seti_degerler['Beraberlik'] = egitim_veri_seti_degerler['Beraberlik'] + 1
    elif egitim == 1:
        egitim_veri_seti_degerler['Galibiyet'] = egitim_veri_seti_degerler['Galibiyet'] + 1
    else:
        egitim_veri_seti_degerler['Mağlubiyet'] = egitim_veri_seti_degerler['Mağlubiyet'] + 1



def Graph_Pie(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d})".format(pct, absolute)


pie_chart_test_data = np.array([test_veri_seti_degerler['Galibiyet'],
                             test_veri_seti_degerler['Beraberlik'],
                             test_veri_seti_degerler['Mağlubiyet']])
plt.pie(pie_chart_test_data, autopct=lambda pct: Graph_Pie(pct, pie_chart_test_data),
                                  labels=['Galibiyet','Beraberlik','Mağlubiyet'])
plt.title("Test Seti İstatistikler")
plt.show()


pie_chart_train_data = np.array([egitim_veri_seti_degerler['Galibiyet'],
                                 egitim_veri_seti_degerler['Beraberlik'],
                                 egitim_veri_seti_degerler['Mağlubiyet']])
plt.pie(pie_chart_train_data, autopct=lambda pct: Graph_Pie(pct,pie_chart_train_data),
        labels=['Galibiyet','Beraberlik','Mağlubiyet'])
plt.title("Eğitim Seti İstatistikler")
plt.show()
# -----------------------------------------------------------------------------



"""
#MinMaxScaler ile Best n = 79, Best n accuracy = 0.5872...
min_max_scaler = MinMaxScaler()
X_train[['Home_Team_Market_Value']] = min_max_scaler.fit_transform(X_train[['Home_Team_Market_Value']].to_numpy())
X_train[['Away_Team_Market_Value']] = min_max_scaler.fit_transform(X_train[['Away_Team_Market_Value']].to_numpy())
X_train[['Attendance']] = min_max_scaler.fit_transform(X_train[['Attendance']].to_numpy())
"""

"""
#StandardScaler ile Best n = 79, Best n accuracy = 0.5872...
std_scaler = StandardScaler()
X_train['Home_Team_Market_Value'] = std_scaler.fit_transform(X_train[['Home_Team_Market_Value']].to_numpy())
X_train['Away_Team_Market_Value'] = std_scaler.fit_transform(X_train[['Away_Team_Market_Value']].to_numpy())
X_train['Attendance'] = std_scaler.fit_transform(X_train[['Attendance']].to_numpy())
"""

"""
Herhangi bir normalizasyon işlemi uygulanmazsa
Best n = 56, Best n accuracy = 0.59060...
"""

print("COL: ",df_original.columns.drop(['Status','HOME_TEAM_SCORE','AWAY_TEAM_SCORE']))
print("gnd Y: ",df_original.Status.unique())


# KNN
knn_values = knn_Algorithm.KNN_Algorithm_Semih(X_train, X_test, y_train, y_test, 100)

# Logistic Regression
logistic_regression_values = logrec_Algorithm.Logistic_Regression_Algorithm(X_train, X_test, y_train, y_test)

# Naive Bayes
naive_bayes_values = nb_Algorithm.NB_Algorithm(X_train, y_train, X_test, y_test)


# Decision Tree
# Eğer mx_depth değerini gönderirsem onu alacak. 
# Göndermezsem maksimum cross_val_score değerini bulup o depth değerini alacak.
# cross_val değeri 10 olarak belirlendi. 
decision_tree_values = dt_Algorithm.Decision_Tree_Algorithm(X_train, X_test, y_train, y_test, df=df)








# -----------------------------------------------------------------------------













































