# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:28:55 2022

@author: semih
"""
# ---------------------------------------------------------------
# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import KNN_Algorithm as knnAlgorithm  # Yazdığım KNN
import NaiveBayes_Algorithm as nbA # Yazdığım Naive Bayes
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# FUNCTIONS

def Durum(home_score,away_score):
    result = ""
    if home_score > away_score:
        result = "Galibiyet"
    elif home_score < away_score:
        result = "Mağlubiyet"
    else:
        result = "Beraberlik"
        
    return result


def Tarih_Duzenleme(tarih):
    duzenlenmis_tarih = ""
    x = tarih.split("-",3)
    d = x[2].split(" ",1)
    duzenlenmis_tarih = "{0}-{1}-{2}".format(x[0],x[1],d[0])
    return duzenlenmis_tarih

# ---------------------------------------------------------------



# ---------------------------------------------------------------
# Categorical Nums
categorical_to_nums = { "SEASON": {"2016-2017" : 0, "2017-2018" : 1, "2018-2019" : 2,
                                   "2019-2020" : 3, "2020-2021" : 4, "2021-2022" : 5},
                       
                       "HOME_TEAM": {"Manchester City" : 0, "Club Brugge KV" : 1,
                                    "Paris Saint-Germain" : 2, "RB Leipzig" : 3,
                                    "Atlético Madrid" : 4, "Liverpool FC" : 5,
                                    "FC Porto" : 6, "AC Milan" : 7, "Beşiktaş" : 8,
                                    "Sporting CP" : 9, "AFC Ajax" : 10, 
                                    "Borussia Dortmund" : 11, "FC Sheriff" : 12,
                                    "Inter" : 13, "Shakhtar Donetsk" : 14,
                                    "Real Madrid" : 15, "FC Barcelona" : 16,
                                    "Dinamo Kiev" : 17, "SL Benfica" : 18,
                                    "Bayern München" : 19, "BSC Young Boys" : 20,
                                    "Villarreal CF" : 21, "Atalanta" : 22, 
                                    "Manchester United" : 23, "Sevilla FC" : 24, 
                                    "Lille OSC" : 25, "RB Salzburg" : 26, 
                                    "VfL Wolfsburg" : 27, "Chelsea FC" : 28,
                                    "Malmö FF" : 29, "Zenit St. Petersburg" : 30,
                                    "Juventus" : 31, "Lokomotiv Moskva" : 32,
                                    "Bor. Mönchengladbach" : 33, "Olympiakos Piraeus" : 34,
                                    "Olympique Marseille" : 35, "FC Midtjylland" : 36,
                                    "Stade Rennes" : 37, "FK Krasnodar" : 38,
                                    "Lazio Roma" : 39, "Ferencvárosi TC" : 40,
                                    "İstanbul Başakşehir" : 41, "Galatasaray" : 42,
                                    "Tottenham Hotspur" : 43, "Crvena Zvezda" : 44,
                                    "Dinamo Zagreb" : 45, "Bayer Leverkusen" : 46,
                                    "SSC Napoli" : 47, "KRC Genk" : 48, "Slavia Praha" : 49,
                                    "Olympique Lyon" : 50, "Valencia CF" : 51, 
                                    "AS Monaco" : 52, "PSV Eindhoven" : 53, 
                                    "FC Schalke 04" : 54, "AEK Athen" : 55,
                                    "1899 Hoffenheim" : 56, "Viktoria Plzeň" : 57,
                                    "CSKA Moskva" : 58, "AS Roma" : 59, "FC Basel" : 60,
                                    "Celtic FC" : 61, "RSC Anderlecht" : 62, 
                                    "Qarabağ FK" : 63, "NK Maribor" : 64,
                                    "Spartak Moskva" : 65, "Feyenoord" : 66,
                                    "APOEL Nikosia" : 67, "Arsenal FC" : 68,
                                    "PFC Ludogorets Razgrad" : 69, "FK Rostov" : 70,
                                    "Legia Warszawa" : 71, "FC København" : 72,
                                    "Leicester City" : 73},
                       
                       "AWAY_TEAM": {"Manchester City" : 0, "Club Brugge KV" : 1,
                                    "Paris Saint-Germain" : 2, "RB Leipzig" : 3,
                                    "Atlético Madrid" : 4, "Liverpool FC" : 5,
                                    "FC Porto" : 6, "AC Milan" : 7, "Beşiktaş" : 8,
                                    "Sporting CP" : 9, "AFC Ajax" : 10, 
                                    "Borussia Dortmund" : 11, "FC Sheriff" : 12,
                                    "Inter" : 13, "Shakhtar Donetsk" : 14,
                                    "Real Madrid" : 15, "FC Barcelona" : 16,
                                    "Dinamo Kiev" : 17, "SL Benfica" : 18,
                                    "Bayern München" : 19, "BSC Young Boys" : 20,
                                    "Villarreal CF" : 21, "Atalanta" : 22, 
                                    "Manchester United" : 23, "Sevilla FC" : 24, 
                                    "Lille OSC" : 25, "RB Salzburg" : 26, 
                                    "VfL Wolfsburg" : 27, "Chelsea FC" : 28,
                                    "Malmö FF" : 29, "Zenit St. Petersburg" : 30,
                                    "Juventus" : 31, "Lokomotiv Moskva" : 32,
                                    "Bor. Mönchengladbach" : 33, "Olympiakos Piraeus" : 34,
                                    "Olympique Marseille" : 35, "FC Midtjylland" : 36,
                                    "Stade Rennes" : 37, "FK Krasnodar" : 38,
                                    "Lazio Roma" : 39, "Ferencvárosi TC" : 40,
                                    "İstanbul Başakşehir" : 41, "Galatasaray" : 42,
                                    "Tottenham Hotspur" : 43, "Crvena Zvezda" : 44,
                                    "Dinamo Zagreb" : 45, "Bayer Leverkusen" : 46,
                                    "SSC Napoli" : 47, "KRC Genk" : 48, "Slavia Praha" : 49,
                                    "Olympique Lyon" : 50, "Valencia CF" : 51, 
                                    "AS Monaco" : 52, "PSV Eindhoven" : 53, 
                                    "FC Schalke 04" : 54, "AEK Athen" : 55,
                                    "1899 Hoffenheim" : 56, "Viktoria Plzeň" : 57,
                                    "CSKA Moskva" : 58, "AS Roma" : 59, "FC Basel" : 60,
                                    "Celtic FC" : 61, "RSC Anderlecht" : 62, 
                                    "Qarabağ FK" : 63, "NK Maribor" : 64,
                                    "Spartak Moskva" : 65, "Feyenoord" : 66,
                                    "APOEL Nikosia" : 67, "Arsenal FC" : 68,
                                    "PFC Ludogorets Razgrad" : 69, "FK Rostov" : 70,
                                    "Legia Warszawa" : 71, "FC København" : 72,
                                    "Leicester City" : 73},
                       
                       "STADIUM": { "Etihad Stadium" : 0, "Jan Breydel Stadion" : 1, 
                                    "Parc des Princes" : 2, "Red Bull Arena" : 3, 
                                    "Wanda Metropolitano" : 4, "Anfield" : 5, 
                                    "Estádio do Dragão" : 6, "Giuseppe Meazza" : 7, 
                                    "Vodafone Park" : 8, "Estádio José Alvalade" : 9, 
                                    "Johan Cruijff ArenA" : 10, "Signal Iduna Park" : 11, 
                                    "Sheriff Stadium" : 12, "Olimpiyskyi" : 13, 
                                    "Santiago Bernabéu" : 14, "Spotify Camp Nou" : 15, 
                                    "Estádio da Luz" : 16, "Allianz Arena" : 17, 
                                    "Stadion Wankdorf" : 18, "Estadio de la Cerámica" : 19, 
                                    "Gewiss Stadium" : 20, "Old Trafford" : 21, 
                                    "Ramón Sánchez Pizjuán" : 22, "Stade Pierre Mauroy" : 23, 
                                    "Volkswagen Arena" : 24, "Stamford Bridge" : 25, 
                                    "Nya Malmö Stadion" : 26, "Gazprom Arena" : 27, 
                                    "Allianz Stadium" : 28, "Stade de France" : 29, 
                                    "RŽD-Arena" : 30, "Alfredo Di Stéfano" : 31, 
                                    "Borussia-Park" : 32, "Georgios Karaiskakis" : 33, 
                                    "Vélodrome" : 34, "MCH Arena" : 35, 
                                    "Roazhon Park" : 36, "Krasnodar Stadium" : 37, 
                                    "Olimpico" : 38, "Groupama Aréna" : 39, 
                                    "Puskás Aréna" : 40, "Fatih Terim Stadium" : 41, 
                                    "Arena Națională" : 42, "Nef Stadyumu" : 43, 
                                    "Tottenham Hotspur Stadium" : 44, "Marakana" : 45, 
                                    "Metalist Stadion" : 46, "Maksimir" : 47, 
                                    "BayArena" : 48, "Diego Maradona" : 49, 
                                    "Cegeka Arena" : 50, "Sinobo Stadium" : 51, 
                                    "Groupama Stadium" : 52, "Estadio de Mestalla" : 53, 
                                    "Stade Louis II" : 54, "Wembley Stadium" : 55, 
                                    "Philips Stadion" : 56, "Veltins-Arena" : 57, 
                                    "Spyros Louis (OAKA)" : 58, "PreZero Arena" : 59, 
                                    "Doosan Arena" : 60, "Luzhniki" : 61, 
                                    "St. Jakob-Park" : 62, "VEB Arena" : 63, 
                                    "Celtic Park" : 64, "Lotto Park" : 65, 
                                    "Baku National Stadium" : 66, "Ljudski vrt" : 67, 
                                    "Otkrytie Arena" : 68, "De Kuip" : 69, 
                                    "Neo GSP Stadium" : 70, "Emirates Stadium" : 71, 
                                    "Vasil Levski" : 72, "Vicente Calderón" : 73, 
                                    "Olimp – 2" : 74, "Arena Khimki" : 75, 
                                    "Stadion Wojska Polskiego" : 76, "Telia Parken" : 77, 
                                    "King Power Stadium" : 78, "Millennium" : 79},
                       
                       "Date_Time" : {"15-SEP-21" : 0, "28-SEP-21" : 1, 
                                    "19-OCT-21" : 2, "03-NOV-21" : 3, 
                                    "24-NOV-21" : 4, "07-DEC-21" : 5, 
                                    "14-SEP-21" : 6, "29-SEP-21" : 7, 
                                    "20-OCT-21" : 8, "02-NOV-21" : 9, 
                                    "23-NOV-21" : 10, "08-DEC-21" : 11, 
                                    "09-DEC-21" : 12, "15-FEB-22" : 13, 
                                    "16-FEB-22" : 14, "22-FEB-22" : 15, 
                                    "23-FEB-22" : 16, "08-MAR-22" : 17, 
                                    "09-MAR-22" : 18, "15-MAR-22" : 19, 
                                    "16-MAR-22" : 20, "05-APR-22" : 21, 
                                    "06-APR-22" : 22, "12-APR-22" : 23, 
                                    "13-APR-22" : 24, "26-APR-22" : 25, 
                                    "27-APR-22" : 26, "03-MAY-22" : 27, 
                                    "04-MAY-22" : 28, "28-MAY-22" : 29, 
                                    "21-OCT-20" : 30, "27-OCT-20" : 31, 
                                    "03-NOV-20" : 32, "25-NOV-20" : 33, 
                                    "01-DEC-20" : 34, "09-DEC-20" : 35, 
                                    "20-OCT-20" : 36, "28-OCT-20" : 37, 
                                    "04-NOV-20" : 38, "24-NOV-20" : 39, 
                                    "02-DEC-20" : 40, "08-DEC-20" : 41, 
                                    "16-FEB-21" : 42, "17-FEB-21" : 43, 
                                    "23-FEB-21" : 44, "24-FEB-21" : 45, 
                                    "09-MAR-21" : 46, "10-MAR-21" : 47, 
                                    "16-MAR-21" : 48, "17-MAR-21" : 49, 
                                    "06-APR-21" : 50, "07-APR-21" : 51, 
                                    "13-APR-21" : 52, "14-APR-21" : 53, 
                                    "27-APR-21" : 54, "28-APR-21" : 55, 
                                    "04-MAY-21" : 56, "05-MAY-21" : 57, 
                                    "29-MAY-21" : 58, "18-SEP-19" : 59, 
                                    "01-OCT-19" : 60, "22-OCT-19" : 61, 
                                    "06-NOV-19" : 62, "26-NOV-19" : 63, 
                                    "11-DEC-19" : 64, "17-SEP-19" : 65, 
                                    "02-OCT-19" : 66, "23-OCT-19" : 67, 
                                    "05-NOV-19" : 68, "27-NOV-19" : 69, 
                                    "10-DEC-19" : 70, "18-FEB-20" : 71, 
                                    "19-FEB-20" : 72, "25-FEB-20" : 73, 
                                    "26-FEB-20" : 74, "10-MAR-20" : 75, 
                                    "11-MAR-20" : 76, "07-AUG-20" : 77, 
                                    "08-AUG-20" : 78, "12-AUG-20" : 79, 
                                    "13-AUG-20" : 80, "14-AUG-20" : 81, 
                                    "15-AUG-20" : 82, "18-AUG-20" : 83, 
                                    "19-AUG-20" : 84, "23-AUG-20" : 85, 
                                    "18-SEP-18" : 86, "03-OCT-18" : 87, 
                                    "24-OCT-18" : 88, "06-NOV-18" : 89, 
                                    "28-NOV-18" : 90, "11-DEC-18" : 91, 
                                    "19-SEP-18" : 92, "02-OCT-18" : 93, 
                                    "23-OCT-18" : 94, "07-NOV-18" : 95, 
                                    "27-NOV-18" : 96, "12-DEC-18" : 97, 
                                    "12-FEB-19" : 98, "13-FEB-19" : 99, 
                                    "19-FEB-19" : 100, "20-FEB-19" : 101, 
                                    "05-MAR-19" : 102, "06-MAR-19" : 103, 
                                    "12-MAR-19" : 104, "13-MAR-19" : 105, 
                                    "09-APR-19" : 106, "10-APR-19" : 107, 
                                    "16-APR-19" : 108, "17-APR-19" : 109, 
                                    "30-APR-19" : 110, "01-MAY-19" : 111, 
                                    "07-MAY-19" : 112, "08-MAY-19" : 113, 
                                    "01-JUN-19" : 114, "12-SEP-17" : 115, 
                                    "27-SEP-17" : 116, "18-OCT-17" : 117, 
                                    "31-OCT-17" : 118, "22-NOV-17" : 119, 
                                    "05-DEC-17" : 120, "13-SEP-17" : 121, 
                                    "26-SEP-17" : 122, "17-OCT-17" : 123, 
                                    "01-NOV-17" : 124, "21-NOV-17" : 125, 
                                    "06-DEC-17" : 126, "13-FEB-18" : 127, 
                                    "14-FEB-18" : 128, "20-FEB-18" : 129, 
                                    "21-FEB-18" : 130, "06-MAR-18" : 131, 
                                    "07-MAR-18" : 132, "13-MAR-18" : 133, 
                                    "14-MAR-18" : 134, "03-APR-18" : 135, 
                                    "04-APR-18" : 136, "10-APR-18" : 137, 
                                    "11-APR-18" : 138, "24-APR-18" : 139, 
                                    "25-APR-18" : 140, "01-MAY-18" : 141, 
                                    "02-MAY-18" : 142, "26-MAY-18" : 143, 
                                    "13-SEP-16" : 144, "28-SEP-16" : 145, 
                                    "19-OCT-16" : 146, "01-NOV-16" : 147, 
                                    "23-NOV-16" : 148, "06-DEC-16" : 149, 
                                    "14-SEP-16" : 150, "27-SEP-16" : 151, 
                                    "18-OCT-16" : 152, "02-NOV-16" : 153, 
                                    "22-NOV-16" : 154, "07-DEC-16" : 155, 
                                    "14-FEB-17" : 156, "15-FEB-17" : 157, 
                                    "21-FEB-17" : 158, "22-FEB-17" : 159, 
                                    "07-MAR-17" : 160, "08-MAR-17" : 161, 
                                    "14-MAR-17" : 162, "15-MAR-17" : 163, 
                                    "11-APR-17" : 164, "12-APR-17" : 165, 
                                    "18-APR-17" : 166, "19-APR-17" : 167, 
                                    "02-MAY-17" : 168, "03-MAY-17" : 169, 
                                    "09-MAY-17" : 170, "10-MAY-17" : 171, 
                                    "03-JUN-17" : 172 }
                       
            }
# ---------------------------------------------------------

ds = pd.read_excel("Champions_League_Match_Data.xlsx")
ds['Status'] = ds.apply(lambda x: Durum(x.HOME_TEAM_SCORE, x.AWAY_TEAM_SCORE), axis=1)
print(ds)

print(ds.isnull().sum().sum())  # Kaç satırda boş veri var diye kontrol ediyoruz
print(ds.isnull().sum())  # Hangi öznitelikte kaç tane null değer var diye bakıyoruz.

# Bu veri setimizde eksik veri olmadığından eksik verileri doldurma işlemi uygulamıyoruz.
# -----------------------------------------------------------------------------

# Eğitim ve Test Verisi Ayırma
x = ds.iloc[:,:-1]  # Son sürun hariç tüm satır ve sütunları seçiyoruz. (Bağımsız değişken)
y = ds.iloc[:,-1].values # Tüm satırların yalnızca son sütununu seçiyoruz. (Bağımlı değişken)
y_new = pd.get_dummies(ds['Status'])

y_new2 = LabelEncoder().fit_transform(ds['Status'])




# Veri seti kolon düzenleme
"""
DATE_TIME sütunu kaldırıldı ve Date_Time sütunu olarak yeniden oluşturulup düzenlenmiş veri eklendi.
Ev sahibi takım özelinde maçın Galibiyet, Maülubiyet ve Beraberlik Durumları Status isimli özniteliğe eklendi.
MATCH_ID sütunu kaldırıldı.
"""

"""
new_df = pd.DataFrame(ds)
new_df["Date_Time"] = new_df["DATE_TIME"].apply(Tarih_Duzenleme)
new_df['Status'] = new_df.apply(lambda x: Durum(x.HOME_TEAM_SCORE,x.AWAY_TEAM_SCORE),axis = 1)  # Ev sahibi takım özelinde maçın sonucu
del new_df['MATCH_ID'],new_df['DATE_TIME']  # MATCH_ID ve DATE_TIME sütunlarını kaldırıyoruz
del new_df['PENALTY_SHOOT_OUT']  # İlgili öznitelikteki tüm değerler 0 olduğundan öğrenmeye etki etmeyecektir.
"""


"""
x_new = pd.DataFrame(new_df)
x_new = x_new.replace(categorical_to_nums)
del x_new['Status'], x_new['AWAY_TEAM_SCORE'],x_new['HOME_TEAM_SCORE']
"""
x_new = pd.DataFrame(x)
del x_new['MATCH_ID'], x_new['DATE_TIME'], x_new['HOME_TEAM_SCORE'], x_new['AWAY_TEAM_SCORE'], x_new['PENALTY_SHOOT_OUT']


x_new2 = pd.DataFrame(x)
x_new2 = pd.concat([x_new,pd.get_dummies(x['SEASON']), pd.get_dummies(x['HOME_TEAM']), pd.get_dummies(x['AWAY_TEAM']), pd.get_dummies(x['STADIUM'])],axis=1)

del x_new2['SEASON'], x_new2['HOME_TEAM'], x_new2['AWAY_TEAM'], x_new2['STADIUM']




x_train, x_test, y_train, y_test = train_test_split(x_new2,y_new2, test_size=0.2)
scaler = StandardScaler()
x_train[['ATTENDANCE']] = scaler.fit_transform(x_train[['ATTENDANCE']])
"""
x_train['SEASON'] = LabelEncoder().fit_transform(x_train['SEASON'])
x_train['HOME_TEAM'] = LabelEncoder().fit_transform(x_train['HOME_TEAM'])
x_train['AWAY_TEAM'] = LabelEncoder().fit_transform(x_train['AWAY_TEAM'])
x_train['STADIUM'] = LabelEncoder().fit_transform(x_train['STADIUM'])
"""
x_test[['ATTENDANCE']] = scaler.transform(x_test[['ATTENDANCE']])




scores = knnAlgorithm.KNN_Algorithm(x_train, y_train, x_test, y_test)


#nb = nbA.NB_Algorithm(x_train, y_train, x_test, y_test)

#nbA.NB_Algorithm(x_train, y_train, x_test, y_test)

#new_df = new_df.replace(categorical_to_nums)
#new_df.head()
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------







# AÇIKLAMALAR

"""
UNIQUE DEĞERLER

home_team_unique = ds['HOME_TEAM'].unique().to_list()
away_team_unique = ds['AWAY_TEAM'].unique()
stadium_unique = ds['STADIUM'].unique()
datetime_unique = new_df['Date_Time'].unique()
"""

"""
Kategorik değerleri numerik değerlere dönüştürmek için önce dosyaya yazıyorum sonra da 
kopyalayıp categorical_to_nums dizisine yapıştırıyorum.

file = open("example.txt","w+",encoding="utf8")
yaz = ""
sayac = 0
for i in datetime_unique:
    yaz += "\"{0}\" : {1}, ".format(i, sayac)
    sayac +=1
    if sayac%2==0:
        yaz += "\n"

file.write(yaz)
file.close()
"""


"""
OneHotEncoder Uygulama

#dsHomeDummies = pd.get_dummies(ds['HOME_TEAM'], prefix='Home')
#dsAwayDummies = pd.get_dummies(ds['AWAY_TEAM'], prefix='Away')
#dsStadium = pd.get_dummies(ds['STADIUM'], prefix='Stadium')

#dsSeason = pd.get_dummies(ds['SEASON'],prefix='Season')
#new_df = pd.concat([new_df,dsStadium],axis=1)
#del new_df['STADIUM']
"""



"""
# Kategorik verileri sayısal verilere dönüştürme
onehotencoder = OneHotEncoder()
ds["HOME_TEAM"] = pd.Categorical(ds['HOME_TEAM'])
"""


"""
Önce elimizdeki veriyi x,y olarak ayırıyoruz.
Daha sonra tüm verileri sayısal verilere dönüştürüyoruz. 
Daha sonra x_train,x_test,y_train,y_test olarak parçalıyoruz.


"""


# -----------------------------------------------------------------------------
















    
