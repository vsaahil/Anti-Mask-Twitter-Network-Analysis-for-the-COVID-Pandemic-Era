# -*- coding: utf-8 -*-
"""
GROUP PROJECT 

SOCIAL NETWORK ANALYSIS
"""

import numpy as np
import pandas as pd
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#####--------------------------------------------------------Anti-Mask tweets for March 26- April 3----------------------------------------------------------------------------
twitter_df = pd.read_excel(r"C:\Users\shiva\Downloads\full_df_for_network-36k.xlsx")
twitter_df.head()

for i in range(len(twitter_df)):
  twitter_df['target'].fillna(twitter_df['user'], inplace=True)

import networkx as nx
G = nx.DiGraph()
lst=list()
for (a,b) in zip(twitter_df['user'], twitter_df['target']):
    lst.append((a,b))
G.add_edges_from(lst)
nx.draw(G, with_labels=True)
#plt.figure(figsize = (12,12))
#plt.show()

nx.write_gexf(G, r"C:\Users\shiva\Downloads\test.gexf")

betweenness_centrality=pd.DataFrame.from_dict(nx.betweenness_centrality(G), orient='index').reset_index()
betweenness_centrality.columns = ['screen_name', 'betweenness_centrality']
print(betweenness_centrality.sort_values(by = 'betweenness_centrality', ascending = False).reset_index(drop = True))
#betweenness_centrality.to_csv(r"C:\Users\shiva\Downloads\betweenness.csv")

degree_centrality=pd.DataFrame.from_dict(nx.degree_centrality(G), orient='index').reset_index()
degree_centrality.columns = ['screen_name', 'degree_centrality']
print(degree_centrality.sort_values(by = 'degree_centrality', ascending = False).reset_index(drop = True))

closeness_centrality = pd.DataFrame.from_dict(nx.closeness_centrality(G), orient='index').reset_index()
closeness_centrality.columns = ['screen_name', 'closeness_centrality']
print(closeness_centrality.sort_values(by = 'closeness_centrality', ascending = False).reset_index(drop = True))


#finding influencers
twitter_df['sum'] = twitter_df['follower_count'] + twitter_df['listed_count']

df = pd.read_csv(r"C:\Users\shiva\Documents\gephclusters.csv")
df1 = betweenness_centrality
df1 = df1.rename(columns={"screen_name": "Label"})
df2 = closeness_centrality
df2 = df2.rename(columns={"screen_name": "Label"})
df3 = degree_centrality
df3 = df3.rename(columns={"screen_name": "Label"})

df4 = df.merge(df1, on='Label')
df5 = df4.merge(df2, on='Label')
df6 = df5.merge(df3, on='Label')

df7 = twitter_df
df7 = df7.rename(columns={"user": "Label"})

df8 = df7.merge(df6, on = 'Label', how ='left' )
#only keeping unique users
df9 = df8.drop_duplicates(subset = ["Label"])


x = df9[['betweenness_centrality','closeness_centrality', 'degree_centrality' ]]
y = df9[['sum']]



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

    
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=0)
model = forest_reg.fit(X_train, y_train)
clv_predictions_RF = model.predict(X_test)
features_rf_c1 = model.feature_importances_
#making dataframe of features along with the values of their importance 
feature_random_forest_c1 = pd.DataFrame(list(zip(x.columns,forest_reg.feature_importances_)), columns = ['predictor','feature importance'])

df6['betweenness_centrality'].apply(lambda x: float(x))
df6['closeness_centrality'].apply(lambda x: float(x))
df6['degree_centrality'].apply(lambda x: float(x))


df6['score'] = 0.18*(df6['betweenness_centrality']) + 0.50*(df6['closeness_centrality']) + 0.32*(df6['degree_centrality'])
influencers = df6[['Label', 'score']]
influencers = influencers.sort_values(by = 'score', ascending = False)
influencers.head(10)



###-------------------------------------------------------------Anti-vaccination analysis--------------------------------------------------

vacc_df = pd.read_excel(r"C:\Users\shiva\Downloads\full_df-antivaxx-final.xlsx",sheet_name = 'march2020')

for i in range(len(vacc_df)):
  vacc_df['target'].fillna(vacc_df['source'], inplace=True)

import networkx as nx
G1 = nx.DiGraph()
lst1=list()
for (a,b) in zip(vacc_df['source'], vacc_df['target']):
    lst1.append((a,b))
G1.add_edges_from(lst1)
nx.draw(G1, with_labels=True)
#plt.figure(figsize = (12,12))
#plt.show()

betweenness_centrality1=pd.DataFrame.from_dict(nx.betweenness_centrality(G1), orient='index').reset_index()
betweenness_centrality1.columns = ['screen_name', 'betweenness_centrality']
print(betweenness_centrality1.sort_values(by = 'betweenness_centrality', ascending = False).reset_index(drop = True))
#betweenness_centrality.to_csv(r"C:\Users\shiva\Downloads\betweenness.csv")

degree_centrality1=pd.DataFrame.from_dict(nx.degree_centrality(G1), orient='index').reset_index()
degree_centrality1.columns = ['screen_name', 'degree_centrality']
print(degree_centrality1.sort_values(by = 'degree_centrality', ascending = False).reset_index(drop = True))

closeness_centrality1 = pd.DataFrame.from_dict(nx.closeness_centrality(G1), orient='index').reset_index()
closeness_centrality1.columns = ['screen_name', 'closeness_centrality']
print(closeness_centrality1.sort_values(by = 'closeness_centrality', ascending = False).reset_index(drop = True))

#merging all metrics in one dataset
df10 = betweenness_centrality1.merge(closeness_centrality1, on = 'screen_name')
df11= df10.merge(degree_centrality1, on = 'screen_name')
df11 = df11.rename(columns={"screen_name": "source"})


#finding influencers
vacc_df['sum'] =  vacc_df['followersCount'] + vacc_df['listedCount']
df12 = vacc_df.merge(df11, on = "source", how = 'left')


#keeping uniquse users
df13 = df12.drop_duplicates(subset = ["source"])


x1 = df13[['betweenness_centrality','closeness_centrality', 'degree_centrality' ]]
y1= df13[['sum']]



from sklearn.model_selection import train_test_split

X_train1,X_test1,y_train1,y_test1=train_test_split(x1,y1,test_size=0.30,random_state=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train1)
X_test1 = scaler.transform(X_test1)

    
from sklearn.ensemble import RandomForestRegressor

forest_reg1 = RandomForestRegressor(random_state=0)
model1 = forest_reg1.fit(X_train1, y_train1)
clv_predictions_RF1 = model1.predict(X_test1)
features_rf_c11 = model1.feature_importances_
#making dataframe of features along with the values of their importance 
feature_random_forest_c11 = pd.DataFrame(list(zip(x1.columns,forest_reg1.feature_importances_)), columns = ['predictor','feature importance'])

df11['betweenness_centrality'].apply(lambda x: float(x))
df11['closeness_centrality'].apply(lambda x: float(x))
df11['degree_centrality'].apply(lambda x: float(x))


df11['score'] = 0.14*(df11['betweenness_centrality']) + 0.75*(df11['closeness_centrality']) + 0.11*(df11['degree_centrality'])
influencers1 = df11[['source', 'score']]
influencers1 = influencers1.sort_values(by = 'score', ascending = False)
influencers1.head(10)

####---------------------------------------------------------- Historical Anti-Mask----------------------------------------------------------------------

mask_df = pd.read_excel(r"C:\Users\shiva\Downloads\full_df (1).xlsx",sheet_name = 'final')

for i in range(len(mask_df)):
  mask_df['target'].fillna(mask_df['source'], inplace=True)

for i in range(len(mask_df)):
  mask_df['followersCount'].fillna(0, inplace=True)
  
for i in range(len(mask_df)):
  mask_df['listedCount'].fillna(0, inplace=True)

import networkx as nx
G2 = nx.DiGraph()
lst2=list()
for (a,b) in zip(mask_df['source'], mask_df['target']):
    lst2.append((a,b))
G2.add_edges_from(lst2)
nx.draw(G2, with_labels=True)
#plt.figure(figsize = (12,12))
#plt.show()

nx.write_gexf(G2, r"C:\Users\shiva\Downloads\testmask.gexf")

betweenness_centrality2=pd.DataFrame.from_dict(nx.betweenness_centrality(G2), orient='index').reset_index()
betweenness_centrality2.columns = ['screen_name', 'betweenness_centrality']
print(betweenness_centrality2.sort_values(by = 'betweenness_centrality', ascending = False).reset_index(drop = True))
#betweenness_centrality.to_csv(r"C:\Users\shiva\Downloads\betweenness.csv")

degree_centrality2=pd.DataFrame.from_dict(nx.degree_centrality(G2), orient='index').reset_index()
degree_centrality2.columns = ['screen_name', 'degree_centrality']
print(degree_centrality2.sort_values(by = 'degree_centrality', ascending = False).reset_index(drop = True))

closeness_centrality2 = pd.DataFrame.from_dict(nx.closeness_centrality(G2), orient='index').reset_index()
closeness_centrality2.columns = ['screen_name', 'closeness_centrality']
print(closeness_centrality2.sort_values(by = 'closeness_centrality', ascending = False).reset_index(drop = True))

#merging all metrics in one dataset
df20 = betweenness_centrality2.merge(closeness_centrality2, on = 'screen_name')
df21= df20.merge(degree_centrality2, on = 'screen_name')
df21 = df21.rename(columns={"screen_name": "source"})


#finding influencers
mask_df['sum'] =  mask_df['followersCount'] + mask_df['listedCount']
df22 = mask_df.merge(df21, on = "source", how = 'left')


#keeping uniquse users
df23 = df22.drop_duplicates(subset = ["source"])


x2 = df23[['betweenness_centrality','closeness_centrality', 'degree_centrality' ]]
y2= df23[['sum']]



from sklearn.model_selection import train_test_split

X_train2,X_test2,y_train2,y_test2=train_test_split(x2,y2,test_size=0.30,random_state=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train2 = scaler.fit_transform(X_train2)
X_test2 = scaler.transform(X_test2)

    
from sklearn.ensemble import RandomForestRegressor

forest_reg2 = RandomForestRegressor(random_state=0)
model2 = forest_reg2.fit(X_train2, y_train2)
clv_predictions_RF2 = model2.predict(X_test2)
features_rf_c12 = model2.feature_importances_
#making dataframe of features along with the values of their importance 
feature_random_forest_c12 = pd.DataFrame(list(zip(x2.columns,forest_reg2.feature_importances_)), columns = ['predictor','feature importance'])

df21['betweenness_centrality'].apply(lambda x: float(x))
df21['closeness_centrality'].apply(lambda x: float(x))
df21['degree_centrality'].apply(lambda x: float(x))


df21['score'] = 0.25*(df21['betweenness_centrality']) + 0.47*(df21['closeness_centrality']) + 0.28*(df23['degree_centrality'])
influencers2 = df21[['source', 'score']]
influencers2 = influencers2.sort_values(by = 'score', ascending = False)
influencers2.head(10)
