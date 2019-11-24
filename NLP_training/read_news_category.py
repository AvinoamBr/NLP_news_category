import pandas as pd
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

# data1 = [{"category": "CRIME", "headline": "There Were"},
#          {"category": "ENTERTAINMENT", "headline": "Wil"}]
# dataset = pd.DataFrame(data1)
# file_content = [{"category": "CRIME", "headline": "There Were 2 Mass Shootings In Texas Last Week, But Only 1 On TV", "authors": "Melissa Jeltsen", "link": "https://www.huffingtonpost.com/entry/texas-amanda-painter-mass-shooting_us_5b081ab4e4b0802d69caad89", "short_description": "She left her husband. He killed their children. Just another day in America.", "date": "2018-05-26"},
# {"category": "ENTERTAINMENT", "headline": "Will Smith Joins Diplo And Nicky Jam For The 2018 World Cup's Official Song", "authors": "Andy McDonald", "link": "https://www.huffingtonpost.com/entry/will-smith-joins-diplo-and-nicky-jam-for-the-official-2018-world-cup-song_us_5b09726fe4b0fdb2aa541201", "short_description": "Of course it has a song.", "date": "2018-05-26"}]
#
# news_category = "datasets/News_Category_Dataset_v2.json"
# file_content  =   open(news_category,'r').readlines()
# list_of_dict = []
# for (i,line ) in enumerate(file_content):
#     # if not i%2000: print (i)
#     list_of_dict.append(eval(line))
# # print (file_content[0],type(file_content))
# dataset = pd.DataFrame(list_of_dict)
dataset = pd.read_pickle("datasets/News_Category_Dataset_v2_mod.pkl")
headlines = list(dataset.headline.values)
# print (dataset)
# print (dataset['category'])
# dataset.to_pickle("datasets/News_Category_Dataset_v2_mod.pkl")
# categories = pd.get_dummies(dataset['category'])


matrix = CountVectorizer(max_features=1000)
# headlines = ['There Were 2 Mass Shootings In Texas Last Week, But Only 1 On TV', "Will Smith Joins Diplo And Nicky Jam For The 2018 World Cup's Official Song"]
X = matrix.fit_transform(headlines).todense()
# print (matrix.get_feature_names())
p = pd.DataFrame(X,columns=matrix.get_feature_names())
print (p)

