from typing import List
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data= pd.read_csv("/content/movie_finalized_data.csv")
data.head()

cv=CountVectorizer()
cv_matrix =cv.fit_transform(data['combined'])
similarity=cosine_similarity(cv_matrix)

def recommend_movie(movie):
  if movie not in data['movie_title'].unique():
    return []
  else:
    i=data.loc[data['movie_title'] == movie].index[0]
    list= List(enumerate(similarity[i]))
    list= sorted(list, key=lambda x: x[1], reverse=True)
    list=  list[1:11]
    result = []
    for i in range(len(list)):
      a=list[i][0]
      result.append(data['movie_title'][a])
      return result

