#importing the Libraries
import numpy as np
import pandas as pd 

data=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
review=data['Review'][0]
corpus=[]

for i in range(0,1000):
    
    review=re.sub('[^a-zA-Z]',' ',data['Review'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()

y=data.iloc[:,1].values

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Fitting Naive Bayes to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=500,criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)

#checking accuracy of results
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)
