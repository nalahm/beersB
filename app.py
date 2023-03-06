
import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

# Read the beers and only consider rows without NA
beers = pd.read_csv("beers.csv")
breweries = pd.read_csv("breweries.csv")
beers = beers.dropna(axis=0, how = 'any') 
breweries = breweries[['name', 'state']]
breweries.columns = ['brewery_name', 'state']
beers_breweries = pd.merge(beers,breweries, left_on='brewery_id', right_index=True)

beers_breweries['name_state'] = beers_breweries['name'].str.cat(beers_breweries['state'], sep=' ')
beers_breweries['name_state_style'] = beers_breweries['name_state'].str.cat(beers_breweries['style'], sep=' ')


vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,3))

X = vectorizer.fit_transform(beers_breweries['name_state'])
encoder = LabelEncoder()
y = encoder.fit_transform(beers_breweries['brewery_name'])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

nb = MultinomialNB()
nb.fit(x_train, y_train)

transformer = TfidfTransformer(smooth_idf=False, sublinear_tf=True)
tfidf = transformer.fit_transform(X)
x2_train, x2_test, y2_train, y2_test = train_test_split(tfidf, y, test_size=0.2)

nb2 = MultinomialNB()
nb2.fit(x2_train, y2_train)

# Streamlit app starts here
st.title("Predict Brewery from Beer Name")

# Get user input
user_input = st.text_input("Enter a beer name:")

if user_input:
    # Transform user input using the same vectorizer and transformer
    user_input_transformed = transformer.transform(vectorizer.transform([user_input]))
    
    # Make prediction using both classifiers
    prediction = encoder.inverse_transform(nb.predict(user_input_transformed))[0]
    prediction2 = encoder.inverse_transform(nb2.predict(user_input_transformed))[0]
    
    # Display prediction results
    st.write("Using CountVectorizer: ", prediction)
    st.write("Using TfidfTransformer: ", prediction2)
