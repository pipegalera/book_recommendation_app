

# ----------------------- Content base recommendation system Setup -------------------------#

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import random
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# ----------------------- NLP based Recommendation Engine -------------------------#


user_book = "The Psychology of Money"

def recommendation_engine(user_book, engine = 'CountVectorizer'):
    # Genre specific DataFrame
    genre_user_book = df.loc[df['Title'] == user_book]['Genre'].iloc[0:].tolist()
    df_user = df[df['Genre'].isin(genre_user_book)].reset_index(drop = True)

    # TfidfVectorizer or CountVectorizer
    if (engine == 'CountVectorizer'):
        model = CountVectorizer()

    elif (engine == 'TfidfVectorizer_2'):
        model = TfidfVectorizer(analyzer='word',
                                ngram_range=(2, 2),
                                min_df = 1,
                                stop_words='english')

    elif (engine == 'TfidfVectorizer_3'):
        model = TfidfVectorizer(analyzer='word',
                                ngram_range=(3, 3),
                                min_df = 1,
                                stop_words='english')

    # Model
    model_matrix = model.fit_transform(df_user['Book description'])

    # Similarity matrix
    similarity = cosine_similarity(model_matrix)

    # Keep the index of the book selected
    index_book = df_user.loc[df_user['Title'] == user_book].index.values[0]

    # Check similarity with all the book descriptions
    pairwise_books = list(enumerate(similarity[index_book]))

    # Sort them and search search for their index
    sorted_similar_books = sorted(pairwise_books, key=lambda x: x[1], reverse=True)

    # Eliminate duplicate books
    sorted_similar_books_df = pd.DataFrame(sorted_similar_books)
    sorted_similar_books_df = sorted_similar_books_df.drop_duplicates(subset= 1).iloc[1:,:]
    sorted_similar_books_df
    #   Keep only the top 5 recommendations.

    # Print this top 5 recommendations
    index_book_list = sorted_similar_books_df[0].head(5).tolist()
    similar_books_df = df_user.iloc[index_book_list]
    similar_books_df

    # Print the 5 book covers
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 7), dpi=80)

    for i in range(similar_books_df.shape[0]):
            url_image = similar_books_df['Book image'].iloc[i]
            response = requests.get(url_image)
            book_image = Image.open(BytesIO(response.content))
            ax[i].imshow(book_image)
            ax[i].axis('off')
    fig


recommendation_engine(user_book)
