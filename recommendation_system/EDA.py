

# ----------------------- Content base recommendation system Setup -------------------------#

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
import altair as alt
alt.renderers.enable('altair_viewer')

# ----------------------- Data Exploration -------------------------#

df = pd.read_csv(r'C:\Users\pipeg\Desktop\storytel_project\data\good_reads_data.csv')


# N-gram distribution for the book description
# For example, a 2-gram is a combination of 2 words
def create_n_gram_plot(column, n_gram):
    # Word Vectorizer
    tf = TfidfVectorizer(ngram_range = (n_gram,n_gram),
                         stop_words = 'english',
                         lowercase = False)
    # Model
    X = column.values.astype(str)
    tf_matrix = tf.fit_transform(X)

    # Word frequency
    vocabulary = tf.vocabulary_.items()
    total_words = tf_matrix.sum(axis = 0)

    freq = [(word, total_words[0, idx]) for word, idx in vocabulary]

    # Tidy column
    freq = sorted(freq, key = lambda x: x[1], reverse = True)
    freq_df = pd.DataFrame(freq)

    # Plot
    fig = alt.Chart(freq_df.head(20),
             title= "Top twenty {}-word combination in the book description".format(n_gram)).mark_bar().encode(
        alt.X(
            '1:Q',
            title = "Distribution of the most used words"
        ),
        alt.Y(
            '0:N',
            title = "Combination of words most used",
            sort='-x'
        ),
        alt.Tooltip('1:Q',
                    format = ".1f"
        )).properties(
            height=alt.Step(17),
            width=600
        ).configure_view(
            strokeWidth = 0

        )


    return fig

create_n_gram_plot(df['Book description'], n_gram = 2)
