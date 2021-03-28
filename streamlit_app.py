
# ----------------------------------- Setup ---------------------------------------#

import pandas as pd
import streamlit as st
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide")

# --------------------------------- Data and Variables ----------------------------#

df = pd.read_csv(r'data\good_reads_data_min.csv')

list_books = list(df.Title.unique())
list_books.insert(0, '') #Makes the selection book bar start with nothing
list_books.sort()

# ----------------------------------- Image --------------------------------------#

header_html = "<img src='https://data.whicdn.com/images/259924818/original.gif' class='img-fluid' height= '120' width = '120'>"


# ----------------------------------- Main header --------------------------------#

# spacer_btwn creates a small space between the image and the title
spacer_left, image_column, spacer_btwn, title_column, spacer_right = st.beta_columns((.1, 0.4, 0.1, 1.5, .1))

with image_column:
    st.markdown(header_html, unsafe_allow_html=True,)

with title_column:
    st.title('Book Recommendation App.')

with title_column:
    st.subheader(':book: Web App by [Pipe Galera](https://www.pipegalera.com/)')



# ----------------------------------- Sub header --------------------------------#

spacer_left, sub_header, spacer_right = st.beta_columns((.1, 3.2, .1))

with sub_header:
        st.markdown("Welcome to Pipe's book recommendation app. The app compares the descriptions of more than 30.000 books to give a similar recommendation to the book that you like. It contains 40 book categories, so feel free to type any kind of book.")

        st.markdown("**To begin, please type a book that you liked (or just use any of these three books that I love!).** 👇")

# ----------------------------------- Main selector --------------------------------#

spacer_left, user_book_row , spacer_right = st.beta_columns((.1, 3.2, .1))

with user_book_row:
    default_book = st.selectbox("Try with one of these books:", (
        "The Psychology of Money", "SprawlBall: A Visual Tour of the New Era of the NBA", "Book of Rhymes: The Poetics of Hip Hop"))


    st.markdown("**or**")

    user_book = st.selectbox("Type or select a book that you liked to get personalized recommendations!", list_books, format_func=lambda x: '' if x == '' else x)

    if user_book:
        st.success('Gotcha! 🎉')

    if not user_book:
        user_book = default_book
# ----------------------------------- Recomendation engine  --------------------------------#


st.markdown("---")
spacer_left, title_engine, spacer_right = st.beta_columns(
    (.1, 2, .1))

title_engine.header('Top 5 recommendations based on **{}**.'.format(user_book))

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

    # Keep only the top 5 recommendations.
    index_book_list = sorted_similar_books_df[0].head(5).tolist()
    similar_books_df = df_user.iloc[index_book_list]

    # Print the 5 book covers
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 7), dpi=80)

    for i in range(similar_books_df.shape[0]):
            url_image = similar_books_df['Book image'].iloc[i]
            response = requests.get(url_image)
            book_image = Image.open(BytesIO(response.content))
            ax[i].imshow(book_image)
            ax[i].axis('off')
    fig

    st.table(similar_books_df.iloc[:, :5].assign(hack='').set_index('hack'))

recommendation_engine(user_book)



# ------------------------------------- How the recommendation engine work -----------------#

sections_source = st.beta_expander('📚 What kind of books or sections the app takes to compare books? 📚')

with sections_source:
    st.markdown("The app takes approximatelly 1250 books from the following 40 topics:")

    image_genre = Image.open(r"images\Genres.png")

    st.image(image_genre)

data_source = st.beta_expander('📁 Where the book data come from? 📁')

with data_source:
    st.markdown("The data comes from Goodreads. I have created a Goodreads web scraper that goes through the most popular sections (called *shelves* in Goodreads). This *web scraper* is basically a bot that though the Goodreads website clicking in every book and storing the information relative to each book.")

    scraper = "<center><img src='https://i.imgur.com/KtGRuqK.gif'</center>"
    st.markdown(scraper, unsafe_allow_html=True,)
    st.markdown(" ")
    st.markdown("After the web scraper goes through all the book in a section or multiples sections, the final dataset looks similar with this (but with more books): ")

code_source = st.beta_expander('👨‍💻 Where can I find the code of this project? 👨‍💻')

with code_source:
    st.markdown("All the code is open source and available in [my project github repository](https://github.com/pipegalera/book_recommendation_app).")
