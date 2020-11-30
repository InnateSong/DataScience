# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:23:51 2020

@author: Deivydas Vladimirovas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

raw_netflix = pd.read_csv("netflix_titles.csv")
raw_columns = raw_netflix.columns
print("List of columns for the raw netflix csv file: \n",list(raw_columns))

#Important columns are taken only
netflix_df = raw_netflix[["show_id","type","title","release_year","rating"]]

#----
#seperating the movies and shows (MOVIES)
netflix_movies = netflix_df[netflix_df["type"]=="Movie"]

#cleaning the MOVIE dataframe (dropping rows if there are any NaN)
clean_n_m = netflix_movies.dropna()

#sorting the MOVIE dataframe
sorted_movies = clean_n_m.sort_values(by="release_year", ascending=False, na_position = 'first')
print("\nNetflix movies dataframe sorted by release year: \n",sorted_movies.head())

#----
#seperating the movies and shows (SHOWS)
netflix_shows = netflix_df[netflix_df["type"]=="TV Show"]

#cleaning the SHOWS dataframe (dropping rows if there are any NaN)
netflix_shows.dropna()

#sorting the MOVIE dataframe
sorted_shows = netflix_shows.sort_values(by="release_year", ascending=False, na_position = 'first')
print("\nNetflix shows datafram sorted by release year: \n",sorted_shows.head())

#----
#getting the ratings from IMDb
title_data = pd.read_csv("data(1).tsv",sep="\t")
#selecting the required columns
title_data_sep = title_data[["tconst","primaryTitle","startYear","genres"]]
title_data_sep['startYear'] = pd.to_numeric(title_data_sep['startYear'],errors='coerce')

ratings_data = pd.read_csv("data.tsv",sep="\t", dtype={'averageRating':'float64'})
print("\nThe data type for average Ratings for movies: \n",ratings_data["averageRating"].dtypes)

#appending a column from two dataframes
title_rating_df = title_data_sep
title_rating_df.insert(loc=2,column='ratings',value=ratings_data["averageRating"])
print("\ndata inserted ratings score of movies to dataframe: \n",title_rating_df.head())

#cleaning the new dataframe from nan
clean_title_rating = title_rating_df.dropna()
#sorting the data by ratings
sorted_clean_t_r = clean_title_rating.sort_values(by='ratings', ascending =False, na_position='first')

#----
#getting the netflix titles and IMDb titles to match
#CREATING A NEW DATAFRAME
data = {'tconst':[],'primaryTitle':[],'ratings':[], 'startYear':[], 'genres':[]}
new_df = pd.DataFrame(data)

user_type = input("Movie or show: ")
if user_type.lower() == "movie":
    #movies
    movies = []
    for i in netflix_movies['title']:
        movies.append(i)
    years = []
    for i in netflix_movies['release_year']:
        years.append(i)
    
    counter = 0 
    for i in movies:
        if counter==50:
            break
        else:
            if i in sorted_clean_t_r["primaryTitle"].values:
                new_df=new_df.append(sorted_clean_t_r[sorted_clean_t_r["primaryTitle"]==i])
                counter = counter + 1
            else:
                continue
            
    user_ratings = float(input("Enter the ratings you want for a movie: "))
    user_years_begin = int(input("Enter the earliest the movie can be: "))
    user_years_end = int(input("Enter the latest the movie can be: "))
    user_title = input("enter the title of the movie you want to watch: ")
    #years
    custom_years = new_df.loc[(new_df["startYear"] >= user_years_begin) & (new_df["startYear"] <= user_years_end)]
    print("\nList of movies based on the custom years: \n",custom_years)
    #ratings
    custom_ratings = new_df.loc[new_df["ratings"]==user_ratings]
    print("\nList of movies based on the custom ratings: \n",custom_ratings)
    #title
    custom_title = new_df.loc[new_df["primaryTitle"]==user_title]
    print("\nList of movies based on the custom title: \n",custom_title)
    
    #----
    #ranking movies
    rank_movie_df = new_df.sort_values(by='ratings', ascending=False)
    clean_rank_movie = rank_movie_df.dropna()
    print(clean_rank_movie.head(10))
    
    #GETTING ALL THE GENRES 
    genres_a = []
    for i in clean_rank_movie.genres.values:
        temp = i.split(',')
        for j in temp:
            if j in genres_a:
                continue
            else:
                genres_a.append(j)

    print(genres_a)
    
    #SORTING BY GENRE (LONG WAY)
    genres_data = {'tconst':[],'primaryTitle':[],'ratings':[], 'startYear':[], 'genres':[]}
    genres_df = pd.DataFrame(genres_data)
    for i in genres_a:
        for j in clean_rank_movie.genres.values:
            if i in j.split(','):
                genres_df = genres_df.append(clean_rank_movie[clean_rank_movie["genres"]==i])
            else:
                continue

    print(genres_df)
    
    """
    rate=[]
    count=0
    for i in genres_df.ratings.values:
        temp_rate = []
        if genres_df.genres. == genres_a[count]:
            temp_rate.append(i)
            continue
        else:
            count=count+1
            rate.append(temp_rate)
        
    
    rate_mean = []
    for i in rate:
        mean = sum(i) / len(i)
        rate_mean.append(mean)
        
    print(rate_mean)
    """
    

    y_pos=np.arange(len(genres_a))
    plt.bar(y_pos,clean_rank_movie.ratings.values, align='center')
    plt.xticks(y_pos, genres_a)
    
    plt.show()
    
    """
    x_genre = clean_rank_movie.genres.values
    y_ratings = clean_rank_movie.ratings.values
    plt.scatter(x_genre,y_ratings)#CHANGE TO BAR CHART
    plt.title("rating-year scatter plot")
    plt.ylabel("ratings")
    plt.xlabel("genres")
    """
    
    #----
    #creating a graph showing the ratings of movies and the year they were released
    x = new_df.startYear.values
    y = new_df.ratings.values
    plt.scatter(x,y)
    plt.title("rating-year scatter plot")
    plt.ylabel("ratings")
    plt.xlabel("year")
    
if user_type.lower() == "show":
    #shows
    shows = []
    for i in netflix_shows['title']:
        shows.append(i)
    years_s = []
    for i in netflix_shows['release_year']:
        years_s.append(i)
    
    counter1 = 0 
    for i in shows:
        if counter1==50:
            break
        else:
            if i in sorted_clean_t_r["primaryTitle"].values:
                new_df=new_df.append(sorted_clean_t_r[sorted_clean_t_r["primaryTitle"]==i])
                counter1 = counter + 1
            else:
                continue           
    
    user_ratings = float(input("Enter the ratings you want for a movie: "))
    user_years_begin = int(input("Enter the earliest the movie can be: "))
    user_years_end = int(input("Enter the latest the movie can be: "))
    user_title = input("enter the title of the movie you want to watch: ")
        
    #years
    custom_years = new_df.loc[(new_df["startYear"] >= user_years_begin) & (new_df["startYear"] <= user_years_end)]
    print("\nList of movies based on the custom years: \n",custom_years)
    #ratings
    custom_ratings = new_df.loc[new_df["ratings"]==user_ratings]
    print("\nList of movies based on the custom ratings: \n",custom_ratings)
    #title
    custom_title = new_df.loc[new_df["primaryTitle"]==user_title]
    print("\nList of movies based on the custom title: \n",custom_title)
    

#----
#COMPARING AVERAGE RATINGS OF MOVIES AND TV SHOWS TO SEE WHICH ARE HIGHER RATED 
#FINDINGS THE RECOMMENDED MOVIES OR SHOWS BASED ON MEAN RATING SCORE ON EVERY GENRE - CREATE A TABLE CHART FOR EVERY GENRE AND SEE WHAT RATINGS ARE ACCORDING


































