import json
from collections import Counter
import pandas as pd
import networkx as nx
import time

cities = set(("Philadelphia", "Tucson", "Tampa", "Indianapolis", "Nashville", "New Orleans", "Reno", "Edmonton", "Saint Louis", "Santa Barbara", "Boise", "St. Louis"))

# extract business ids corresponding to metropolitan areas
def extract_city_data(json_file_path):
    
    data = {}

    with open(json_file_path, 'r') as file:

        for line in file:
            obj = json.loads(line)

            business_id = obj.get("business_id", None)
            city = obj.get("city", None)

            if city in cities:
                if city == "St. Louis":
                    city = "Saint Louis"
                data[business_id] = city

    return data

# extract relevant data from json file
def extract_review_data(json_file_path):

    # get metropolitan area map
    print("Getting metropolitan area map")
    metropolitan_areas = extract_city_data('data-processing/yelp_academic_dataset_business.json')

    ratings_data = {}
    city_data = {}
    wordcount_data = {}
    frequency = {}
    num_iters = 0

    print("Starting data filtering")
    with open(json_file_path, 'r') as file:

        start_time = time.time()

        for line in file:
            obj = json.loads(line)
            num_iters += 1

            user_id = obj.get('user_id', None)
            business_id = obj.get('business_id', None)
            if business_id not in metropolitan_areas:
                if num_iters % 100000 == 0:
                    time_elapsed = time.time()
                    print("Time elapsed: ", time_elapsed - start_time, "seconds")
                    start_time = time.time()
                continue
            city = metropolitan_areas[business_id]
            stars = obj.get('stars', None)
            wordcount = sum(Counter(obj.get('review', None)).values())

            city_data[user_id] = city

            if user_id in frequency and business_id in frequency[user_id]:
                freq = frequency[user_id][business_id]
                curr_rating = ratings_data[user_id][business_id]
                curr_wordcount = wordcount_data[user_id][business_id]
                ratings_data[user_id][business_id] = (curr_rating * freq + stars) / (freq + 1)
                wordcount_data[user_id][business_id] = (curr_wordcount * freq + wordcount) / (freq + 1)
                frequency[user_id][business_id] = freq + 1
            else:
                if user_id in frequency:
                    ratings_data[user_id][business_id] = stars
                    wordcount_data[user_id][business_id] = wordcount
                    frequency[user_id][business_id] = 1
                else:
                    ratings_data[user_id] = {business_id : stars}
                    wordcount_data[user_id] = {business_id : wordcount}
                    frequency[user_id] = {business_id : 1}
            
            if num_iters % 100000 == 0:
                time_elapsed = time.time()
                print("Time elapsed: ", time_elapsed - start_time, "seconds")
                start_time = time.time()

    print("Done with data filtering, starting conversion to dataframe")

    df = pd.DataFrame([ratings_data, wordcount_data, city_data]).transpose()
    df.columns = ['ratings', 'wordcount', 'metropolitan_area']
    df.index.name = 'user_id'

    return df

json_file_path_review = 'data-processing/yelp_academic_dataset_review.json'
df = extract_review_data(json_file_path_review)

grouped = df.groupby('metropolitan_area')

separate_dfs = {}
for feature, group in grouped:
    separate_dfs[feature] = group.copy()

for feature, separate_df in separate_dfs.items():
    print(f"DataFrame for Feature {feature}:")
    print(separate_df)
    print()

# extract friendship data
def extract_friendship_data(json_file_path):

    data = {}
    #num_iters = 0

    with open(json_file_path, 'r') as file:

        start_time = time.time()

        for line in file:
            obj = json.loads(line)

            user_id = obj.get("user_id", None)
            friends = obj.get("friends", None)
            data[user_id] = friends.split(',')

            # num_iters += 1
            # if num_iters % 100000 == 0:
            #     time_elapsed = time.time()
            #     print("Time elapsed: ", time_elapsed - start_time, "seconds")
            #     start_time = time.time()

    return data

# json_file_path_review = 'data-processing/yelp_academic_dataset_review.json'
# # ratings_data, wordcount_data = extract_review_data(json_file_path_review)

# json_file_path_friendship = "data-processing/yelp_academic_dataset_user.json"
# friendship_data = extract_friendship_data(json_file_path_friendship)
# print(friendship_data["qVc8ODYU5SZjKXVBgXdI7w"])
