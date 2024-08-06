#!/bin/bash

num_lines=$(wc -l < yelp_academic_dataset_review.json) 


for (( i=1; i<=num_lines; i++ )); do 
    review=$(sed "${i}q;d" yelp_academic_dataset_review.json | tr -d '\n\r' | sed 's/\\n//g' | sed 's/\\r//g' | sed 's/\\t//g' | sed 's/\\\\//g' | sed 's/\\"//g')
    echo "$review" | jq -c '{user_id: .user_id, business_id: .business_id, stars: .stars, word_count: (.text | split(" ") | length)}' >> filtered_data_review.json 
done
