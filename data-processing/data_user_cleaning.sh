#!/bin/bash

while IFS= read -r line; do      
    echo "$line" | jq -c '{user_id: .user_id, friends: (.friends | split(","))}' >> filtered_data_user.json
done < yelp_academic_dataset_users.json
