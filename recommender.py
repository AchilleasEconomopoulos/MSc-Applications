### IMPORTS ###

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
from termcolor import colored


########################################################################################################


### DATASET LOADING ###

print(colored("Loading datasets and creating the structures...","yellow"))
artists = pd.read_csv("data/artists.dat", delimiter='\t')
tags = pd.read_csv("data/tags.dat", delimiter='\t')
users_artists = pd.read_csv("data/user_artists.dat", delimiter='\t')
users_taggedartists = pd.read_csv("data/user_taggedartists.dat", delimiter='\t')

# Not used
users_friends = pd.read_csv("data/user_friends.dat", delimiter='\t')
users_taggedartists_time = pd.read_csv("data/user_taggedartists-timestamps.dat", delimiter='\t')
print(colored("Done.\n", "green"))


########################################################################################################


### DATA PREPROCESSING ###

# We'll consider an artistID as valid if:
#   1. They've had minimal interaction with any userID
#   2. We have metadata for that artistID
valid_artists = set(users_artists['artistID']).intersection(set(artists['id']))

# Drop any rows with invalid artistIDs from the user-artist-tag matrix
#   i.e. We can't do CBF with an artist we don't have listening time for or that we can't visualize later on
users_taggedartists = users_taggedartists[users_taggedartists['artistID'].isin(valid_artists)]
users_artists = users_artists[users_artists['artistID'].isin(valid_artists)]


# Making sure we have metadata for tags (for visualization later)
valid_tags = set(users_taggedartists['tagID'])
users_taggedartists = users_taggedartists[users_taggedartists['tagID'].isin(valid_tags)]


########################################################################################################


### ARTIST, USER, TAG IDs ###

# Starting off, look into the user-artist-tag table to extract artist and tag IDs for CBF
cb_artist_ids = set(users_taggedartists['artistID'])
tag_ids = set(users_taggedartists['tagID'])

# Mapping between the tag_id actual values and their indexes in a list
tagmap = {tag_id: tag_idx for tag_idx, tag_id in enumerate(tag_ids)}


# Do the same for CF. Also the final userset comes from this matrix.
#   i.e. the system was trained and tested on this userset's splits
cf_artist_ids = set(users_artists['artistID'])
user_ids = set(users_artists['userID'])

# Then cross-check cf_artists with the artist table (need metadata to visualize)
cf_artist_ids = cf_artist_ids.intersection(set(artists['id']))

tag_N = len(tag_ids)
artists_tags_dict = {k: np.full(tag_N,np.nan) for k in cb_artist_ids}


########################################################################################################


### ARTISTS-TAGS DATAFRAME ###

# While it is definetly a better option to save the dataframe as a file and just load it during runtime
#   its size (~115MB) poses some complexity at the moment.

# artists_tags_df = pd.read_csv('./data/artists_tags.csv')

grouped = (users_taggedartists.groupby(['artistID','tagID']).size().to_dict())

print(colored("Loading tag counts...", "yellow"))
for (artist_id,tag_id) , count in tqdm(grouped.items()):
    artists_tags_dict[artist_id][tagmap[tag_id]] = count
print(colored("Done.\n", "green"))

print(colored("Calculating TF for each tag-artist...", "yellow"))
for artist_id, raw_tag_counts in tqdm(artists_tags_dict.items()):
    artists_tags_dict[artist_id] = raw_tag_counts/np.nanmax(raw_tag_counts)
print(colored("Done.\n", "green"))

# Dict -> DF -> numpy Array for better calculations
intermediate_df = pd.DataFrame(data=artists_tags_dict)
array = np.array(intermediate_df)

print (colored("Finally calculating TFIDF for each tag-artist...", "yellow"))
N = len(cb_artist_ids)
for idx, tag_tfs in tqdm(enumerate(array)):
    idf = np.log(N/np.sum(~np.isnan(tag_tfs)))
    array[idx] = tag_tfs * idf
print(colored("Done.\n", "green"))


# Back to DF for interpretability
artists_tags_df = pd.DataFrame(data=array.transpose(), index=list(cb_artist_ids))
artists_tags_df.columns = list(tag_ids)


# Mapping the list index of the artistIDs to the actual values
cb_artist_reverse_map = {idx : artist_id for idx, artist_id in enumerate(cb_artist_ids)}


########################################################################################################


### USER-ARTIST-LISTENING COUNTS ###

# Dict to keep each user's interactions

print (colored("Collecting all user-artist interactions...", "yellow"))
user_weights = {}
for idx, row in tqdm(users_artists.iterrows()):
    artist_id = row['artistID']
    user_id = row['userID']
    weight = row['weight']

    log_weight = np.log1p(weight)   # Log scaling to keep the impact of very high weights 

    if(user_id in user_weights.keys()):
        user_weights[user_id]['weights'][artist_id] = log_weight
    else:
        user_weights[user_id] = {
            'weights': {
                artist_id: log_weight
            }
        }
print(colored("Done.\n", "green"))


########################################################################################################


### BUILD / LOAD USER PROFILES ###

# Same filesize issue here.

# For faster execution of the application, run the code below once to save the user_profiles on the disk,
#   then comment the code out and uncomment the following command.

# user_profiles = np.load('./data/user_profiles.npy', allow_pickle=True)
# user_profiles = user_profiles.item()


# ## BUILD PROFILES ##
user_profiles = {}
print(colored("Building user profiles...","yellow") )
for user_id in tqdm(user_weights.keys()):
    user_profile = np.zeros(len(tag_ids))
    for artist_id, weight in user_weights[user_id]['weights'].items():

        # artist has to be tagged
        if(artist_id in cb_artist_ids):
            artist_profile = np.nan_to_num(artists_tags_df.loc[artist_id],0)
            user_profile += weight.item() * artist_profile

    user_profiles[user_id] = user_profile.reshape(1,-1)
np.save('./data/user_profiles.npy', user_profiles, allow_pickle=True)
# print(colored("Done.\n", "green"))


########################################################################################################


### INITIALIZE THE CF's DATAFRAME ###

# Initialize the artist-user interactions with NaNs
cf_users_weights = {user_id: np.full(len(cf_artist_ids),np.nan) for user_id in user_ids}

artistmap = {artist_id:idx for idx,artist_id in enumerate(cf_artist_ids)}

# Fill in the corresponding cells with the user-artist log transformed weights (from the train dataset)

print(colored("Building the artists-users-interactions matrix...","yellow"))
for idx, row in tqdm(users_artists.iterrows()):
    artist_id = row['artistID']
    user_id = row['userID']
    weight = row['weight']

    cf_users_weights[user_id][artistmap[artist_id]] = np.log1p(weight)

# Convert dict to DF for easy kNN calculation
cf_df = pd.DataFrame(cf_users_weights,index=list(cf_artist_ids))
print(colored("Done.\n", "green"))


########################################################################################################


### LOAD THE SIMILARITY AND NEIGHBOR MATRICES AND TRANSLATE THEM TO DATAFRAMES ###

# Here the filesizes are ~7MB which is more managable, so we'll load them
# sklearn's k-NearestNeighbors algorithm was used to create the matrices, using k=50
k_param=50

print(colored("Loading similarities and neighbors for the artists...","yellow"))
similarities = np.load('./data/similarities.npy', allow_pickle=True)
neighbor_indices = np.load('./data/neighbors.npy', allow_pickle=True)

cf_artist_map = {idx : artist_id for idx, artist_id in enumerate(cf_artist_ids)}

mapped_neighbor_indices = np.vectorize(cf_artist_map.get)(neighbor_indices)

neighbor_df = pd.DataFrame(
    mapped_neighbor_indices,
    columns=[f'neighbor_{i+1}' for i in range(k_param)],
    index=cf_df.index
)

similarity_df = pd.DataFrame(
    similarities,
    columns=[f'similarity_{i+1}' for i in range(k_param)],
    index=cf_df.index
)
print(colored("Done.\n", "green"))


########################################################################################################


### RECOMMENDERS ###

def recommend_cbf(user_id,k=1,new_only=True):
    if(user_id in user_ids):
        recommendations = { 'artist_ids': [], 'similarities': []}
        user_profile = user_profiles[user_id]

        similarities = cosine_similarity(user_profile.reshape(1,-1),artists_tags_df.fillna(0))   # Returns cos similarities with every row
        top_sim = np.argsort(similarities[0])[::-1]                                              # Sorts the indexes in descending order

        count = 0
        i = 0

        while count < k and i<len(top_sim):
            idx = top_sim[i]
            artist_id = cb_artist_reverse_map[idx]

            # Choose whether the recommendation is something that the user has never interacted with
            if (new_only):
                if (artist_id not in user_weights[user_id]['weights'].keys()):
                    recommendations['artist_ids'].append(artist_id)
                    recommendations['similarities'].append(similarities[:,idx])
                    count+=1

            # Otherwise recommendations may contain artists the user has already interacted with
            else:
                recommendations['artist_ids'].append(artist_id)
                recommendations['similarities'].append(similarities[:,idx])
                count+=1

            i+=1

        return recommendations
    else:
        return None
    
def recommend_cf(user_id,k=1,new_only=True):
    if(user_id in user_ids):
        user_dict = user_weights[user_id]['weights']
        predictions = []
        neighbours_used = []
        recommendations = {'artist_ids': [], 'predictions':[], 'neighbours_used':[]}
        for artist_id in cf_artist_ids:
            if(new_only and artist_id in user_dict.keys()):
                predictions.append(-100)
                neighbours_used.append(-1)
            else:
                neighbors = neighbor_df.loc[artist_id]
                similarities = similarity_df.loc[artist_id]

                nbr_contributions = []

                for idx, nbr_artist in enumerate(neighbors):
                    if(nbr_artist) in user_dict.keys():
                        nbr_contributions.append(user_dict[nbr_artist] * similarities.iloc[idx])

                pred_value = 0
                if(len(nbr_contributions) > 2):
                    pred_value = np.sum(nbr_contributions)/len(nbr_contributions)

                predictions.append(pred_value)
                neighbours_used.append(len(nbr_contributions))

        top_pred_indices = np.argsort(predictions)[::-1]

        for idx in top_pred_indices[:k]:
            recommendations['artist_ids'].append(cf_artist_map[idx])
            recommendations['predictions'].append(predictions[idx])
            recommendations['neighbours_used'].append(neighbours_used[idx])
            
        return recommendations
    else:
        return None
    
def recommend_hybrid(user_id, k=1, new_only=True):
    
    cf_rec = recommend_cf(user_id,k,new_only)
    
    cbf_rec = recommend_cbf(user_id,k,new_only)

    num_of_interactions = np.sum(~np.isnan(cf_df[user_id]))
    # If no collaborative data available, use pure content-based


    # If CF doesn't work OR the user hasn't had any interactions OR CF isn't very confident about its predictions,
    #   then use CBF IF available for that user
    if not cf_rec or num_of_interactions<5 or cf_rec['predictions'][0] < 3.5:
        if cbf_rec:
            return cbf_rec
    
    else:
        return cf_rec
    


if __name__ == "__main__":
    print(colored("The recommendation system is ready to use!","black","on_green"))
    print()

    # Helps with indexing
    artists.set_index('id', inplace=True)

    while(True):
        chosen_user_id = int(input("Type in the user ID you want recommendations for: "))

        chosen_user_artists = users_artists[users_artists['userID'] == chosen_user_id].sort_values('weight', ascending=False)
        
        chosen_user_artists = chosen_user_artists.head(5)

        try: 
            print()
            print(colored("Here are your Top 5 Artists:","green"))

            entries = []
            for idx, row in chosen_user_artists.iterrows():
                artist_id = int(row['artistID'])
                listening_time = int(row['weight'])

                artist_info = artists.loc[artist_id]

                artist_name = artist_info['name']
                url = artist_info['url']

                entries.append([artist_name, listening_time, url])

            for idx,row in enumerate(entries):
                print("{}. {: <20} {: <20} {: <20}".format(idx+1,*row))


            # Recommendations
            print()
            k = int(input("Type in the amount of recommendations you'd like to get: "))
            recommendations = recommend_hybrid(chosen_user_id, k)

            rec_artists = recommendations['artist_ids']
            
            rec_entries = []
            print()
            print(colored("Recommendations:", "green"))
            for idx, rec_artist_id in enumerate(rec_artists):
                artist_info = artists.loc[rec_artist_id]
                artist_name = artist_info['name']
                url = artist_info['url']

                rec_entries.append([artist_name, url])

            for idx,row in enumerate(rec_entries):
                print("{}. {: <20} {: <20}".format(idx+1,*row))
        
        except:
            print(colored("An error occured. Please try another user.","red"))

        print()
        print("----------------------------------------------")
