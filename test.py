# Imports

import pandas as pd
from collections import Counter
import numpy as np

# Data loading

artists = pd.read_csv("data/artists.dat", delimiter='\t')
tags = pd.read_csv("data/tags.dat", delimiter='\t')
user_artist = pd.read_csv("data/user_artists.dat", delimiter='\t')
user_friends = pd.read_csv("data/user_friends.dat", delimiter='\t')
user_taggedartists = pd.read_csv("data/user_taggedartists.dat", delimiter='\t')
user_taggedartists_time = pd.read_csv("data/user_taggedartists-timestamps.dat", delimiter='\t')
