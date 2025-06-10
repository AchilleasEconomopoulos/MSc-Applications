1. Dataset download από https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
2. Τοποθέτηση των .dat αρχείων σε ένα φάκελο ./data
3. pip install -r requirements.txt
4. python recommender.py

```
Αναφέρεται και στα σχόλια του recommender.py ότι το runtime μπορεί να επιταχυνθεί μετά την πρώτη εκτέλεση.
1. Comment out τον κώδικα που δημιουργεί τα user profiles
2. Uncomment τα commands για να τα κάνει load από τη μνήμη.
```