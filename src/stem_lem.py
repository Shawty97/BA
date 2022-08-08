import json
from pathlib import Path

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# stem
porter = PorterStemmer()
wordnet = WordNetLemmatizer()

# paths
curent_directory = Path(__file__).parent
data_source = curent_directory / "data" / "tweets_cleaned.json"
data_target_stemmed = curent_directory / "data" / "tweets_stemmed.json"
data_target_lemmatized = curent_directory / "data" / "tweets_lemmatized.json"

data_lemmatized = []

with open(data_source, encoding="utf-8") as file_in:
    tweets_stemmed = json.load(file_in)

tweets_lemmatized = tweets_stemmed.copy()

for i in range(len(tweets_stemmed)):
    for j in range(len(tweets_stemmed[i]["text"])):
        tweets_stemmed[i]["text"][j] = porter.stem(tweets_stemmed[i]["text"][j])
        tweets_lemmatized[i]["text"][j] = wordnet.lemmatize(tweets_lemmatized[i]["text"][j])

with open(data_target_stemmed, mode="w", encoding="utf-8") as file_out:
    json.dump(tweets_stemmed, fp=file_out, indent=2)

with open(data_target_lemmatized, mode="w", encoding="utf-8") as file_out:
    json.dump(tweets_lemmatized, fp=file_out, indent=2)
