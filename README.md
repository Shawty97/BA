# Bachelorarbeit - [TITEL]

## Implementation
### Removing Duplicates
maybe not applicable as we dont merge 2 datasets → so no duplicates possible?

### Cleaning
#### Convert to lower case
```python
some_string.lower()
```

#### Tokenization
```python
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

# word tokenization
tokenized_docs = [word_tokenize(d) for d in raw_docs]

# sentence tokenization
sent_token = [sent_tokenize(d) for d in raw_docs]
```

#### Punctuation Removal
```python
import re

punctuation_re = f'[{re.escape(string.punctuation)}]'
tokenized_docs_no_punctuation = []

for review in tokenized_docs:
    new_review = []
    for token in review:
        new_token = regex.sub('', token)
        if new_token != '':
            new_review.append(new_token)

    tokenized_docs_no_punctuation.append(new_review)
```

#### Removing Stopwords
```python
from nltk.corpus import stopwords

tokenized_docs_no_stopwords = []
for doc in tokenized_docs_no_punctuation:
    new_term_vector = []
    for word in doc:
        if word not in stopwords.words('english'):
            new_term_vector.append(word)

    tokenized_docs_no_stopwords.append(new_term_vector)
```

#### Stemming and Lemmatization
```python
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
wordnet = WordNetLemmatizer()

preprocessed_docs = []

for doc in tokenized_docs_no_stopwords:
    final_doc = []
    for word in doc:
        final_doc.append(porter.stem(word))
        final_doc.append(wordnet.Lemmatize(word))

    preprocessed_docs.append(final_doc)
```
