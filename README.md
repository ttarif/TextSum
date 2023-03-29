# Text Summarization using TextRank and LSA
This code uses Natural Language Processing techniques to perform text summarization of a given URL or text. It leverages the TextRank algorithm and Latent Semantic Analysis (LSA) algorithm for summarization.

# Getting Started
To use this code, you need to have the following libraries installed:

requests
beautifulsoup4
nltk
spacy
scikit-learn
You can install these libraries using pip. For example:

```python
pip install requests beautifulsoup4 nltk spacy scikit-learn
```

# Usage
The summarize_text function can be used to summarize text. It takes two parameters:

* text: The text to be summarized. It can be a URL or a plain text.
* n_sentences: The number of sentences in the summary. Default value is 3.
Here's an example usage:

```python
import requests
from bs4 import BeautifulSoup
from summarize import summarize_text

url = 'https://www.example.com'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
text = soup.get_text()
summary = summarize_text(text, n_sentences=4)
print(summary)
# This will print a summary of the text.
```

# Text Preprocessing
The preprocess_text function is used for text preprocessing. It takes one parameter:

* text: The text to be preprocessed.
* 
This function performs the following steps:

* Tokenization
* Lemmatization
* Removing stop words and punctuations

# Summarization

The summarize_text function uses two algorithms for summarization:

* TextRank algorithm
* Latent Semantic Analysis (LSA) algorithm

The TextRank algorithm is based on the PageRank algorithm used by Google. It calculates the importance of a sentence based on the number of other sentences that refer to it. The sentences with the highest score are selected for the summary.

The LSA algorithm is a technique used in natural language processing for dimensionality reduction of high-dimensional text data. It is used to extract the most important concepts from the text. In this code, the LSA algorithm is used to select the most important sentences for the summary.
