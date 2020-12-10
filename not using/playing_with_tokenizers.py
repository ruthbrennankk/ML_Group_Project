import nltk
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


nltk.download('punkt')

from sklearn.feature_extraction.text import CountVectorizer

tokenizer = CountVectorizer().build_tokenizer()
print(tokenizer("Here’s example text, isn’t it?"))

from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize

print(WhitespaceTokenizer().tokenize("Here’s example text, isn’t it?"))
print(word_tokenize("Here’s example text, isn’t it?"))
print(tokenizer("likes liking liked"))
print(WhitespaceTokenizer().tokenize("likes liking liked"))
print(word_tokenize("likes liking liked"))

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
tokens = word_tokenize("Here’s example text, isn’t it?")
stems = [stemmer.stem(token) for token in tokens]
print(stems)
tokens = word_tokenize("likes liking liked")
stems = [stemmer.stem(token) for token in tokens]
print(stems)