import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()
no_of_lines = 10000000


def create_lexicon(pos,neg):
        lexicon = []
        for file in [pos,neg]:
            with open(file,'r') as f:
                contents = f.readlines()
                for lines in contents[:no_of_lines]:
                    #all_words = word_tokenize(l)
                    #lexicon +=list(all_words)
