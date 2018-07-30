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
                    all_words = word_tokenize(lines.lower())
                    lexicon +=list(all_words)

        lexicon = [lemmatizer.lemmatize() for i in lexicon]
        word_counts = Counter(lexicon)
        l2 = []
        for word in word_counts:
            if 1000 >  word_counts[w] > 50:
                l2.append(w)
        return l2
