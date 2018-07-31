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


#classify feature sets

def sample_handling(sample,lexicon,classification):
    featureset = []
    with open(sample,'r') as file:
        contents = file.readlines():
            for line in contents[:no_of_lines]:
                current_words = word_tokenize(line.lower())
                current_words = [lemmatizer.lemmatize(i) for i in current_words]
                features = np.zeros(len(lexicon))
                for word in current_words:
                    if word.lower() in lexicon:
                        index_value = lexicon.index(word.lower())
                        features[index_value] += 1
                features = list(features)
                featureset.append([features,classification])
    return featureset


#create feature sets and labels

def create_feature_sets_and_labels(pos,neg,test_size=0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling('pos.txt',lexicon,[1,0])
    features += sample_handling('neg.txt',lexicon,[0,1])
    random.shuffle(features)
    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])

    test_x = list(features[:,0][:-testing_size:])
    test_y = list(featuers[:,1][-testing_size:])

    return train_x,train_y,test_x,test_y
