
from sklearn.ensemble import GradientBoostingClassifier 
from nltk.corpus import wordnet as wn

class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,max_depth=3)

    def extract_features(self, word):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        pos = 0
        means = 0
        for wor in word.split(' '):
            s = [w.pos() for w in wn.synsets(word)]
            pos+=len(set(s))
            means+=len(s)
        #print([len_chars, len_tokens,pos/2,means/10])
        return [len_chars, len_tokens,pos/2,means/10]

    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])
        with open('x.txt','w')as f:
            for x in X:
                line = ''
                for i in x:
                    line += str(i)+' '
                f.write(line.strip()+'\n')
        with open('y.txt','w')as f:
            for y_ in y:
                f.write(str(y_)+'\n')
        self.model.fit(X, y)

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word']))

        return self.model.predict(X)
