from sklearn.linear_model import LogisticRegression
import nltk


class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model = LogisticRegression()

    def extract_features(self, word, dic):
        word_arr = word.split(' ')
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word_arr)
        couple_num = 0
        couple_arr = []
        for w in word_arr:
            couple_arr.extend(nltk.bigrams(w,pad_left=True,pad_right=True))
        frequency = 0.0
        for couple in couple_arr:
            couple_num+=1
            frequency+=dic[couple]
       
            #print(frequency)
        return [len_chars,len_chars*len_chars,len_chars*len_chars*len_chars,len_tokens*len_tokens,frequency/(couple_num*6000)]

    def train(self, trainset, dic):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word'],dic))
            y.append(sent['gold_label'])

        self.model.fit(X, y)

    def test(self, testset, dic):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word'],dic))

        return self.model.predict(X)
