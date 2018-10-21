import csv,nltk
from collections import Counter


class Dataset(object):

    def __init__(self, language):
        self.language = language

        trainset_path = "datasets/{}/{}_Train.tsv".format(language, language.capitalize())
        devset_path = "datasets/{}/{}_Dev.tsv".format(language, language.capitalize())
        test_path = "datasets/{}/{}_Test.tsv".format(language, language.capitalize())

        self.trainset,self.bigram_dic= self.read_dataset(trainset_path)
        self.devset, uselessOne= self.read_dataset(devset_path)
        self.testset, uselessTwo= self.read_dataset(devset_path)

    def read_dataset(self, file_path):
        with open(file_path) as file:
            fieldnames = ['hit_id', 'sentence', 'start_offset', 'end_offset', 'target_word', 'native_annots',
                          'nonnative_annots', 'native_complex', 'nonnative_complex', 'gold_label', 'gold_prob']
            reader = csv.DictReader(file, fieldnames=fieldnames, delimiter='\t')

            dataset = []
            total_couple = []
            for sent in reader:
                #if self.language=='english' and sent['gold_label'] == '1':
                    #print(sent['sentence'],'\t',sent['target_word'])
               
                dataset.append(sent)
                l = sent['target_word'].split()
                for word in l:
                    total_couple.extend(nltk.bigrams(word,pad_left=True,pad_right=True))
                
        return dataset, Counter(total_couple)
