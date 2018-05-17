import os
import sys

#Sparse matrix implementation
from scipy.sparse import csr_matrix
from Vocab import Vocab
import numpy as np
from collections import Counter

np.random.seed(1)

class IMDBdata:
    def __init__(self, directory, vocab=None):
        """ Reads in data into sparse matrix format """
        #print directory
        pFiles = os.listdir("%s/pos" % directory)
        nFiles = os.listdir("%s/neg" % directory)

        if not vocab:
            self.vocab = Vocab()
        else:
            self.vocab = vocab

        #For csr_matrix (see http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)
        
        self.X_reviews = []
        X_values = []
        X_row_indices = []
        X_col_indices = []
        Y = []
        #Read positive files
        for i in range(len(pFiles)):
            f = pFiles[i]
            lines = ""
            for line in open("%s/pos/%s" %(directory, f), encoding="utf8"):
                lines += line
                wordCounts = Counter([self.vocab.GetID(w.lower()) for w in line.split(" ")])
                for (wordId, count) in wordCounts.items():
                    if wordId >= 0:
                        X_row_indices.append(i)
                        X_col_indices.append(wordId)
                        X_values.append(count)
            Y.append(+1.0)
            self.X_reviews.append(lines)
        
#            print(wordCounts)
#        print(len(num_positive))
        #Read negative files
        for i in range(len(nFiles)):
            f = nFiles[i]
            lines = ""
            for line in open("%s/neg/%s" % (directory, f), encoding="utf8"):
                lines += line
                wordCounts = Counter([self.vocab.GetID(w.lower()) for w in line.split(" ")])
                for (wordId, count) in wordCounts.items():
                    if wordId >= 0:
                        X_row_indices.append(len(pFiles)+i)
                        X_col_indices.append(wordId)
                        X_values.append(count)
            Y.append(-1.0)
            self.X_reviews.append(lines)
        
        self.num_positive_reviews = len(pFiles)
        self.num_negative_reviews = len(nFiles)      
        
        self.X = csr_matrix((X_values, (X_row_indices, X_col_indices)), shape=(max(X_row_indices)+1, self.vocab.GetVocabSize())) 
#        self.count_negative= self.X[len(pFiles):].sum(axis=0)
#        self.count_positive= self.X[:len(pFiles)].sum(axis=0)
        
        self.Y = np.asarray(Y)
        
        self.vocab.Lock()
        
        self.vocab_len = self.X.shape[1]
#        self.total_count_positive = self.count_positive.sum()
#        self.total_count_negative = self.count_negative.sum()
#        self.P_positive= self.count_positive.sum()/ (self.count_positive.sum()+self.count_negative.sum())
#        self.P_negative= self.count_negative.sum()/ (self.count_positive.sum()+self.count_negative.sum())
        #Create a sparse matrix in csr format
        

        index = np.arange(self.X.shape[0])
        np.random.shuffle(index)
        self.X = self.X[index,:]
        self.Y = self.Y[index]

if __name__ == "__main__":
    data = IMDBdata("data/aclImdb/train/")
#    print("vocab_len:")
#    print(data.X.shape[1])
#    print(data.X.sum(axis=0))
###    print(data.X.todense())
#    print("count_positive:")
#    print(data.count_positive)
#    print("count_negative:")
#    print(data.count_negative)
#    print("num_positive_reviews:")
#    print(data.num_positive_reviews)
#    print("num_negative_reviews:")
#    print(data.num_negative_reviews)
#    print("self.total_positive_words:")
#    print(data.count_positive.sum())
#    print("self.total_negative_words:")
#    print(data.count_negative.sum())
#    print("self.P_positive:")
#    print(data.P_positive)
#    print("self.P_negative:")
#    print(data.P_negative)
#    print("self.deno_pos:")
#    print(data.deno_pos.sum())
#    print("self.deno_neg:")
#    print(data.deno_neg.sum())
    
