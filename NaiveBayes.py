import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import matplotlib.pyplot as plt 
from imdb import IMDBdata
from Vocab import Vocab
from collections import defaultdict
class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.data = data # training data
        #TODO: Initalize parameters
        # the length of the vocab is equal to the number of columns in the csr_matrix
        self.vocab_len = data.X.shape[1]
        #We call the Train function with the data of csr_matrix X and the array with the labeles
        self.Train(data.X,data.Y)
    # Train model - X are instances, Y are labels (+1 or -1)
    # X and Y are sparse matrices
    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        #The total number of positive reviews coincides with the number of reviews with label 1. The same happens with the negative reviews.
        positive_indices = np.argwhere(Y == 1.0).flatten()
        negative_indices = np.argwhere(Y == -1.0).flatten()
        self.num_positive_reviews = len(positive_indices) 
        self.num_negative_reviews = len(negative_indices)
        
        self.count_positive = defaultdict(int)
        self.count_negative = defaultdict(int)
        #We have to go through all the reviews
        for row in range(0, X.shape[0]):
            review_words = X.getrow(row).nonzero()[1]
            #And have the count the word count according to be positive review or negative.
            for index in range(0, review_words.shape[0]):
                current_word_id = review_words[index]
                count = X[row, review_words[index]]
                label = Y[row]
                if label == 1:
                    self.count_positive[current_word_id] += count
                elif label== -1:
                    self.count_negative[current_word_id] += count
        self.total_positive_words = sum(self.count_positive.values())
        self.total_negative_words = sum(self.count_negative.values())
        #Denominator taking into account the smooth we have to apply
        self.deno_pos = (self.vocab_len*self.ALPHA)+ self.total_positive_words
        self.deno_neg = (self.vocab_len*self.ALPHA)+ self.total_negative_words
        return

    def PredictLabel(self, X):
        #The probability that is positive or negative is equal to the total number positive reviews between the total number
        self.P_positive = self.num_positive_reviews / (self.num_positive_reviews+self.num_negative_reviews)
        self.P_negative = 1-self.P_positive
        pred_labels = []
        #We have to go through all the reviews
        for i in range(0,X.shape[0]):
            review_words = X.getrow(i).nonzero()[1]
            pos_probs = 0
            neg_probs = 0
            #And calculate the probability in each review.
            for index in range(0, review_words.shape[0]):
                current_word_id = review_words[index]
                pos_probs += log((self.count_positive[current_word_id] + self.ALPHA)/self.deno_pos)
                neg_probs += log((self.count_negative[current_word_id] + self.ALPHA)/self.deno_neg)
            pos_prob = log(self.P_positive) + pos_probs
            neg_prob = log(self.P_negative) + neg_probs
            #If the positive probability is greater than the negative probability, it will be positive class.
            if pos_prob > neg_prob:
                pred_labels.append(1.0)
            else:
                pred_labels.append(-1.0)
        return pred_labels

    def LogSum(self, logx, logy):   
        m = max(logx, logy)        
        return m + log(exp(logx - m) + exp(logy - m))

    def PredictProb(self, test, indexes):
        predicted_prob = []
        count = 0
        for i in indexes:
            review_words = test.X.getrow(i).nonzero()[1]
            pos_probs = 0
            neg_probs = 0
            for j in range(0, review_words.shape[0]):
                current_word_id = review_words[j]
                pos_probs += log((self.count_positive[current_word_id] + self.ALPHA)/self.deno_pos)
                neg_probs += log((self.count_negative[current_word_id] + self.ALPHA)/self.deno_neg)
            pos_prob = log(self.P_positive) + pos_probs
            neg_prob = log(self.P_negative) + neg_probs
            denominator = self.LogSum(pos_prob, neg_prob)
            predicted_prob.append(exp(pos_prob - denominator))
            #We show the first 10 reviews on the screen along with the labels and probabilities
            if count < 10:
                count += 1 
                print(test.Y[i], self.PredictLabel(test.X.getrow(i)), exp(pos_prob - denominator), 1-(exp(pos_prob - denominator)), test.X_reviews[i])
        return predicted_prob

    def PredictTreshold(self, test, j):
        predicted_prob = self.PredictProb(test, range(test.X.shape[0]))
        lista = np.arange(0, 1, 0.001) 
        precision = [] 
        recall = [] 
        #We are going to perform the precision recall curve using these values of treshold that we have in the list.
        for threshold in lista:
            thresholdlabel = []
            for i in range(len(predicted_prob)):
                if predicted_prob[i] >  threshold:
                    thresholdlabel.append(1)
                else:
                    thresholdlabel.append(-1)
            #Calculate the confusion matrix for this threshold
            conf_arr = self.Curveprecision(test, thresholdlabel)
            precision.append(self.EvalPrecision(conf_arr[0][0],conf_arr[1][0]))
            recall.append(self.EvalRecall(conf_arr[0][0],conf_arr[0][1]))
        plt.plot(precision,recall,'g') 
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        title = 'Curve precision-recall alpha=' + str(j)
        plt.title(title)
        name = 'precision_recall_studio' + str(j) + '.png'
        plt.savefig(name)
        plt.close()
        return thresholdlabel
    
    def Curveprecision(self, test, tresholdlabel):
        conf_arr = [[0, 0], [0, 0]]
        for i in range(len(tresholdlabel)):
            if test.Y[i] == 1:
                if tresholdlabel[i] == -1:
                    conf_arr[0][1] = conf_arr[0][1] + 1
                else:
                    conf_arr[0][0] = conf_arr[0][0] + 1
            elif test.Y[i] == -1:
                if tresholdlabel[i] == 1:
                    conf_arr[1][0] = conf_arr[1][0] +1
                else:
                    conf_arr[1][1] = conf_arr[1][1] +1
        return conf_arr

    def EvalPrecision(self, truepositive, falsenegative):
        return float(truepositive) / (truepositive + falsenegative)
    
    def EvalRecall(self, truepositive, falsepositive):
        return float(truepositive) / (truepositive + falsepositive)
    
    def Eval(self, test):
        #With the labels obtained and those that really are we obtain the accuracy
        Y_pred = self.PredictLabel(test.X)
        ev = Eval(Y_pred, test.Y)
        return ev.Accuracy()
    
    def printtop20 (self, X):
        weights_positive = {}
        weights_negative = {}
        for i in range(X.GetVocabSize()):
            #We calculate the probabilities of each word taking into account the amount of words 
            #that appear in positive reviews and negative reviews to remove the words of common use
            positive_w = ((self.count_positive[i] + self.ALPHA)/self.deno_pos)/(self.total_positive_words)
            negative_w = ((self.count_negative[i] + self.ALPHA)/self.deno_neg)/(self.total_negative_words)
            weights_positive[i] = (log(positive_w) - log(negative_w))*self.count_positive[i]- self.count_negative[i]
            weights_negative[i] = (log(negative_w) - log(positive_w))*self.count_negative[i]- self.count_positive[i]

        print ('Positive words with weights')
        weights_positive = sorted(weights_positive.items(), key=lambda t: t[1])
        for wordId, value in weights_positive[::-1][:20]:
            currWord = X.GetWord(wordId)
            currWeight = value
            print (currWord , currWeight)
        print ('\n')
        weights_negative = sorted(weights_negative.items(), key=lambda t: t[1])
        print ('Negative words with weights')
        for wordId, value in weights_negative[::-1][:20]:
            currWord = X.GetWord(wordId)
            currWeight = value
            print (currWord , currWeight)


if __name__ == "__main__":
    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print("Reading Test Data")
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)    
    print("Computing Parameters")
    nb = NaiveBayes(traindata, float(sys.argv[2]))
    print("Test Accuracy: ", nb.Eval(testdata))
    nb.PredictTreshold(testdata, 10)
    nb.printtop20(traindata.vocab)
