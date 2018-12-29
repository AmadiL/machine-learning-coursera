from utils import *
import sys
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import textwrap
import scipy.io as sio
from sklearn import svm

class Ex6Spam:
    def __init__(self):
        self.data = None # Original data
        self.X = None # X matrix
        self.Xtest = None # X test matrix
        self.y = None # y vector
        self.ytest = None # y test vector
        self.m = None  # Number of training examples
        self.n = None  # Number of features
        self.model = None # SVM classificator

    def run(self):
        self.email_preprocessing()
        self.feature_extraction()
        self.train_linear_svm_for_spam_classification()
        self.test_spam_classification()
        self.top_predictors_of_spam()
        self.try_your_own_emails()

    @print_name
    # @pause_after
    def email_preprocessing(self):
        with open('ex6-data/emailSample1.txt') as f:
            file_contents = f.read()
        word_indices = self.process_email(file_contents)
        print("Word indices:")
        print(textwrap.fill(str(word_indices), width=77))

    def process_email(self, email_contents):
        vocab_list = self.get_vocab_list()

        # Lower case
        email_contents = email_contents.lower()
        # Strip all HTML
        email_contents = re.sub('<[^<>]+>', ' ', email_contents)
        # Handle numbers
        email_contents = re.sub('[0-9]+', 'number', email_contents)
        # Handle URLs
        email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
        # Handle email addresses
        email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
        # Handle $ sign
        email_contents = re.sub('[$]+', 'dollar', email_contents)
        # Tokenize
        email_contents = re.split("[ @$/#.-:&*+=[\]?!(){},'\">_<;%\n\r]", email_contents)

        email_contents = list(filter(None, email_contents))
        ps = PorterStemmer()
        email_contents = [ ps.stem(re.sub('[^a-zA-Z0-9]', '', word)) for word in email_contents ]
        word_indices = []
        for word in email_contents:
            try:
                idx = vocab_list.index(word)
                word_indices.append(idx)
            except ValueError:
                pass
        print("==== Processed Email ====")
        print(textwrap.fill(' '.join(email_contents), width=77))
        print("=========================")
        return word_indices

    def get_vocab_list(self):
        with open("ex6-data/vocab.txt") as f:
            vocab_list = [ line.rsplit("\t", 1)[1] for line in f.read().splitlines() ]
        return vocab_list

    @print_name
    # @pause_after
    def feature_extraction(self):
        with open('ex6-data/emailSample1.txt') as f:
            file_contents = f.read()
        word_indices = self.process_email(file_contents)
        features = self.email_features(word_indices)
        print("Length of feature vector: {}".format(len(features)))
        print("Number of non-zero entries: {}".format(np.sum(features)))

    def email_features(self, word_indices):
        return np.isin(np.arange(1899), word_indices).astype(int)

    @print_name
    # @pause_after
    def train_linear_svm_for_spam_classification(self):
        self.data = sio.loadmat("ex6-data/spamTrain.mat")
        self.X = self.data['X']
        self.y = self.data['y']
        self.m, self.n = self.X.shape

        print("Training Linear SVM (Spam Classification)")
        print("(this may take some time) ...")

        C = 0.1
        self.model = svm.SVC(C=C, kernel='linear')
        self.model.fit(self.X, self.y.flatten())
        print("Training Accuracy: {:.3f}".format(self.model.score(self.X, self.y) * 100))

    @print_name
    # @pause_after
    def test_spam_classification(self):
        data_test = sio.loadmat("ex6-data/spamTest.mat")
        self.Xtest = data_test['Xtest']
        self.ytest = data_test['ytest']
        print("Evaluating the trained Linear SVM on a test set ...")
        print("Test Accuracy: {:.3f}".format(self.model.score(self.Xtest, self.ytest) * 100))

    @print_name
    # @pause_after
    def top_predictors_of_spam(self):
        weights = self.model.coef_.flatten()
        # Sort descending
        desc_sort_indices = np.argsort(weights)[::-1][:15]
        vocab_list = self.get_vocab_list()
        print("Top predictors of spam:")
        for i in desc_sort_indices:
            print("{:15} ({:.6f})".format(vocab_list[i], weights[i]))

    @print_name
    # @pause_after
    def try_your_own_emails(self):
        filename = 'ex6-data/spamSample1.txt'
        with open(filename) as f:
            file_contents = f.read()
        word_indices = self.process_email(file_contents)
        x = self.email_features(word_indices)
        p = self.model.predict(x.reshape(1, -1))
        print("Processed {}".format(filename))
        print("Spam Classification: {}".format(*p))
        print("(1 indicates spam, 0 indicates not spam)")

if __name__ == "__main__":
    ex6spam = Ex6Spam()
    sys.exit(ex6spam.run())