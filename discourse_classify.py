from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np


label_categories = ["question", "answer", "announcement",
                    "appreciation","agreement", "elaboration",
                    "disagreement", "humor", "negativereaction"]

print('Loading data')
# load data
train_comments = []
train_labels = []
with open('train_comments.csv', 'rb') as f:
    for line in f:
        sep_loc = line.rfind(bytes(',', 'utf-8'))
        train_labels.append(int(line[sep_loc+1:]))
        train_comments.append(line[:sep_loc])

test_comments = []
test_labels = []
with open('test_comments.csv', 'rb') as f:
    for line in f:
        sep_loc = line.rfind(bytes(',', 'utf-8'))
        test_labels.append(int(line[sep_loc+1:]))
        test_comments.append(line[:sep_loc])

# try two types of data representation, bag of words and tf-idf
# try with three classifiers: logistic regression, linear svm, random forest
# do a grid search to do model selection

vectorizers = [CountVectorizer, TfidfVectorizer]
classifiers = [LogisticRegression, LinearSVC, RandomForestClassifier]
param_grid = {LogisticRegression: {'penalty':['l1', 'l2'],
                                   'C':[0.001, 0.01, 0.1, 1, 10]},
              LinearSVC: {'C':[0.001,0.01,0.1,1,10]},
              RandomForestClassifier:{'n_estimators':[100], # fix to 100 trees
                                      'criterion':['gini', 'entropy'],
                                      'min_samples_split':[2, 4, 10],
                                      'n_jobs':[-1] # multi process
                                      }}


for vect in vectorizers:
    print('Extracting features')
    feat_ext = vect(ngram_range=(1,3), # use up to 3-gram
                    max_df=0.9 # use max df to filter stop words
                    )
    train_feat = feat_ext.fit_transform(train_comments)
    test_feat = feat_ext.transform(test_comments)

    int_to_term = {}
    for term in feat_ext.vocabulary_:
        int_to_term[feat_ext.vocabulary_[term]] = term

    for classifier_type in classifiers:
        print('Training {}'.format(classifier_type))
        model = classifier_type()
        # select model using only training data, doing a 3-fold cv on it
        model_selector = GridSearchCV(model,
                                      param_grid[classifier_type],
                                      n_jobs=4,
                                      cv=3)
        model_selector.fit(train_feat, train_labels)
        print('Best classification acc: {}'.format(model_selector.best_score_))

        test_acc = model_selector.score(test_feat, test_labels)
        print('Test classification acc: {}'.format(test_acc))

        if classifier_type == RandomForestClassifier:
            feat_score = model_selector.best_estimator_.feature_importances_
            feat_idx = np.argsort(-feat_score) # sort high to low

            # get top 50 features
            for i in range(50):
                print('Top {}: {}'.format(i+1, int_to_term[feat_idx[i]]))
