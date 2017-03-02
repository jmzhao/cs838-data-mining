import os
import re
import functools
import collections
import logging
from pprint import pprint
import csv
import numpy as np
import sklearn as sk
import sklearn.tree
import sklearn.model_selection
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics

logger = logging.Logger('feature_vector')

TRAINING_DATA_DIRS = ["../dataset/labeled_texts/examples/"]#fortune500_wiki/"]
TESTING_DATA_DIRS  = ["../dataset/labeled_texts/examples/"]#fortune500/"]
TRAINING_DATA_DIRS = ["../dataset/labeled_texts/fortune500_wiki/"]
TESTING_DATA_DIRS  = ["../dataset/labeled_texts/fortune500/"]

STOP_LIST = frozenset(s.strip() for s in open("SmartStoplist.txt"))
PROPER_NOUN_LIST =frozenset(s.strip() for s in open("proper_nouns.txt")) 
SKIP_WORDS = ('of', 'and', 'the', 'to')


class Preprocesser :
    TABLE = {
        '{' : '', '}' : '', ## eliminate {}
        "’": "'", "'""'s'" : '', "s'" : 's', ## eliminate 's
    }
    preprocess = functools.reduce(lambda f, kv : lambda s : f(s).replace(*kv), TABLE.items(), lambda s : s)

class FeatureExtractor :
    FEATURES = (
        'n-words', #'n-characters',
        'in-stop-list',
        'all-capitalized', 'all-inintial-capitalized',
        'no-pre-inintial-capitalized', 'no-aft-inintial-capitalized',
        'no-skip-boundary',
        'common-ends', 'common-pres',
        'contain-hyphen', 'contain-ampersand',
        'contain-seps', 'contain-digit', 'contain-strange',
        'sentence-start',
        )
    r_after = re.compile(r'\bcorporation|\bcorp\b|\bincorporation|\binc\b|\bcompany|\bco\b|\bllc\b')
    r_before = re.compile(r'\bacqui|\bbuy|\bbought|\bcompetit|\bformer|\bmerg|\bown|\brival|\bsell|\bsold')
    r_digit = re.compile(r'\d')

    @staticmethod
    def extract_features(pres, words, afts) :
        s = ' '.join(words)
        p = ' '.join(pres)
        return {
            'n-words' : len(words),
            'n-characters' : len(s),
            'in-stop-list' : len(words) == 1 and words[0].lower() in STOP_LIST,
            'all-capitalized' : all(w.isupper() or w in SKIP_WORDS for w in words),
            'all-inintial-capitalized' : all(w[0].isupper() or w in SKIP_WORDS for w in words),
            'no-pre-inintial-capitalized' : len(pres) == 0 or not pres[-1][0].isupper(),
            'no-aft-inintial-capitalized' : len(afts) == 0 or not afts[-1][0].isupper(),
            'no-skip-boundary' : words[0] in SKIP_WORDS or words[-1] in SKIP_WORDS,
            'common-ends' : FeatureExtractor.r_after.search(s.lower()) is not None,
            'common-pres' : FeatureExtractor.r_before.search(p.lower()) is not None,
            'contain-hyphen' : '-' in s,
            'contain-ampersand' : '&' in s,
            'contain-digit' : FeatureExtractor.r_digit.search(s) is not None,
            'contain-seps' : any(c in s for c in ',.'),
            'contain-strange' : any(c in s for c in '()[]%/\\<>^_'),
            'sentence-start' : len(pres) == 0 or pres[-1][-1] in '.!?',
        }


def load_data(DATA_DIRS) :
    fnames = list()
    for data_dir in DATA_DIRS :
        for (dirpath, dirnames, filenames) in os.walk(data_dir) :
            fnames.extend(dirpath + fname for fname in filenames if '.txt' in fname)

    instances = list()
    for fname in fnames :
        # print("Reading", fname)
        for i_line, line in enumerate(open(fname)) :
            line_words = line.split()
            for length in range(1, 4 + 1) :
                for i_word, gram in enumerate(zip(*(line_words[i:] for i in range(length)))) :
                    words = list(gram)
                    pres = line_words[max(0, i_word-2) : i_word]
                    afts = line_words[i_word + length : min(len(line_words), i_word + length + 1)]
                    label = '{' in words[0] and '}' in words[-1] and (
                            all('{' not in word for word in words[1:]))
    
                    pres = [Preprocesser.preprocess(s) for s in pres]
                    words = [Preprocesser.preprocess(s) for s in words]
    
                    s = ' ' .join(words)
                    if s.lower() in SKIP_WORDS :
                        continue
                    info = ({
                        'str' : s,
                        'line-no' : i_line + 1,
                        'word-no' : i_word + 1,
                        'file-path' : fname,
                        'label' : label,
                        'features' : FeatureExtractor.extract_features(pres, words, afts),
                    })
                    if all(v is False for v in info['features'].values() if type(v) == type(True)) :
                        continue
                    instances.append(info)
    return instances

# print("# Some instances")
# for ins in np.random.choice(instances, 10, replace=False) :
#     print(ins['str'])
#     pprint(ins)
#     print()
# print('\n\n'.join(
#     '- "%s":\n'%(ins['str']) + '\n'.join(
#         '%s: %s'%(k, v)
#         for k, v in ins['features'].items())
#     ))
print()

training_instances = load_data(TRAINING_DATA_DIRS)
#random_indices = np.random.permutation(len(instances))
X = np.array([[ins['features'].get(f) for f in FeatureExtractor.FEATURES] for ins in training_instances], dtype='int')
Y = np.array([ins['label'] for ins in training_instances])
#X , Y = X[random_indices], Y[random_indices]

testing_instances = load_data(TESTING_DATA_DIRS)
#random_indices = np.random.permutation(len(instances))
X_test = np.array([[ins['features'].get(f) for f in FeatureExtractor.FEATURES] for ins in testing_instances], dtype='int')
Y_test = np.array([ins['label'] for ins in testing_instances])

print("X looks like:", X[:10])
print("Y looks like:", Y[:10])
print(collections.Counter(Y))

clfs = {
    'decision-tree' : sk.tree.DecisionTreeClassifier(criterion='entropy',
        # min_weight_fraction_leaf=0.001,
        class_weight={True: 1, False: 0.255}),
     'random-forest' : sk.ensemble.RandomForestClassifier(criterion='entropy',
         min_weight_fraction_leaf=0.001,
         class_weight={True: 1, False: 0.4}),
     'SVM' : sk.svm.SVC(),
     'linear-regression' : sk.svm.LinearSVC(),
     'logistic-regression' : sk.linear_model.SGDClassifier(loss='log'),
}

def report(instances,X,Y,Y_pred,csv_name) :
    print("- %s: %0.2f"%('precision', sk.metrics.precision_score(Y, Y_pred)))
    print("- %s: %0.2f"%('recall', sk.metrics.recall_score(Y, Y_pred)))
    with open(csv_name, 'w') as csvfile:
        fieldnames = list(instances[0].keys())
        fieldnames.remove('label')
        fieldnames.remove('str')
        fieldnames = ['predict', 'label', 'str'] + fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for ins, predict in zip(instances, Y_pred) :
            ins['predict'] = predict
            writer.writerow(ins)

print('# Results')
for clfname, clf in clfs.items() :
    print('## %s'%(clfname))
    print('### cross validation')
    cv = sk.model_selection.ShuffleSplit()
    for scoring in ('precision', 'recall') :
        scores = sk.model_selection.cross_val_score(clf, X, Y, cv=cv, scoring=scoring)
        print("- %s: %0.2f (+/- %0.2f)" % (scoring, scores.mean(), scores.std() * 2))
    clf.fit(X, Y)

    print('### training set')
    Y_pred = clf.predict(X)
    report (training_instances,X,Y,Y_pred,clfname+"_training.csv")
    print('### testing set')
    Y_test_pred = clf.predict(X_test)
    report (testing_instances,X_test,Y_test,Y_test_pred,clfname+"_tesing.csv")

    #postprocess
    Y_test_pred_post = Y_test_pred.copy()
    white_list=[]
  #  for i, (ins,pred) in enumerate(zip(testing_instances,Y_test_pred)) :
  #      l = len(ins['str'].split());
  #      if l == 2 and pred==True:
  #          print ( ins['str'],l)
  #          white_list.append(ins['str'].split()[0])
  #  print (white_list)
            

    for i, (ins,pred) in enumerate(zip(testing_instances,Y_test_pred)) :
        if any(n in ins['str'] for n in PROPER_NOUN_LIST) :
            Y_test_pred_post[i] = False
    print('### after post-processing')
    report (testing_instances,X_test,Y_test,Y_test_pred_post,clfname+"_tesing_post.csv")

    
    try :
        import pydotplus
        if clfname == 'decision-tree' :
            dot_data = sk.tree.export_graphviz(clf, out_file=None,
                         feature_names=FeatureExtractor.FEATURES,
                        #  class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
            graph = pydotplus.graph_from_dot_data(dot_data)
            graph.write_png(clfname + '.png')
    except ImportError :
        print("NO pyplotplus!")
        pass
