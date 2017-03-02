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

DATA_DIRS = ["../dataset/labeled_texts/fortune500/", "../dataset/labeled_texts/fortune500_wiki/"]

fnames = list()
for data_dir in DATA_DIRS :
    for (dirpath, dirnames, filenames) in os.walk(data_dir) :
        fnames.extend(dirpath + fname for fname in filenames if '.txt' in fname)

SKIP_WORDS = ('of', 'and', 'the', 'to')

class Preprocesser :
    TABLE = {
        '{' : '', '}' : '', ## eliminate {}
        "'s'" : '', "s'" : 's', ## eliminate 's
    }
    preprocess = functools.reduce(lambda f, kv : lambda s : f(s).replace(*kv), TABLE.items(), lambda s : s)

class FeatureExtractor :
    FEATURES = (
        'n-words', 'n-characters',
        'all-capitalized', 'all-inintial-capitalized',
        'no-pre-inintial-capitalized', 'no-aft-inintial-capitalized',
        'common-ends', 'common-pres',
        'contain-hyphen', 'contain-ampersand',
        'sentence-start',
        )
    r_after = re.compile(r'\bcorporation|\bcorp\b|\bincorporation|\binc\b|\bcompany|\bco\b|\bllc\b')
    r_before = re.compile(r'\bacqui|\bbuy|\bbought|\bcompetit|\bformer|\bmerg|\bown|\brival|\bsell|\bsold')

    @staticmethod
    def extract_features(pres, words, afts) :
        s = ' '.join(words)
        p = ' '.join(pres)
        return {
            'n-words' : len(words),
            'n-characters' : len(s),
            'all-capitalized' : all(w.isupper() or w in SKIP_WORDS for w in words),
            'all-inintial-capitalized' : all(w[0].isupper() or w in SKIP_WORDS for w in words),
            'no-pre-inintial-capitalized' : len(pres) == 0 or not pres[-1][0].isupper(),
            'no-aft-inintial-capitalized' : len(afts) == 0 or not afts[-1][0].isupper(),
            'common-ends' : FeatureExtractor.r_after.search(s.lower()) is not None,
            'common-pres' : FeatureExtractor.r_before.search(p.lower()) is not None,
            'contain-hyphen' : '-' in s,
            'contain-ampersand' : '&' in s,
            'sentence-start' : len(pres) == 0 or pres[-1][-1] in '.!?',
        }


instances = list()
for fname in fnames :
    print("Reading", fname)
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

print("# Some instances")
for ins in np.random.choice(instances, 10, replace=False) :
    print(ins['str'])
    pprint(ins)
    print()
# print('\n\n'.join(
#     '- "%s":\n'%(ins['str']) + '\n'.join(
#         '%s: %s'%(k, v)
#         for k, v in ins['features'].items())
#     ))
print()

#random_indices = np.random.permutation(len(instances))
X = np.array([[ins['features'].get(f) for f in FeatureExtractor.FEATURES] for ins in instances], dtype='int')
Y = np.array([ins['label'] for ins in instances])
#X , Y = X[random_indices], Y[random_indices]

print("X looks like:", X[:10])
print("Y looks like:", Y[:10])
print(collections.Counter(Y))

clfs = {
    'decision-tree' : sk.tree.DecisionTreeClassifier(criterion='entropy',
        min_weight_fraction_leaf=0.001,
        class_weight={True: 1, False: 1}),
    # 'random-forest' : sk.ensemble.RandomForestClassifier(criterion='entropy',
    #     min_weight_fraction_leaf=0.001,
    #     class_weight={True: 1, False: 0.2}),
    # 'SVM' : sk.svm.SVC(),
    # 'linear-regression' : sk.svm.LinearSVC(),
    # 'logistic-regression' : sk.linear_model.SGDClassifier(loss='log'),
}

print('# Results')
for clfname, clf in clfs.items() :
    print('## %s'%(clfname))
    print('### cross validation')
    cv = sk.model_selection.ShuffleSplit()
    for scoring in ('precision', 'recall') :
        scores = sk.model_selection.cross_val_score(clf, X, Y, cv=cv, scoring=scoring)
        print("- %s: %0.2f (+/- %0.2f)" % (scoring, scores.mean(), scores.std() * 2))
    print('### whole set')
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    print("- %s: %0.2f"%('precision', sk.metrics.precision_score(Y, Y_pred)))
    print("- %s: %0.2f"%('recall', sk.metrics.recall_score(Y, Y_pred)))
    with open(clfname + '.csv', 'w') as csvfile:
        fieldnames = list(instances[0].keys())
        fieldnames.remove('label')
        fieldnames.remove('str')
        fieldnames = ['predict', 'label', 'str'] + fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for ins, predict in zip(instances, Y_pred) :
            ins['predict'] = predict
            writer.writerow(ins)
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
