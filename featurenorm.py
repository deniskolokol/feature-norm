# /usr/bin/python

import sys
import csv
import optparse
import itertools
import numpy as np


class FeatureEngineer():
    functions = {
        'unary': lambda a: np.all(a == a[0, :], axis=0),
        'correlated': lambda a: [np.unique(np.cov(a[:, [n, n-1]].T)).size == 1
                                 for n in xrange(1, a.shape[1])],
        }

    def __init__(self, dstrain, dstest=None, N=1, **kwargs):
        # Number of examples.
        self.N = None

        # Fill training dataset (X).
        self.X = self.from_txt(dstrain, nlines=N, **kwargs)[:, :-1]

        # Target vector (Y) - the last column of training dataset (integer!).
        t_kwargs = kwargs.copy()
        t_kwargs.update({'dtype': int, 'usecols': -1})
        self.Y = self.from_txt(dstrain, nlines=N, **t_kwargs)

        # Target is a R-vector.
        self.Y = np.reshape(self.Y, (1, self.Y.shape[0]))

        # Test datasets.
        if dstest:
            self.ds_test = self.from_txt(dstest, nlines=N, **kwargs)

    def from_txt(self, filename, nlines=None, dtype=float, **kwargs):
        """
        Extract data from file to a single big ndarray.
        """
        with open(filename, 'rb') as f:
            delimiter = kwargs.get('delimiter', ',')
            skiprows = kwargs.get('skiprows', 0)

            # Total number of columns.
            num_cols = len(f.readline().split(delimiter))

            # Use columns defined by indexes.
            usecols = kwargs.get('usecols', None)
            if not usecols:

                # Skip columns from left side.
                skipcols = kwargs.get('skipcols', 0)
                usecols=range(skipcols, num_cols)

            # Ensure list. 
            if isinstance(usecols, int):
                usecols = [usecols]

            # Number of examples.
            f.seek(0)
            if nlines > 0:
                f = itertools.islice(f, nlines + skiprows)
            data = np.ndfromtxt(f,
                                dtype=dtype,
                                skiprows=skiprows,
                                delimiter=delimiter,
                                usecols=usecols)
            return np.nan_to_num(data)

    def to_txt(self, filename, arr, **kwargs):
        np.savetxt(filename, arr, **kwargs)
        
    def _remove_cols_under_condition(self, cond):
        try:
            func = self.functions[cond]
        except KeyError:
            return

        i_X = func(self.X)
        i_test = func(self.ds_test)

        # Take only columns that satisfy condition in both datasets.
        indices = [i for i in range(len(i_X)) if i_X[i] and i_test[i]]

        # Remove features from both train and test sets.
        self.X = np.delete(self.X, indices, axis=1)
        self.ds_test = np.delete(self.ds_test, indices, axis=1)

    def remove_unary_cols(self):
        """
        Removes features with no variance in the data.
        """
        self._remove_cols_under_condition('unary')

    def remove_correlated_features(self):
        """
        Removes features highly correlated with other features.
        """
        self._remove_cols_under_condition('correlated')

    def normalize_features(self):
        """
        Normalizes features in the training set.
        """
        # m = np.max(np.abs(self.X), axis=0) == 0
        # self.X = np.delete(self.X, np.where(m==True)[0], axis=1)

        self.X /= np.max(np.abs(self.X), axis=0)

        # Fix for zero-valued unary fields.
        self.X = np.nan_to_num(self.X)


def main(csv_train, csv_test, **kwargs):
    N = kwargs.pop('N', None)
    if N is None:
        N = -1

    # Original
    # fex = FeatureEngineer(csv_train, csv_test, int(N), **kwargs)
    # fex.remove_unary_cols()
    # fex.remove_correlated_features()
    # fex.normalize_features()

    # result = np.zeros((fex.X.shape[0], fex.X.shape[1] + 1))
    # result[:, :fex.X.shape[1]] = fex.X
    # result[:, -1] = fex.Y

    # # M-1 columns (features) should be float, but the last one (classes) int.
    # fmt = ['%.8e' for i in range(result.shape[1 ]-1)]
    # fmt.append('%d')

    # kw = {'fmt': ' '.join(fmt), 'delimiter': ' '}
    # fname = csv_train.rsplit('.', 1)[0] + ('_clean_%s.csv' % fex.X.shape[0])

    # fex.to_txt(fname, result, **kw)


    # TEST
    def add_Y(arr, y):
        res = np.zeros((arr.shape[0], arr.shape[1] + 1))
        res[:, :arr.shape[1]] = arr
        res[:, -1] = y
        return res

    result = None
    n = 1000 # Iterations.
    out_class = [50, 1000, 1500] # Num of records with output [==0, <50%, >50%]
    skiprows = kwargs.get('skiprows', 1)
    for i in range(n):
        cl = i % len(out_class)
        kwargs.update({'skiprows': skiprows})

        fex = FeatureEngineer(csv_train, N=out_class[cl], **kwargs)
        fex.normalize_features()
        res = add_Y(fex.X, fex.Y)

        if cl == 0:
            res = res[res[:,-1] == 0, :]
        elif cl == 1:
            res = res[np.logical_and(res[:,-1] > 0, res[:,-1] <= 50), :]
        elif cl == 2:
            res = res[res[:,-1] > 50, :]

        try:
            result = np.append(result, res, axis=0)
        except ValueError:
            result = res

        print i, ':', cl, res.shape, result.shape

        skiprows += out_class[cl]

    np.random.shuffle(result)
    fname = csv_train.rsplit('.', 1)[0] + ('_clean_%s.csv' % result.shape[0])
    with open(fname, 'wb') as f:
        csv.writer(f).writerows(result)



if __name__ == '__main__':
    cmdparser = optparse.OptionParser(usage="usage: python %prog [OPTIONS]")
    cmdparser.add_option("-N", "--n_examples",
                         action="store",
                         dest="n_examples",
                         help="Process N examples")
    cmdparser.add_option("-r", "--skiprows",
                         action="store",
                         dest="skiprows",
                         default=0,
                         help="Skip rows [default \'%default\'] - do not include header")
    cmdparser.add_option("-c", "--skipcolumnns",
                         action="store",
                         dest="skipcols",
                         default=0,
                         help="Skip columns [default \'%default\'] - do not include row id")
    cmdparser.add_option("-t", "--train",
                         action="store",
                         dest="csv_train",
                         help="Training dataset (csv file)")
    cmdparser.add_option("-s", "--test",
                         action="store",
                         dest="csv_test",
                         help="Test dataset (csv file)")
    opts, args = cmdparser.parse_args()

    try:
        csv_train = opts.csv_train
        csv_test  = opts.csv_test
    except IndexError:
        print >> sys.stderr, 'Both files with training and test datasets should be specified!'
        sys.exit()

    kwargs = {'skiprows': int(opts.skiprows),
              'skipcols': int(opts.skipcols),
              'N': opts.n_examples}

    main(csv_train, csv_test, **kwargs)
