# /usr/bin/python

import sys
import optparse
import itertools
import numpy as np


class FeatureEngineer():
    functions = {
        'unary': lambda a: np.all(a == a[0, :], axis=0),
        'correlated': lambda a: [np.unique(np.cov(a[:, [n, n-1]].T)).size == 1
                                 for n in xrange(1, a.shape[1])],
        }

    def __init__(self, dstrain, dstest, N, **kwargs):
        # Number of examples.
        self.N = None

        # TO-DO: 
        # Optimize here: 'sniff' read both files first and fill datasets metadata:
        # file object / slice, number of rows, number of columns, etc.
        # Only after that fill self.X, self.Y and self.test.

        # Fill training dataset (X).
        self.X = self.fill_array(dstrain, nlines=N, **kwargs)[:, :-1]

        # Target vector (Y) - the last column of training dataset (integer!).
        t_kwargs = kwargs.copy()
        t_kwargs.update({'dtype': int, 'usecols': -1})
        self.Y = self.fill_array(dstrain, nlines=N, **t_kwargs)

        # Target is a R-vector.
        self.Y = np.reshape(self.Y, (1, self.Y.shape[0]))

        # Test datasets.
        self.ds_test = self.fill_array(dstest, nlines=N, **kwargs)

    def fill_array(self, filename, nlines=None, dtype=float, **kwargs):
        """
        Extract data from file to a single big ndarray.
        """
        def file_len(f):
            for i, l in enumerate(f): pass
            return i + 1

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

            # Ensure list to 
            if isinstance(usecols, int):
                usecols = [usecols]
            missing_values = tuple(["NA" for i in range(len(usecols))])
            filling_values = tuple([0 for i in range(len(usecols))])

            # Number of examples.
            if nlines < 0:
                nlines = file_len(f)
            self.N = nlines

            f.seek(0)
            return np.ndfromtxt(itertools.islice(f, nlines),
                                dtype=dtype,
                                skiprows=skiprows,
                                delimiter=delimiter,
                                usecols=usecols,
                                missing_values=missing_values,
                                filling_values=filling_values)

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
        self.X /= np.max(np.abs(self.X), axis=0)


def main(csv_train, csv_test, **kwargs):
    N = kwargs.pop('N', None)
    if N is None:
        N = -1

    fex = FeatureEngineer(csv_train, csv_test, int(N), **kwargs)
    fex.remove_unary_cols()
    fex.remove_correlated_features()
    # fex.normalize_features()

    result = np.zeros((fex.X.shape[0], fex.X.shape[1] + 1))
    result[:, :fex.X.shape[1]] = fex.X
    result[:, -1] = fex.Y

    # M-1 columns (features) should be float,
    # but the last one (classes) int.
    fmt = ['%.8e' for i in range(result.shape[1]-1)]
    fmt.append('%d')

    np.savetxt(csv_train.rsplit('.', 1)[0] + ('_clean_%s.csv' % fex.N),
               result,
               fmt=' '.join(fmt),
               delimiter=" ")


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
