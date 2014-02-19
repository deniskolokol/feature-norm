# /usr/bin/python

import sys
import optparse
import itertools
import numpy as np


class FeatureEngineer():
    def __init__(self, dstrain, dstest, N, **kwargs):
        self.X = None # Features.
        self.Y = None # Targets.
        self.N = None # Number of examples.

        # Filenames.
        self.fname_train = dstrain
        self.fname_test = dstest

        # Fill training and testing datasets.
        self.ds_train = self.fill_ndarray(dstrain, N, **kwargs)
        self.ds_test = self.fill_ndarray(dstest,  N, **kwargs)

    def fill_ndarray(self, filename, nlines=None, **kwargs):
        """
        Extract data from file to a single big ndarray.
        """
        def file_len(f):
            for i, l in enumerate(f): pass
            return i + 1

        with open(filename, 'rb') as f:
            delimiter = kwargs.get('delimiter', ',')
            skiprows = kwargs.get('skiprows', 0)
            skipcols = kwargs.get('skipcols', 0)

            # number of columns.
            num_cols = len(f.readline().split(delimiter))
            f.seek(0)
            
            missing_values = tuple(["NA" for i in range(num_cols - skipcols)])
            filling_values = tuple([0 for i in range(num_cols - skipcols)])

            if nlines < 0:
                nlines = file_len(f)
            self.N = nlines # Update number of examples

            return np.ndfromtxt(itertools.islice(f, nlines),
                                dtype=float,
                                skiprows=skiprows,
                                delimiter=delimiter,
                                usecols=range(skipcols, num_cols),
                                missing_values=missing_values,
                                filling_values=filling_values)


    def prepare_data(self, target_cols=1):
        """
        Extract features and target from data.
        `target_cols` is a dimensionality of target vector (default 1).
        """
        self.X = self.ds_train[:, :np.negative(target_cols)]
        self.Y = self.ds_train[:, np.negative(target_cols):]
        self.Y = np.reshape(self.Y, (target_cols, self.Y.shape[0]))

    def remove_correlated_features(self):
        # TO-DO: Reshape each pair of columns into a separate ndarray,
        # remove all with covariance > 0.95
        pass

    def remove_unary_cols(self):
        unary_cols = np.all(self.X == self.X[0, :], axis=0)
        indices = [i for i in range(len(unary_cols)) if unary_cols[i]]
        self.X = np.delete(self.X, indices, axis=1)

    def normalize_features(self):
        self.X /= np.max(np.abs(self.X), axis=0)


def main(csv_train, csv_test, **kwargs):

    N = kwargs.get('N', None)
    if N is None:
        N = -1
    params = {'skiprows': kwargs.get('skiprows', 0),
              'skipcols': kwargs.get('skipcols', 0)}

    fex = FeatureEngineer(csv_train, csv_test, int(N), **params)
    fex.prepare_data()
    fex.remove_correlated_features()
    fex.remove_unary_cols()
    fex.normalize_features()

    result = np.zeros((fex.X.shape[0], fex.X.shape[1] + 1))
    result[:, :fex.X.shape[1]] = fex.X
    result[:, -1] = fex.Y

    np.savetxt(fex.fname_train.rsplit('.', 1)[0] + ('_clean_%s.csv' % fex.N),
               result,
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
                         default=1,
                         help="Skip rows [default \'%default\'] - do not include header")
    cmdparser.add_option("-c", "--skipcolumnns",
                         action="store",
                         dest="skipcols",
                         default=1,
                         help="Skip columns [default \'%default\'] - do not include row id")
    cmdparser.add_option("-t", "--train",
                         action="store",
                         dest="csv_train",
                         default=1,
                         help="Training dataset (csv file)")
    cmdparser.add_option("-s", "--test",
                         action="store",
                         dest="csv_test",
                         default=1,
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
