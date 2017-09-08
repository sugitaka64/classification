#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""SVM.

Usage:
    svm.py
        --input_file_path=<input_file_path>
        --output_file_path=<output_file_path>
    svm.py -h | --help

Options:
    -h --help  show this screen and exit.
"""

from datetime import datetime
from docopt import docopt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sys

class RunSVM(object):
    """SVM."""

    def run(
        self,
        input_file_path: str,
        output_file_path: str
    ) -> bool:
        """execute."""
        # input
        input_df = pd.read_csv(
            input_file_path,
            header=None,
            dtype=np.float64
        ).fillna(0.0)

        # split data
        (training_df, test_df) = train_test_split(input_df)

        # get label column
        label_column_name = input_df.columns[0]
        training_label_array = training_df.iloc[:, 0].as_matrix()
        test_label_df = pd.DataFrame(
            test_df.iloc[:, 0],
            columns=[label_column_name],
            dtype=np.int64
        ).reset_index(drop=True)

        # get feature columns
        training_features_df = training_df.drop(label_column_name, axis=1)
        training_features_array = training_features_df.as_matrix()
        test_features_df = test_df.drop(label_column_name, axis=1)
        test_features_array = test_features_df.as_matrix()

        # svm
        params = [{
            'gamma': [0.001, 0.01, 0.1, 1.0, 'auto'],
            'C': [1.0, 10.0, 100.0, 1000.0],
        }]
        gscv = GridSearchCV(SVC(), params, cv=3)
        gscv.fit(training_features_array, training_label_array)

        """
        print(gscv.best_estimator_)
        print(gscv.best_score_)
        print(gscv.best_params_)
        """

        # predict
        predicts_label_array = gscv.predict(test_features_array)
        predicts_label_df = pd.DataFrame(
            predicts_label_array,
            dtype=np.float64
        )

        # join svm presicts and label column
        output_df = pd.concat([test_label_df, predicts_label_df], axis=1)

        # save
        output_df.to_csv(output_file_path, index=False, header=False)

        return True

if __name__ == '__main__':
    print('%s %s start.' % (datetime.today(), __file__))

    # get parameters
    args = docopt(__doc__)
    input_file_path = args['--input_file_path']
    output_file_path = args['--output_file_path']

    # execute
    run_svm = RunSVM()
    run_svm.run(
        input_file_path,
        output_file_path
    )

    print('%s %s end.' % (datetime.today(), __file__))

    sys.exit(0)
