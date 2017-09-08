#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""SVD.

Usage:
    svd.py
        --input_file_path=<input_file_path>
        --output_file_path=<output_file_path>
        --num_topics=<num_topics>
    svd.py -h | --help

Options:
    -h --help  show this screen and exit.
"""

from datetime import datetime
from docopt import docopt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import sys

class RunSVD(object):
    """SVD."""

    def run(
        self,
        input_file_path: str,
        output_file_path: str,
        num_topics: int
    ) -> bool:
        """execute."""
        # input
        input_df = pd.read_csv(
            input_file_path,
            header=None,
            dtype=np.float64
        ).fillna(0.0)

        # get label column
        label_column_name = input_df.columns[0]
        label_df = pd.DataFrame(
            input_df.iloc[:, 0],
            columns=[label_column_name],
            dtype=np.int64
        )
        input_df = input_df.drop(label_column_name, axis=1)

        # svd
        svd = TruncatedSVD(n_components=num_topics, n_iter=10)
        transformed = svd.fit_transform(input_df)
        transformed_df = pd.DataFrame(
            transformed,
            dtype=np.float64
        )

        # join svd and label column
        output_df = pd.concat([label_df, transformed_df], axis=1)

        # save
        output_df.to_csv(output_file_path, index=False, header=False)

        return True

if __name__ == '__main__':
    print('%s %s start.' % (datetime.today(), __file__))

    # get parameters
    args = docopt(__doc__)
    input_file_path = args['--input_file_path']
    output_file_path = args['--output_file_path']
    num_topics = int(args['--num_topics'])

    # execute
    run_svd = RunSVD()
    run_svd.run(
        input_file_path,
        output_file_path,
        num_topics
    )

    print('%s %s end.' % (datetime.today(), __file__))

    sys.exit(0)
