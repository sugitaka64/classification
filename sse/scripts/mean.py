#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""mean.

Usage:
    mean.py
        --input_file_path=<input_file_path>
        --output_file_path=<output_file_path>
    mean.py -h | --help

Options:
    -h --help  show this screen and exit.
"""

from datetime import datetime
from docopt import docopt
import numpy as np
import pandas as pd
import sys

class CalculateMean(object):
    """mean."""

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
        input_df.rename(columns={0: 'label'}, inplace=True)

        # get label column
        label_df = pd.DataFrame(
            input_df.iloc[:, 0],
            dtype=np.int64
        )

        # mean
        meam_array = []
        # unique label
        unique_label_df = label_df.drop_duplicates()
        for k, v in unique_label_df.iterrows():
            # select target label
            label = v['label']
            query = 'label == %d' % (label)
            tmp_df = input_df.query(query)

            # features
            tmp_df = tmp_df.drop('label', axis=1)

            # to nd array
            tmp_array = tmp_df.as_matrix()

            # calculate mean
            mean = tmp_array.mean(0).tolist()
            mean.insert(0, label)

            # append
            meam_array.append(mean)

        # convert to dataframe
        mean_df = pd.DataFrame(
            meam_array,
            dtype=np.float64
        )

        # save
        mean_df.to_csv(output_file_path, index=False, header=False)

        return True

if __name__ == '__main__':
    print('%s %s start.' % (datetime.today(), __file__))

    # get parameters
    args = docopt(__doc__)
    input_file_path = args['--input_file_path']
    output_file_path = args['--output_file_path']

    # execute
    calculate_mean = CalculateMean()
    calculate_mean.run(
        input_file_path,
        output_file_path
    )

    print('%s %s end.' % (datetime.today(), __file__))

    sys.exit(0)
