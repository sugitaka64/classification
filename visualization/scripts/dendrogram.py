#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""LDA.

Usage:
    dendrogram.py
        --input_file_path=<input_file_path>
        [--display_node_num=<display_node_num>]
        [--x_size=<x_size>]
        [--y_size=<y_size>]
    dendrogram.py -h | --help

Options:
    -h --help  show this screen and exit.
"""

from datetime import datetime
from docopt import docopt
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
import sys

class PlotDendrogram(object):
    """plot dendrogram."""

    def run(
        self,
        input_file_path: str,
        display_node_num: int,
        x_size: int,
        y_size: int
    ) -> bool:
        """execute."""
        # input
        input_df = pd.read_csv(
            input_file_path,
            header=None,
            dtype=np.float64
        ).fillna(0.0)

        # get user id column
        label_column_name = input_df.columns[0]
        label_df = pd.DataFrame(
            input_df.iloc[:, 0],
            columns=[label_column_name],
            dtype=np.int64
        )
        input_df = input_df.drop(label_column_name, axis=1)

        # convert to nd array
        label_array = label_df.as_matrix()
        input_array = input_df.as_matrix()

        # plot
        result = linkage(input_array, method='ward')
        plt.figure(figsize=(x_size, y_size))
        rcParams['lines.linewidth'] = 0.5
        if display_node_num is None:
            dendrogram(result, truncate_mode='lastp', labels=label_array)
        else:
            dendrogram(result, truncate_mode='lastp', p=display_node_num, labels=label_array)
        plt.show()

        return True

if __name__ == '__main__':
    print('%s %s start.' % (datetime.today(), __file__))

    # get parameters
    args = docopt(__doc__)
    input_file_path = args['--input_file_path']
    display_node_num = args['--display_node_num']
    if display_node_num is not None:
        display_node_num = int(args['--display_node_num'])
    x_size = args['--x_size']
    if x_size is not None:
        x_size = int(args['--x_size'])
    else:
        x_size = 8
    y_size = args['--y_size']
    if y_size is not None:
        y_size = int(args['--y_size'])
    else:
        y_size = 6

    # execute
    plot_dendrogram = PlotDendrogram()
    plot_dendrogram.run(
        input_file_path,
        display_node_num,
        x_size,
        y_size
    )

    print('%s %s end.' % (datetime.today(), __file__))

    sys.exit(0)
