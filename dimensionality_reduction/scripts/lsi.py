#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""LSI.

Usage:
    lsi.py
        --input_file_path=<input_file_path>
        --output_file_path=<output_file_path>
        --num_topics=<num_topics>
    lsi.py -h | --help

Options:
    -h --help  show this screen and exit.
"""

from datetime import datetime
from docopt import docopt
from gensim.models import LsiModel
import numpy as np
import pandas as pd
import sys

class RunLSI(object):
    """LSI."""

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

        # convert to dictionaty
        input_df_column_names = input_df.columns.T.tolist()
        dict_values = input_df.T.to_dict().values()

        # make corpus
        corpus = []
        for row in dict_values:
            tmp = []
            for word, score in row.items():
                tmp.append((input_df_column_names.index(word), score))

            corpus.append(tmp)

        # lsi
        lsi_model = LsiModel(corpus, num_topics=num_topics)
        all_topics = lsi_model.print_topics(num_topics)

        # convert to list
        corpus_lsi = lsi_model[corpus]
        data = []
        for doc in corpus_lsi:
            tmp = []
            for i in range(len(all_topics)):
                try:
                    tmp.append(doc[i][1])
                except IndexError:
                    tmp.append(0.0)

            data.append(tmp)
        # convert to dataframe
        lsi_df = pd.DataFrame(
            data,
            dtype=np.float64
        )

        # join svd and label column
        output_df = pd.concat([label_df, lsi_df], axis=1)

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
    run_lsi = RunLSI()
    run_lsi.run(
        input_file_path,
        output_file_path,
        num_topics
    )

    print('%s %s end.' % (datetime.today(), __file__))

    sys.exit(0)
