#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""PLSA.

Usage:
    plsa.py
        --input_file_path=<input_file_path>
        --output_file_path=<output_file_path>
        --num_topics=<num_topics>
    plsa.py -h | --help

Options:
    -h --help  show this screen and exit.
"""

from datetime import datetime
from docopt import docopt
import numpy as np
import pandas as pd
import sys

class PLSA(object):
    """cf. http://qiita.com/HZama/items/0957f74f8da1302f7652 .
           http://qiita.com/HZama/items/561cb240620991d3a0e2 .
    """

    def __init__(self, n, z):
        """init."""
        self.n = n
        self.x = n.shape[0]
        self.y = n.shape[1]
        self.z = z

        # P(z)
        self.pz = np.random.rand(self.z)
        # P(x|z)
        self.px_z = np.random.rand(self.z, self.x)
        # P(y|z)
        self.py_z = np.random.rand(self.z, self.y)

        # normalization
        self.pz /= np.sum(self.pz)
        self.px_z /= np.sum(self.px_z, axis=1)[:, None]
        self.py_z /= np.sum(self.py_z, axis=1)[:, None]

    def train(self, k=200, t=1.0e-7):
        """train."""
        prev_llh = 100000
        for i in range(k):
            self.em_algorithm()
            llh = self.llh()

            if abs((llh - prev_llh) / prev_llh) < t:
                break

            prev_llh = llh

    def em_algorithm(self):
        """em_algorithm."""
        tmp = self.n / np.einsum('k,ki,kj->ij', self.pz, self.px_z, self.py_z)
        tmp[np.isnan(tmp)] = 0
        tmp[np.isinf(tmp)] = 0

        pz = np.einsum('ij,k,ki,kj->k', tmp, self.pz, self.px_z, self.py_z)
        px_z = np.einsum('ij,k,ki,kj->ki', tmp, self.pz, self.px_z, self.py_z)
        py_z = np.einsum('ij,k,ki,kj->kj', tmp, self.pz, self.px_z, self.py_z)

        self.pz = pz / np.sum(pz)
        self.px_z = px_z / np.sum(px_z, axis=1)[:, None]
        self.py_z = py_z / np.sum(py_z, axis=1)[:, None]

    def llh(self):
        """llh."""
        pxy = np.einsum('k,ki,kj->ij', self.pz, self.px_z, self.py_z)
        pxy /= np.sum(pxy)
        lpxy = np.log(pxy)
        lpxy[np.isinf(lpxy)] = -1000

        return np.sum(self.n * lpxy)

class RunPLSA(object):
    """PLSA."""

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

        # convert to nd array
        input_array = input_df.as_matrix()

        # plsa
        plsa = PLSA(input_array, num_topics)
        plsa.train()

        pz_x = plsa.px_z.T * plsa.pz[None, :]
        data = pz_x / np.sum(pz_x, axis=1)[:, None]

        # convert to dataframe
        plsa_df = pd.DataFrame(
            data,
            dtype=np.float64
        )

        # join svd and label column
        output_df = pd.concat([label_df, plsa_df], axis=1)

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
    run_plsa = RunPLSA()
    run_plsa.run(
        input_file_path,
        output_file_path,
        num_topics
    )

    print('%s %s end.' % (datetime.today(), __file__))

    sys.exit(0)
