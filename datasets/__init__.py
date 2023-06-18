"""Basic dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random


def update_argparser(parser):
  parser.add_argument(
      '--train_batch_size',
      help='Batch size for training.',
      type=int,
      default=32)
  parser.add_argument(
      '--eval_batch_size',
      help='Batch size for evaluation.',
      type=int,
      default=32)
  parser.add_argument(
      '--num_data_threads',
      help='Number of threads for data transformation.',
      type=int,
      default=8)
