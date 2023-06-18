from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def update_argparser(parser):
  parser.add_argument(
      '--learning_rate',
      help='Learning rate.',
      default=0.001,
      type=float,
  )