import os

import common.modes
import torch
from PIL import Image
import torchvision.transforms as transforms


EVAL_LR_DIR = lambda source,t,c,s: '{}'.format(source)+'{}/'.format(t) + 'DIV2K_train_LR_bicubic_{}'.format(c)+'/X{}/'.format(s)
EVAL_HR_DIR = lambda source,t,c,s: '{}'.format(source)+'{}/'.format(t) + 'DIV2K_train_HR_{}'.format(c)+'/X{}/'.format(s)
TRAIN_LR_DIR = lambda source,t,c,s: '{}'.format(source)+'{}/'.format(t) + 'DIV2K_train_LR_bicubic_{}'.format(c)+'/X{}/'.format(s)
TRAIN_HR_DIR = lambda source,t,c,s: '{}'.format(source)+'{}/'.format(t) + 'DIV2K_train_HR_{}'.format(c)+'/X{}/'.format(s)

# save_file_name = []


def update_argparser(parser):
  # datasets._isr.update_argparser(parser)
  parser.add_argument(
      '--input_dir', help='Directory of input files in predict mode.')
  parser.add_argument(
      '--scale',
      help='Scale factor for image super-resolution.',
      default=2,
      type=int)
  parser.set_defaults(
      num_data_threads=4,
      num_channels=3,
      num_patches=1,
      train_batch_size=16,
      eval_batch_size=1,
      image_mean=0.5,)


def get_dataset(mode, params):
  if mode == common.modes.PREDICT:
    return DIV2K_(mode, params)
  else:
    return DIV2K(mode, params)

def get_filename(params):
    file_name = sorted(os.listdir(TRAIN_HR_DIR(params.source_path,params.tt,params.chunk,params.scale)))
    file_name.sort(key=lambda x: int(x[:-4]))
    return file_name


class DIV2K(torch.utils.data.Dataset):

    def __init__(self, mode, params):
        self.lr_dir = {
            common.modes.TRAIN: TRAIN_LR_DIR(params.source_path,params.tt,params.chunk,params.scale),
            common.modes.EVAL: EVAL_LR_DIR(params.source_path,params.tt,params.chunk,params.scale),
        }[mode]
        self.hr_dir = {
            common.modes.TRAIN: TRAIN_HR_DIR(params.source_path,params.tt,params.chunk,params.scale),
            common.modes.EVAL: EVAL_HR_DIR(params.source_path,params.tt,params.chunk,params.scale),
        }[mode]

        self.low_res_files = os.listdir(self.lr_dir)
        self.low_res_files = list_image_files(self.lr_dir)
        self.high_res_files = os.listdir(self.hr_dir)
        self.high_res_files = list_image_files(self.hr_dir)

    def __getitem__(self, index):
        low_res_path = os.path.join(self.lr_dir, self.low_res_files[index][1])
        high_res_path = os.path.join(self.hr_dir, self.high_res_files[index][1])

        low_res_image = Image.open(low_res_path).convert('RGB')
        high_res_image = Image.open(high_res_path).convert('RGB')


        low_res_image = transforms.functional.to_tensor(low_res_image)
        high_res_image = transforms.functional.to_tensor(high_res_image)

        return low_res_image, high_res_image

    def __len__(self):
        return len(self.low_res_files)



class DIV2K_(torch.utils.data.Dataset):

    def __init__(self, mode, params):
        self.lr_dir = {
            common.modes.TRAIN: TRAIN_LR_DIR(params.source_path,params.tt, params.chunk, params.scale),
            common.modes.EVAL: EVAL_LR_DIR(params.source_path,params.tt, params.chunk, params.scale),
        }[mode]
        self.hr_dir = {
            common.modes.TRAIN: TRAIN_HR_DIR(params.source_path,params.tt, params.chunk, params.scale),
            common.modes.EVAL: EVAL_HR_DIR(params.source_path,params.tt, params.chunk, params.scale),
        }[mode]

        self.low_res_files = os.listdir(self.lr_dir)
        self.low_res_files = list_image_files(self.low_res_files)
        self.high_res_files = os.listdir(self.hr_dir)
        self.high_res_files = list_image_files(self.high_res_files)
    def __getitem__(self, index):
        low_res_path = self.low_res_files[index][1]
        high_res_path = self.high_res_files[index][1]

        low_res_image = Image.open(low_res_path).convert('RGB')
        high_res_image = Image.open(high_res_path).convert('RGB')

        low_res_image = transforms.functional.to_tensor(low_res_image)
        high_res_image = transforms.functional.to_tensor(high_res_image)

        return low_res_image, high_res_image

    def __len__(self):
        return len(self.low_res_files)


def list_image_files(d):
  file_name = sorted(os.listdir(d))
  file_name.sort(key=lambda x: int(x[:-4]))
  file_name = [(f, os.path.join(d, f)) for f in file_name if f.endswith('.png')]
  return file_name

