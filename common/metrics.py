"""Metrics."""

import torch
import torch.nn.functional as F


def psnr_mse(sr, hr, shave=4):
  sr = sr.to(hr.dtype)
  sr = (sr * 255).round().clamp(0, 255) / 255
  diff = sr - hr
  if shave:
    diff = diff[..., shave:-shave, shave:-shave]
  mse = diff.pow(2).mean([-3, -2, -1])
  # psnr = -10 * mse.log10()
  return torch.sum(mse)

def psnr(sr, hr, shave=4):
  sr = sr.to(hr.dtype)
  sr = (sr * 255).round().clamp(0, 255) / 255
  diff = sr - hr
  if shave:
    diff = diff[..., shave:-shave, shave:-shave]
  mse = diff.pow(2).mean([-3, -2, -1])
  psnr = -10 * mse.log10()
  return psnr


