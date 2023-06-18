"""File IO helper."""

import warnings


class Hdf5:

  def __init__(self, fname, lib='h5py'):
    self.fname = fname
    self.lib = lib
    self.file = None

  def add(self, key, value):
    if self.lib == 'h5py':
      import h5py
      with h5py.File(self.fname, 'a', libver='latest') as f:
        f.create_dataset(
            key,
            data=value,
            maxshape=value.shape,
            compression='lzf',
            shuffle=True,
            track_times=False,
            #track_order=False,
        )
    elif self.lib == 'pytables':
      import tables
      filters = tables.Filters(complevel=8, complib='blosc', bitshuffle=True)
      original_warnings = list(warnings.filters)
      warnings.simplefilter('ignore', tables.NaturalNameWarning)
      with tables.File(self.fname, 'a', filters=filters) as f:
        f.create_carray(
            f.root,
            key,
            obj=value,
            track_times=False,
        )
      warnings.filters = original_warnings
    else:
      raise NotImplementedError

  def get(self, key):
    if self.lib == 'h5py':
      if not self.file:
        import h5py
        self.file = h5py.File(self.fname, 'r', libver='latest')
      return self.file[key]
    elif self.lib == 'pytables':
      if not self.file:
        import tables
        self.file = tables.File(self.fname, 'r')
      return self.file.root[key]
    else:
      raise NotImplementedError
