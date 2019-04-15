# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defined Reader and ReadInstruction to read tfrecord files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import itertools
import math
import re

import numpy as np
import tensorflow as tf
from tensorflow_datasets.core import api_utils
from tensorflow_datasets.core import example_parser
from tensorflow_datasets.core import naming
from tensorflow_datasets.core import utils

_BUFFER_SIZE = 8<<20  # 8 MiB per file.

_SUB_SPEC_RE = re.compile(r'''
^
 (?P<split>\w+)
 (\[
  ((?P<from>-?\d+)
   (?P<from_pct>%)?)?
  :
  ((?P<to>-?\d+)
   (?P<to_pct>%)?)?
 \])?
$
''', re.X)

_ADDITION_SEP_RE = re.compile(r'\s*\+\s*')


def _set_dataset_options(dataset):
  """Applies optimization options to given dataset."""
  options = tf.data.Options()
  options.experimental_threading.max_intra_op_parallelism = 1
  options.experimental_threading.private_threadpool_size = 16
  options.experimental_optimization.apply_default_optimizations = True
  options.experimental_optimization.map_fusion = True
  options.experimental_optimization.map_parallelization = True
  return dataset.with_options(options)


def _get_dataset_from_filename(filename_skip_take, do_skip, do_take):
  """Returns a tf.data.Dataset instance from given (filename, skip, take)."""
  filename, skip, take = (filename_skip_take['filename'],
                          filename_skip_take['skip'],
                          filename_skip_take['take'],)
  dataset = tf.data.TFRecordDataset(
      filename,
      buffer_size=_BUFFER_SIZE,
      num_parallel_reads=1,
      )
  if do_skip:
    dataset = dataset.skip(skip)
  if do_take:
    dataset = dataset.take(take)
  dataset = _set_dataset_options(dataset)
  return dataset


def _get_dataset_files(name, path, instruction, name2shard_lengths):
  """Returns a list of files (+skip/take) corresponding to given instruction.

  This is the core of the reading logic, to translate from absolute instructions
  (split + left/right boundaries) to files + skip/take.

  Args:
    name: Name of the dataset.
    path: path to tfrecords.
    instruction: _AbsoluteInstruction instance.
    name2shard_lengths: dict associating number of examples to split names.

  Returns:
    list of dict(filename, skip, take).
  """
  shard_lengths = name2shard_lengths[instruction.splitname]
  if not shard_lengths:
    msg = ('`DatasetInfo.SplitInfo.num_shards` is empty. S3 tfrecords_reader '
           'cannot be used. Make sure the data you are trying to read was '
           'generated using tfrecords_writer module (S3).')
    raise AssertionError(msg)
  filenames = naming.filepaths_for_dataset_split(
      dataset_name=name, split=instruction.splitname,
      num_shards=len(shard_lengths),
      data_dir=path,
      filetype_suffix='tfrecord')
  from_ = 0 if instruction.from_ is None else instruction.from_
  to = sum(shard_lengths) if instruction.to is None else instruction.to
  index_start = 0  # Beginning (included) of moving window.
  index_end = 0  # End (excluded) of moving window.
  files = []
  for filename, length in zip(filenames, shard_lengths):
    index_end += length
    if from_ < index_end and to > index_start:  # There is something to take.
      skip = from_ - index_start if from_ > index_start else 0
      take = to - index_start if to < index_end else -1
      files.append(dict(filename=filename, skip=skip, take=take))
    index_start += length
  return files


class Reader(object):
  """Build a tf.data.Dataset object out of Instruction instance(s).

  This class should not typically be exposed to the TFDS user.
  It replaces file_format_adapter.TFRecordExampleAdapter.dataset_from_filename
  and dataset_utils._build_{ds_from_instruction,instruction_ds,mask_ds}
  functions which will eventually be deleted once S3 is fully rolled-out.
  """

  def __init__(self, example_specs):
    self._parser = example_parser.ExampleParser(example_specs)

  def _read_single_instruction(self, instruction,
                               name, path, name2len, name2shard_lengths):
    """Returns tf.data.Dataset for instruction of the form 'test+train[:-10%]'."""
    if not isinstance(instruction, ReadInstruction):
      instruction = ReadInstruction(instruction)
    absolute_instructions = instruction.translate(name2len)
    files = list(itertools.chain.from_iterable([
        _get_dataset_files(name, path, abs_instr, name2shard_lengths)
        for abs_instr in absolute_instructions]))
    if not files:
      msg = 'Instruction "%s" corresponds to no data!' % instruction
      raise AssertionError(msg)

    do_skip = any(f['skip'] > 0 for f in files)
    do_take = any(f['take'] > -1 for f in files)

    # Transpose the list[dict] into dict[list]
    tensor_inputs = {
        # skip/take need to be converted to int64 explicitly
        k: list(vals) if k == 'filename' else np.array(vals, dtype=np.int64)
        for k, vals in utils.zip_dict(*files)
    }

    dataset = tf.data.Dataset.from_tensor_slices(tensor_inputs).interleave(
        functools.partial(_get_dataset_from_filename,
                          do_skip=do_skip, do_take=do_take),
        cycle_length=16,  # len(files),
        block_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    # TODO(pierrot): `parse_example` uses
    # `tf.io.parse_single_example`. It might be faster to use `parse_example`,
    # after batching.
    # https://www.tensorflow.org/api_docs/python/tf/io/parse_example
    return dataset.map(self._parser.parse_example)

  def read(self, name, path, instructions, split_infos):
    """Returns tf.data.Dataset instance(s).

    Args:
      name (str): name of the dataset.
      path (str): path where tfrecords are stored.
      instructions (ReadInstruction, List[], Dict[]): instruction(s) to read.
        Instructions can be string and will then be passed to the Instruction
        constructor as it.
      split_infos (list of SplitInfo proto): the available splits for dataset.

    Returns:
       a single tf.data.Dataset instance if instruction is a single
       ReadInstruction instance. Otherwise a dict/list of tf.data.Dataset
       corresponding to given instructions param shape.
    """
    name2shard_lengths = {info.name: info.shard_lengths for info in split_infos}
    name2len = {name: sum(lengths)
                for name, lengths in name2shard_lengths.items()}
    read_instruction = functools.partial(
        self._read_single_instruction,
        name=name, path=path,
        name2len=name2len, name2shard_lengths=name2shard_lengths)
    datasets = utils.map_nested(read_instruction, instructions, map_tuple=True)
    return datasets


class _AbsoluteInstruction(collections.namedtuple('_AbsoluteInstruction', [
    'splitname',  # str.
    'from_',  # uint (starting index) or None if no lower boundary.
    'to',  # uint (ending index) or None if no upper boundary.
])):
  """Represents a machine friendly instruction (absolute boundaries)."""


class _RelativeInstruction(collections.namedtuple('_RelativeInstruction', [
    'splitname',  # str.
    'from_',  # uint (starting index) or None if no lower boundary.
    'to',  # uint (ending index) or None if no upper boundary.
    'unit',  # str, one of {'%', 'abs'}.
    'rounding',  # str, one of {'closest', 'pct1'}.
])):
  """Represents a parsed relative instruction."""


def _str_to_relative_instruction(spec, from_, to, unit, rounding):
  """Given a relative instruction as str, returns _RelativeInstruction."""
  res = _SUB_SPEC_RE.match(spec)
  if not res:
    raise AssertionError('Unrecognized instruction format: %s' % spec)
  if (any([res.group('from'), res.group('to')]) and any([from_, to, unit])):
    raise AssertionError('Instructions must either be given as str or args.')
  unit = '%' if res.group('from_pct') or res.group('to_pct') else 'abs'
  instr = _RelativeInstruction(
      splitname=res.group('split'),
      from_=int(res.group('from')) if res.group('from') else from_,
      to=int(res.group('to')) if res.group('to') else to,
      unit=unit,
      rounding=rounding)
  if instr.unit == '%' and (
      (instr.from_ is not None and abs(instr.from_) > 100) or
      (instr.to is not None and abs(instr.to) > 100)):
    raise AssertionError('Percent slice boundaries must be > -100 and < 100.')
  return instr


def _pct_to_abs_multiple1(boundary, num_examples):
  return boundary * math.trunc(num_examples / 100.)


def _pct_to_abs_closest(boundary, num_examples):
  return round(boundary * num_examples / 100.)


def _rel_to_abs_instr(rel_instr, name2len):
  """Returns _AbsoluteInstruction instance for given RelativeInstruction.

  Args:
    rel_instr: RelativeInstruction instance.
    name2len: dict {split_name: num_examples}.
  """
  pct_to_abs = (_pct_to_abs_closest if rel_instr.rounding == 'closest'
                else _pct_to_abs_multiple1)
  split = rel_instr.splitname
  if split not in name2len:
    raise AssertionError('Requested split "%s" does not exist.' % split)
  num_examples = name2len[split]
  from_ = rel_instr.from_
  to = rel_instr.to
  if rel_instr.unit == '%':
    from_ = 0 if from_ is None else pct_to_abs(from_, num_examples)
    to = num_examples if to is None else pct_to_abs(to, num_examples)
  else:
    from_ = 0 if from_ is None else from_
    to = num_examples if to is None else to
  if abs(from_) > num_examples or abs(to) > num_examples:
    msg = 'Requested slice [%s:%s] incompatible with %s examples.' % (
        from_ or '', to or '', num_examples)
    raise AssertionError(msg)
  if from_ < 0:
    from_ = num_examples + from_
  elif from_ == 0:
    from_ = None
  if to < 0:
    to = num_examples + to
  elif to == num_examples:
    to = None
  return _AbsoluteInstruction(split, from_, to)


class ReadInstruction(object):
  """Reading instruction for a dataset.

  Args:
    spec (str): split(s) + optional slice(s) to read. A slice can be
      specified, using absolute numbers (int) or percentages (int). E.g.
        `test`: test split.
        `test + validation`: test split + validation split.
        `test[10:]`: test split, minus its first 10 records.
        `test[:10%]`: first 10% records of test split.
        `test[:-5%]+train[40%:60%]`: first 95% of test + middle 20% of train.
    rounding (str): The rounding behaviour to use when percent slicing is used.
      Ignored when slicing with absolute indices.
      Possible values:
       - 'closest' (default): The specified percentages are rounded to the
         closest value. Use this if you want specified percents to be as much
         exact as possible.
       - 'multiple1': the specified percentages are treated as multiple of 1%.
         Use this option if you want consistency. Eg: len(5%) == 5 * len(1%).
    from_ (int):
    to (int): alternative way of specifying slicing boundaries. If any of
      {from_, to, unit} argument is used, slicing cannot be specified as string.
    unit (str): optional, one of:
      '%': to set the slicing unit as percents of the split size.
      'abs': to set the slicing unit as absolute numbers.

  Examples of usage:

  # The following lines are equivalent:
  ds = tfds.load('mnist', ReadInstruction('test[:33%]')
  ds = tfds.load('mnist', ReadInstruction('test', to=33, unit='%')
  ds = tfds.load('mnist', ReadInstruction('test', from_=0, to=33, unit='%')

  # The following lines are equivalent:
  ds = tfds.load('mnist', ReadInstruction('test[:33%]+train[1:-1]')
  ds = tfds.load('mnist', ReadInstruction('test[:33%]+train[1:-1]')

  # 10-fold validation:
  tests = tfds.load(
      'mnist',
      [ReadInstruction('train', from_=k, to=k+10, unit='%')
       for k in range(0, 100, 10)])
  trains = tfds.load(
      'mnist',
      [RI('train', to=k, unit='%') + RI('train', from_=k+10, unit='%')
       for k in range(0, 100, 10)])
  """

  @api_utils.disallow_positional_args(allowed=['spec'])
  def __init__(self, spec, rounding='closest', from_=None, to=None, unit=None):
    if rounding not in ('multiple1', 'closest'):
      raise ValueError('Wrong arg `rounding`: "%s".' % rounding)
    spec = str(spec)  # Need to convert to str in case of NamedSplit instance.
    self._relative_instructions = [
        _str_to_relative_instruction(sub, from_, to, unit, rounding)
        for sub in _ADDITION_SEP_RE.split(spec)]

  def __str__(self):
    return 'ReadInstruction(%s)' % self._relative_instructions

  def translate(self, name2len):
    """Translate instruction into a list of absolute instructions (to be added).

    Args:
      name2len: dict associating split names to number of examples.

    Returns:
      list of _AbsoluteInstruction instances (corresponds to the + in spec).
    """
    return [_rel_to_abs_instr(rel_instr, name2len)
            for rel_instr in self._relative_instructions]
