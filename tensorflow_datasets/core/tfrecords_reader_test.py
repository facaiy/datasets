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

"""Tests for tensorflow_datasets.core.tfrecords_reader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import absltest
import six

import tensorflow_datasets as tfds
from tensorflow_datasets import testing
from tensorflow_datasets.core import example_parser
from tensorflow_datasets.core import splits
from tensorflow_datasets.core import tfrecords_reader
from tensorflow_datasets.core import tfrecords_writer


class GetDatasetFilesTest(testing.TestCase):

  NAME2SHARD_LENGTHS = {
      'train': [3, 2, 3, 2, 3],  # 13 examples.
  }

  PATH_PATTERN = '/foo/bar/mnist-train.tfrecord-0000%d-of-00005'

  def _get_files(self, instruction):
    return tfrecords_reader._get_dataset_files(
        'mnist', '/foo/bar', instruction, self.NAME2SHARD_LENGTHS)

  def test_no_skip_no_take(self):
    instruction = tfrecords_reader._AbsoluteInstruction('train', None, None)
    files = self._get_files(instruction)
    self.assertEqual(files, [
        {'skip': 0, 'take': -1, 'filename': self.PATH_PATTERN % i}
        for i in range(5)])

  def test_skip(self):
    # One file is not taken, one file is partially taken.
    instruction = tfrecords_reader._AbsoluteInstruction('train', 4, None)
    files = self._get_files(instruction)
    self.assertEqual(files, [
        {'skip': 1, 'take': -1, 'filename': self.PATH_PATTERN % 1},
        {'skip': 0, 'take': -1, 'filename': self.PATH_PATTERN % 2},
        {'skip': 0, 'take': -1, 'filename': self.PATH_PATTERN % 3},
        {'skip': 0, 'take': -1, 'filename': self.PATH_PATTERN % 4},
    ])

  def test_take(self):
    # Two files are not taken, one file is partially taken.
    instruction = tfrecords_reader._AbsoluteInstruction('train', None, 6)
    files = self._get_files(instruction)
    self.assertEqual(files, [
        {'skip': 0, 'take': -1, 'filename': self.PATH_PATTERN % 0},
        {'skip': 0, 'take': -1, 'filename': self.PATH_PATTERN % 1},
        {'skip': 0, 'take': 1, 'filename': self.PATH_PATTERN % 2},
    ])

  def test_skip_take(self):
    # 2 elements in across two shards are taken in middle.
    instruction = tfrecords_reader._AbsoluteInstruction('train', 7, 9)
    files = self._get_files(instruction)
    self.assertEqual(files, [
        {'skip': 2, 'take': -1, 'filename': self.PATH_PATTERN % 2},
        {'skip': 0, 'take': 1, 'filename': self.PATH_PATTERN % 3},
    ])

  def test_touching_boundaries(self):
    # Nothing to read.
    instruction = tfrecords_reader._AbsoluteInstruction('train', 0, 0)
    files = self._get_files(instruction)
    self.assertEqual(files, [])

    instruction = tfrecords_reader._AbsoluteInstruction('train', None, 0)
    files = self._get_files(instruction)
    self.assertEqual(files, [])

    instruction = tfrecords_reader._AbsoluteInstruction('train', 3, 3)
    files = self._get_files(instruction)
    self.assertEqual(files, [])

    instruction = tfrecords_reader._AbsoluteInstruction('train', 13, None)
    files = self._get_files(instruction)
    self.assertEqual(files, [])

  def test_missing_shard_lengths(self):
    instruction = tfrecords_reader._AbsoluteInstruction('train', None, None)
    with self.assertRaisesWithPredicateMatch(
        AssertionError, 'S3 tfrecords_reader cannot be used'):
      tfrecords_reader._get_dataset_files(
          'mnist', '/foo/bar', instruction, {'train': None})


class ReadInstructionTest(testing.TestCase):

  def setUp(self):
    self.splits = {
        'train': 200,
        'test': 101,
        'validation': 30,
    }

  def check_translate(self, spec, expected, **kwargs):
    ri = tfrecords_reader.ReadInstruction(spec, **kwargs)
    res = ri.translate(self.splits)
    expected_result = []
    for split_name, from_, to_ in expected:
      expected_result.append(tfrecords_reader._AbsoluteInstruction(
          split_name, from_, to_))
    self.assertEqual(res, expected_result)
    return ri

  def assertRaises(self, spec, msg, **kwargs):
    with self.assertRaisesWithPredicateMatch(AssertionError, msg):
      ri = tfrecords_reader.ReadInstruction(spec, **kwargs)
      ri.translate(self.splits)

  def test_valid(self):
    # Simple split:
    ri = self.check_translate('train', [('train', None, None)])
    self.assertEqual(
        str(ri),
        ("ReadInstruction(["
         "_RelativeInstruction(splitname='train', from_=None, to=None, "
         "unit='abs', rounding='closest')])"))
    self.check_translate('test', [('test', None, None)])
    # Addition of splits:
    self.check_translate('train+test', [
        ('train', None, None),
        ('test', None, None),
    ])
    # Absolute slicing:
    self.check_translate('train[0:0]', [('train', None, 0)])
    self.check_translate('train[:10]', [('train', None, 10)])
    self.check_translate('train[0:10]', [('train', None, 10)])
    self.check_translate('train[-10:]', [('train', 190, None)])
    self.check_translate('train[-100:-50]', [('train', 100, 150)])
    self.check_translate('train[-10:200]', [('train', 190, None)])
    self.check_translate('train[10:-10]', [('train', 10, 190)])
    self.check_translate('train[42:99]', [('train', 42, 99)])
    # Percent slicing, no rounding:
    self.check_translate('train[:10%]', [('train', None, 20)])
    self.check_translate('train[90%:]', [('train', 180, None)])
    self.check_translate('train[-1%:]', [('train', 198, None)])
    # Percent slicing, with rounding:
    ri = self.check_translate('test[:99%]', [('test', None, 99)],
                              rounding='multiple1')
    self.assertEqual(
        str(ri),
        ("ReadInstruction([_RelativeInstruction(splitname='test', from_=None,"
         " to=99, unit='%', rounding='multiple1')])"))
    ri = self.check_translate('test[:99%]', [('test', None, 100)],
                              rounding='closest')
    self.assertEqual(
        str(ri),
        ("ReadInstruction([_RelativeInstruction(splitname='test', from_=None,"
         " to=99, unit='%', rounding='closest')])"))

  def test_invalid_rounding(self):
    with self.assertRaisesWithPredicateMatch(ValueError, 'rounding'):
      tfrecords_reader.ReadInstruction('test', rounding='unexisting_rounding')

  def test_duplicate_spec(self):
    # It is invalid to specify instructions using both string and args.
    msg = 'Instructions must either be given as str or args'
    self.assertRaises('validation[123:456]', msg, from_=123)
    self.assertRaises('validation[123:]', msg, to=789)
    self.assertRaises('validation[123:456]', msg, from_=1, to=789)
    self.assertRaises('validation[123:456]', msg, from_=1, to=789)
    self.assertRaises('validation[123:456]', msg, unit='%')

  def test_invalid_spec(self):
    # Invalid format:
    self.assertRaises('validation[:250%:2]',
                      'Unrecognized instruction format: validation[:250%:2]')
    # Unexisting split:
    self.assertRaises('imaginary',
                      'Requested split "imaginary" does not exist')
    # Invalid boundaries abs:
    self.assertRaises('validation[:31]',
                      'incompatible with 30 examples')
    # Invalid boundaries %:
    self.assertRaises('validation[:250%]',
                      'Percent slice boundaries must be > -100 and < 100')
    self.assertRaises('validation[:-101%]',
                      'Percent slice boundaries must be > -100 and < 100')


class ReaderTest(testing.TestCase):

  SPLIT_INFOS = [
      splits.SplitInfo(name='train', shard_lengths=[2, 3, 2, 3, 2]),  # 12 ex.
      splits.SplitInfo(name='test', shard_lengths=[2, 3, 2]),  # 7 ex.
  ]

  def setUp(self):
    super(ReaderTest, self).setUp()
    with absltest.mock.patch.object(example_parser,
                                    'ExampleParser', testing.DummyParser):
      self.reader = tfrecords_reader.Reader('some_spec')

  def _write_tfrecord(self, split_name, shards_number, records):
    path = os.path.join(self.tmp_dir, 'mnist-%s.tfrecord' % split_name)
    writer = tfrecords_writer._TFRecordWriter(path, len(records), shards_number)
    for rec in records:
      writer.write(six.b(rec))
    with absltest.mock.patch.object(tfrecords_writer, '_get_number_shards',
                                    return_value=shards_number):
      writer.finalize()

  def _write_tfrecords(self):
    self._write_tfrecord('train', 5, 'abcdefghijkl')
    self._write_tfrecord('test', 3, 'mnopqrs')

  def test_nodata_instruction(self):
    # Given instruction corresponds to no data.
    with self.assertRaisesWithPredicateMatch(AssertionError,
                                             'corresponds to no data!'):
      self.reader.read('mnist', '/foo/bar', 'train[0:0]', self.SPLIT_INFOS)

  def test_noskip_notake(self):
    self._write_tfrecord('train', 5, 'abcdefghijkl')
    ds = self.reader.read('mnist', self.tmp_dir, 'train', self.SPLIT_INFOS)
    read_data = list(tfds.as_numpy(ds))
    self.assertEqual(read_data, [six.b(l) for l in 'abcdefghijkl'])

  def test_complex(self):
    self._write_tfrecord('train', 5, 'abcdefghijkl')
    self._write_tfrecord('test', 3, 'mnopqrs')
    ds = self.reader.read('mnist', self.tmp_dir, 'train[1:-1]+test[:-50%]',
                          self.SPLIT_INFOS)
    read_data = list(tfds.as_numpy(ds))
    self.assertEqual(read_data, [six.b(l) for l in 'bcdefghijkmno'])


if __name__ == '__main__':
  testing.test_main()
