"""Tests for vocab_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import tensorflow as tf

from ..utils import vocab_utils


class VocabUtilsTest(tf.test.TestCase):

  def testCheckVocab(self):
    # Create a vocab file
    vocab_dir = os.path.join(tf.test.get_temp_dir(), "vocab_dir")
    os.makedirs(vocab_dir)
    vocab_file = os.path.join(vocab_dir, "vocab_file")
    vocab = ["a", "b", "c"]
    with codecs.getwriter("utf-8")(tf.gfile.GFile(vocab_file, "wb")) as f:
      for word in vocab:
        f.write("%s\n" % word)

    # Call vocab_utils
    out_dir = os.path.join(tf.test.get_temp_dir(), "out_dir")
    os.makedirs(out_dir)
    vocab_size, new_vocab_file = vocab_utils.check_vocab(
        vocab_file, out_dir)

    # Assert: we expect the code to add  <unk>, <s>, </s> and
    # create a new vocab file
    self.assertEqual(len(vocab) + 3, vocab_size)
    self.assertEqual(os.path.join(out_dir, "vocab_file"), new_vocab_file)
    new_vocab, _ = vocab_utils.load_vocab(new_vocab_file)
    self.assertEqual(
        [vocab_utils.UNK, vocab_utils.SOS, vocab_utils.EOS] + vocab, new_vocab)


if __name__ == "__main__":
  tf.test.main()
