"""Tests for vocab_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ..utils import misc_utils


class MiscUtilsTest(tf.test.TestCase):

  def testFormatBpeText(self):
    bpe_line = (
        b"En@@ ough to make already reluc@@ tant men hesitate to take screening"
        b" tests ."
    )
    expected_result = (
        b"Enough to make already reluctant men hesitate to take screening tests"
        b" ."
    )
    self.assertEqual(expected_result,
                     misc_utils.format_bpe_text(bpe_line.split(b" ")))

  def testFormatSPMText(self):
    spm_line = u"\u2581This \u2581is \u2581a \u2581 te st .".encode("utf-8")
    expected_result = "This is a test."
    self.assertEqual(expected_result,
                     misc_utils.format_spm_text(spm_line.split(b" ")))


if __name__ == "__main__":
  tf.test.main()
