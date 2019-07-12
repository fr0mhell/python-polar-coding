from unittest import TestCase

from utils import int_to_bin_list


class UtilsTestCase(TestCase):

    def test_int_to_bin_list(self):
        len_a = 8
        a = pow(2, len_a) - 1
        a_bin = [1 for _ in range(len_a)]
        self.assertEqual(int_to_bin_list(a, len_a), a_bin)

        len_b = 16
        b = pow(2, len_b) - 1
        b_bin = [1 for _ in range(len_b)]
        self.assertEqual(int_to_bin_list(b, len_b), b_bin)
