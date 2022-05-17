"""C_index_loss_test.py."""

import tensorflow as tf
import numpy as np

import C_index_loss


class CITest(tf.test.TestCase):

    def test_get_equal_times(self):
        with self.test_session():
            t_sim = np.ones((3, 1))
            expected = tf.ones((3, 3))
            expected = tf.linalg.set_diag(expected, tf.zeros(3, dtype='float32'))
            actual = C_index_loss.get_equal_times(t_sim)
            self.assertAllClose(expected, actual)
            t_sim = np.zeros((3, 1))
            t_sim[0, 0] = 1
            t_sim[1, 0] = 3
            t_sim[2, 0] = 2
            expected = tf.zeros((3, 3))
            actual = C_index_loss.get_equal_times(t_sim)
            self.assertAllClose(expected, actual)

    def test_get_greater_times(self):
        with self.test_session():
            t_sim = np.ones((3, 1))
            expected = tf.zeros((3, 3))
            actual = C_index_loss.get_greater_times(t_sim)
            self.assertAllClose(expected, actual)
            t_sim = np.zeros((3, 1))
            t_sim[0, 0] = 1
            t_sim[1, 0] = 3
            t_sim[2, 0] = 2
            expected = np.zeros((3, 3))
            expected[1, 0] = 1.
            expected[2, 0] = 1.
            expected[1, 2] = 1.
            expected = tf.cast(expected, dtype="float32")
            actual = C_index_loss.get_greater_times(t_sim)
            self.assertAllClose(expected, actual)

    def test_c_index(self):
        with self.test_session():
            # if f is ordered in inverselly to surv[:,0], it should be 1.
            surv = np.zeros((3, 2))
            surv[0, 1] = 0
            surv[1, 1] = 0
            surv[2, 1] = 0
            surv[0, 0] = 1
            surv[1, 0] = 2
            surv[2, 0] = 3
            surv = tf.cast(surv, dtype="float32")
            f = np.zeros((3, 1))
            f[0, 0] = 3
            f[1, 0] = 2
            f[2, 0] = 1
            expected = tf.cast(-1, dtype='float32')
            actual = C_index_loss.c_index_loss(surv, f)
            self.assertEqual(expected, actual)
            # if f is ordered in inverselly to surv[:,0], it should be 0.
            surv = np.zeros((3, 2))
            surv[0, 1] = 0
            surv[1, 1] = 0
            surv[2, 1] = 0
            surv[0, 0] = 1
            surv[1, 0] = 2
            surv[2, 0] = 3
            surv = tf.cast(surv, dtype="float32")
            f = np.zeros((3, 1))
            f[0, 0] = 1
            f[1, 0] = 2
            f[2, 0] = 3
            expected = tf.cast(-0., dtype='float32')
            actual = C_index_loss.c_index_loss(surv, f)
            self.assertEqual(expected, actual)
            # test ties.
            surv = np.zeros((3, 2))
            surv[0, 1] = 0
            surv[1, 1] = 0
            surv[2, 1] = 0
            surv[0, 0] = 1
            surv[1, 0] = 2
            surv[2, 0] = 2
            surv = tf.cast(surv, dtype="float32")
            f = np.zeros((3, 1))
            f[0, 0] = 2
            f[1, 0] = 1
            f[2, 0] = 1
            expected = tf.cast(-1., dtype='float32')
            actual = C_index_loss.c_index_loss(surv, f)
            self.assertEqual(expected, actual)
            # only two  not censored events.
            surv = np.zeros((3, 2))
            surv[0, 1] = 0
            surv[1, 1] = 1
            surv[2, 1] = 1
            surv[0, 0] = 1
            surv[1, 0] = 2
            surv[2, 0] = 3
            surv = tf.cast(surv, dtype="float32")
            f = np.zeros((3, 1))
            f[0, 0] = 3
            f[1, 0] = 1
            f[2, 0] = 2
            expected = tf.cast(-1., dtype='float32')
            actual = C_index_loss.c_index_loss(surv, f)
            self.assertEqual(expected, actual)


if __name__ == '__main__':
    tf.test.main()

surv = np.zeros((3, 2))
surv[0, 1] = 1
surv[1, 1] = 1
surv[2, 1] = 1
surv[0, 0] = 1
surv[1, 0] = 2
surv[2, 0] = 3

f = np.zeros((3, 1))
f[0, 0] = 3
f[1, 0] = 2
f[2, 0] = 1
