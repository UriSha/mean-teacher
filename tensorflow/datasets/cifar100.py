import os

import numpy as np
# import tensorflow as tf

from .utils import random_balanced_partitions, random_partitions


class Cifar100ZCA:
    DATA_PATH = os.path.join('data', 'images', 'cifar', 'cifar100', 'cifar100_gcn_zca_v2.npz')
    VALIDATION_SET_SIZE = 5000  # 10% of the training set
    UNLABELED = -1

    def __init__(self, data_seed=0, n_labeled='all', test_phase=False, mixup_coef = 4):
        random = np.random.RandomState(seed=data_seed)
        self._load()

        if test_phase:
            self.evaluation, self.training = self._test_and_training()
        else:
            self.evaluation, self.training = self._validation_and_training(random)

        if n_labeled != 'all':
            self.training = self._unlabel(self.training, n_labeled, random)
            # self.training = self._unlabel_mixup(self.training, n_labeled, random, mixup_coef)

    def _load(self):
        file_data = np.load(self.DATA_PATH)
        self._train_data = self._data_array(50000, file_data['train_x'], file_data['train_y'])
        self._test_data = self._data_array(10000, file_data['test_x'], file_data['test_y'])

    def _data_array(self, expected_n, x_data, y_data):
        array = np.zeros(expected_n, dtype=[
            ('x', np.float32, (32, 32, 3)),
            ('y', np.int32, ())  # We will be using -1 for unlabeled
        ])
        array['x'] = x_data
        array['y'] = y_data
        return array

    def _validation_and_training(self, random):
        return random_partitions(self._train_data, self.VALIDATION_SET_SIZE, random)

    def _test_and_training(self):
        return self._test_data, self._train_data

    def _unlabel(self, data, n_labeled, random):
        labeled, unlabeled = random_balanced_partitions(
            data, n_labeled, labels=data['y'], random=random)
        unlabeled['y'] = self.UNLABELED
        return np.concatenate([labeled, unlabeled])

    # def cshift(self, values):  # Circular shift in batch dimension
    #     return tf.concat([values[-1:, ...], values[:-1, ...]], 0)
    #
    # def _unlabel_mixup(self, data, n_labeled, random, mixup_coef, batch_size):
    #     labeled, unlabeled = random_balanced_partitions(
    #         data, n_labeled, labels=data['y'], random=random)
    #     unlabeled['y'] = self.UNLABELED
    #
    #     if mixup_coef > 0:
    #         mixup = 1.0 * mixup_coef  # Convert to float, as tf.distributions.Beta requires floats.
    #         beta = tf.distributions.Beta(mixup, mixup)
    #         lam = beta.sample(batch_size)
    #         ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
    #         labeled['x'] = ll * labeled['x'] + (1 - ll) * self.cshift(labeled['x'])
    #         labeled['y'] = lam * labeled['y'] + (1 - lam) * self.cshift(labeled['y'])
    #
    #     return np.concatenate([labeled, unlabeled])

    def _unlabel_mixup(self, data, n_labeled, random, mixup_coef=0):
        labeled, unlabeled = random_balanced_partitions(
            data, n_labeled, labels=data['y'], random=random)
        unlabeled['y'] = self.UNLABELED

        if mixup_coef == 0:
            return np.concatenate([labeled, unlabeled])

        labeled_x = labeled['x']
        labeled_y = labeled['y']

        mixed_data = np.zeros(10000, dtype=[
            ('x', np.float32, (32, 32, 3)),
            ('y', np.int32, ())  # We will be using -1 for unlabeled
        ])

        k = 0
        if labeled_y[0] == labeled_y[100]:
            print("mix these")

        for i in range(0, len(labeled_x) - 1, 2):
            if labeled_y[i] != labeled_y[i + 1]:
                continue

            mixup_coef = 1.0 * mixup_coef
            # beta = np.random.beta(mixup_coef, mixup_coef)
            # beta = tf.distributions.Beta(mixup_coef, mixup_coef)
            # lam = beta.sample(1)
            lam = np.random.beta(mixup_coef, mixup_coef)
            mixed_data[k]['x'] = labeled_x[i] * lam + labeled_x[i + 1] * (1 - lam)
            mixed_data[k]['y'] = labeled_y[i]
            k += 1

            # for lam in range(0.2, 1, 0.2):
            #     mixed_data[k]['x'] = labeled_x[i] * lam + labeled_x[i+1] * (1-lam)
            #     mixed_data[k]['y'] = labeled_y[i]
            #     k += 1

        mixed_data = mixed_data[:k]
        labeled = np.concatenate([labeled, mixed_data])

        return np.concatenate([labeled, unlabeled])
