import os

import numpy as np
# import tensorflow as tf

from .utils import random_balanced_partitions, random_partitions


class Cifar100ZCA:
    DATA_PATH = os.path.join('data', 'images', 'cifar', 'cifar100', 'cifar100_gcn_zca_v2.npz')
    VALIDATION_SET_SIZE = 5000  # 10% of the training set
    NUM_OF_FINE_LABELS = 100
    NUM_OF_COARSE_LABELS = 20
    NUM_OF_FINE_LABELS_PER_COARSE_LABELS = 20
    UNLABELED = -1
    UNLABELED_VECTOR = np.zeros(NUM_OF_FINE_LABELS, )

    def __init__(self, data_seed=0, n_labeled='all', test_phase=False, mixup_coef=0, n_mixed_examples=0):
        random = np.random.RandomState(seed=data_seed)
        self._load()

        if test_phase:
            self.evaluation, self.training = self._test_and_training()
        else:
            self.evaluation, self.training = self._validation_and_training(random)

        if n_labeled != 'all':
            # self.training = self._unlabel(self.training, n_labeled, random)
            # self.training = self._unlabel_labels_as_vectors(self.training, n_labeled, random)
            self.training = self._unlabel_mixup_labels_as_vectors(self.training, n_labeled, random, mixup_coef,
                                                                  n_mixed_examples)

    def _load(self):
        file_data = np.load(self.DATA_PATH)
        # self._train_data = self._data_array(50000, file_data['train_x'], file_data['train_y'])
        # self._test_data = self._data_array(10000, file_data['test_x'], file_data['test_y'])
        self._train_data = self._data_array_labels_as_vectors(50000, file_data['train_x'], file_data['train_y'],
                                                              file_data['train_z'])
        self._test_data = self._data_array_labels_as_vectors(10000, file_data['test_x'], file_data['test_y'],
                                                             file_data['test_z'])

    def _data_array(self, expected_n, x_data, y_data):
        array = np.zeros(expected_n, dtype=[
            ('x', np.float32, (32, 32, 3)),
            ('y', np.int32, ())  # We will be using -1 for unlabeled
        ])
        array['x'] = x_data
        array['y'] = y_data
        return array

    def one_hot_labels(self, labels):
        return np.eye(100)[labels]

    def _data_array_labels_as_vectors(self, expected_n, x_data, y_data, z_data):
        array = np.zeros(expected_n, dtype=[
            ('x', np.float32, (32, 32, 3)),
            ('y', np.float32, (self.NUM_OF_FINE_LABELS,)),  # ('y', np.zeros(), ())  # We will be using -1 for unlabeled
            ('z', np.int32, ())]
                         )

        array['x'] = x_data
        # for i in range(len(y_data)):
        #     array['y'][i][y_data[i]] = 1.0
        array['y'] = self.one_hot_labels(y_data)
        array['z'] = z_data
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

    def _unlabel_labels_as_vectors(self, data, n_labeled, random):
        labels_as_ints = [np.where(entry == 1)[0][0] for entry in data['y']]

        labeled, unlabeled = random_balanced_partitions(
            data, n_labeled, labels=labels_as_ints, random=random)
        for i in range(len(unlabeled['y'])):
            unlabeled['y'][i] = self.UNLABELED_VECTOR
        return np.concatenate([labeled, unlabeled])

    def _mixup_labeled_data(self, labeled_data, random, mixup_coef, n_mixed_examples):
        labeled_sorted_by_cluster = np.sort(labeled_data, order=['z', 'y'])
        labeled_x = labeled_sorted_by_cluster['x']
        labeled_y = labeled_sorted_by_cluster['y']
        labeled_z = labeled_sorted_by_cluster['z']

        mixed_data = np.zeros(n_mixed_examples, dtype=[
            ('x', np.float32, (32, 32, 3)),
            ('y', np.float32, (self.NUM_OF_FINE_LABELS,)),  # ('y', np.zeros(), ())  # We will be using -1 for unlabeled
            ('z', np.float32, (self.NUM_OF_COARSE_LABELS,))]
                              )

        n_mixed_examples_per_fine_label_pair = n_mixed_examples / 200
        mixup_coef = 1.0 * mixup_coef

        num_of_mixed_examples = 0
        for l in range(self.NUM_OF_COARSE_LABELS):
            for i in range(self.NUM_OF_FINE_LABELS_PER_COARSE_LABELS):
                for j in range(i + 1, self.NUM_OF_FINE_LABELS_PER_COARSE_LABELS):
                    indices_i = random.choice(self.NUM_OF_FINE_LABELS, n_mixed_examples_per_fine_label_pair)
                    indices_j = random.choice(self.NUM_OF_FINE_LABELS, n_mixed_examples_per_fine_label_pair)
                    for k in range(n_mixed_examples_per_fine_label_pair):
                        lam = np.random.beta(mixup_coef, mixup_coef)
                        l_factor = l * 500
                        first = l_factor + i * 100 + indices_i[k]
                        second = l_factor + j * 100 + indices_j[k]
                        mixed_data[num_of_mixed_examples]['x'] = labeled_x[first] * lam + labeled_x[second] * (1 - lam)
                        mixed_data[num_of_mixed_examples]['y'] = labeled_y[first] * lam + labeled_y[second] * (1 - lam)
                        mixed_data[num_of_mixed_examples]['z'] = labeled_z[l]
                        num_of_mixed_examples += 1

        return mixed_data

    def _unlabel_mixup_labels_as_vectors(self, data, n_labeled, random, mixup_coef, n_mixed_examples):
        labels_as_ints = [np.where(entry == 1)[0][0] for entry in data['y']]

        labeled, unlabeled = random_balanced_partitions(
            data, n_labeled, labels=labels_as_ints, random=random)
        for i in range(len(unlabeled['y'])):
            unlabeled['y'][i] = self.UNLABELED_VECTOR

        if mixup_coef == 0:
            return np.concatenate([labeled, unlabeled])

        mixed_data = self._mixup_labeled_data(labeled, random, mixup_coef, n_mixed_examples)

        labeled = np.concatenate([labeled, mixed_data])

        return np.concatenate([labeled, unlabeled])
