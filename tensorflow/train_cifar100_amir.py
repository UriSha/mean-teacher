"""Train ConvNet Mean Teacher on CIFAR-10 training set and evaluate against a validation set

This runner converges quickly to a fairly good accuracy.
On the other hand, the runner experiments/cifar10_final_eval.py
contains the hyperparameters used in the paper, and converges
much more slowly but possibly to a slightly better accuracy.
"""

import logging
import tensorflow as tf
flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_float('entropy_reg', 0.01, 'Entropy regularization coefficient.')

from datetime import datetime

from experiments.run_context import RunContext
from datasets import Cifar100ZCA
from mean_teacher.model import Model
#from mean_teacher.myresnet_model import Model
from mean_teacher import minibatching


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('main')


def run(data_seed=0):
    n_labeled = 10000

    print('Running')

    model = Model(RunContext(__file__, 0))

    model['entropy_factor'] = FLAGS.entropy_reg

    model['flip_horizontally'] = True
    model['adam_beta_2_during_rampup'] = 0.999
    model['ema_decay_during_rampup'] = 0.999
    model['normalize_input'] = False  # Keep ZCA information
    model['rampdown_length'] = 25000
    model['training_length'] = 150000

    print('Reg:' , FLAGS.entropy_reg)
    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    cifar = Cifar100ZCA(data_seed, n_labeled)
    if n_labeled == 'all':
        n_labeled_per_batch = 100
    else:
        n_labeled_per_batch = 50
    print('N Labeled Per Batch: ', n_labeled_per_batch)
    training_batches = minibatching.training_batches(cifar.training, batch_size = 100)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(cifar.evaluation)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    run()




