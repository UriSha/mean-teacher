"""Vary dropout parameter on 250-label SVHN for the NIPS paper"""

import logging
import sys

from .run_context import RunContext
import tensorflow as tf

from datasets import SVHN
from mean_teacher.model import Model
from mean_teacher import minibatching


LOG = logging.getLogger('main')


def all_parameters():
    n_runs = 4
    for data_seed in range(1000, 1000 + n_runs):
        for student_dropout_probability in [0, 0.25, 0.5, 0.75]:
            for teacher_dropout_probability in [0, 0.25, 0.5, 0.75]:
                if student_dropout_probability in [teacher_dropout_probability, 0.5]:
                    yield {
                        'data_seed': data_seed,
                        'student_dropout_probability': student_dropout_probability,
                        'teacher_dropout_probability': teacher_dropout_probability,
                        'model_type': 'pi'
                    }


def parameters():
    for idx, param in enumerate(all_parameters()):
        if idx >= 21:
            yield param


def model_hyperparameters(model_type, n_labeled, n_extra_unlabeled):
    assert model_type in ['mean_teacher', 'pi']
    training_length = {
        0: 180000,
        100000: 400000,
        500000: 600000,
    }
    if n_labeled == 'all':
        return {
            'training_length': training_length[n_extra_unlabeled],
            'n_labeled_per_batch': 100,
            'max_consistency_cost': 100.0,
            'apply_consistency_to_labeled': True,
            'ema_consistency': model_type == 'mean_teacher'
        }
    elif isinstance(n_labeled, int):
        return {
            'training_length': training_length[n_extra_unlabeled],
            'n_labeled_per_batch': 1,
            'max_consistency_cost': 1.0,
            'apply_consistency_to_labeled': False,
            'ema_consistency': model_type == 'mean_teacher'
        }
    else:
        msg = "Unexpected combination: {model_type}, {n_labeled}, {n_extra_unlabeled}"
        assert False, msg.format(locals())


def run(data_seed, student_dropout_probability, teacher_dropout_probability,
        test_phase=False, n_labeled=250, n_extra_unlabeled=0, model_type='mean_teacher'):
    minibatch_size = 100
    hyperparams = model_hyperparameters(model_type, n_labeled, n_extra_unlabeled)

    tf.reset_default_graph()
    model = Model(RunContext(__file__, data_seed))

    svhn = SVHN(n_labeled=n_labeled,
                n_extra_unlabeled=n_extra_unlabeled,
                data_seed=data_seed,
                test_phase=test_phase)

    model['ema_consistency'] = hyperparams['ema_consistency']
    model['max_consistency_cost'] = hyperparams['max_consistency_cost']
    model['apply_consistency_to_labeled'] = hyperparams['apply_consistency_to_labeled']
    model['training_length'] = hyperparams['training_length']
    model['student_dropout_probability'] = student_dropout_probability
    model['teacher_dropout_probability'] = teacher_dropout_probability

    training_batches = minibatching.training_batches(svhn.training,
                                                     minibatch_size,
                                                     hyperparams['n_labeled_per_batch'])
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(svhn.evaluation,
                                                                    minibatch_size)

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
