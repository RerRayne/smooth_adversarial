import functools
import jax
import numpy as np
import random
import sys

import flax.optim as optim
import flax.jax_utils as flax_utils
from flax.training import common_utils

import input_pipeline
import models
import utils

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_integer('number_epochs', 25, '')
flags.DEFINE_string('dataset', 'cifar10', '')
flags.DEFINE_string('model', 'PreResNet18', 'architecture to use')
flags.DEFINE_integer('test_every_steps', 100, '')
flags.DEFINE_integer('save_every', 100, '')
flags.DEFINE_string('checkpoint_name', './model.npz', '')
flags.DEFINE_integer('crop_size', 32, '')
flags.DEFINE_integer('resize_size', 32, '')
flags.DEFINE_float('lr', 0.2, '')
flags.DEFINE_float('eps', 8.0 / 255.0, '')
flags.DEFINE_float('alpha', 10.0 / 255.0, '')
flags.DEFINE_float('pgd_alpha', 2.0 / 255.0, '')
flags.DEFINE_integer('pgd_restarts', 10, '')


models = {
    'ResNet50': models.ResNet50,
    'ResNet101': models.ResNet101,
    'ResNet152': models.ResNet152,
    'ResNet50x2': models.ResNet50x2,
    'ResNet101x2': models.ResNet101x2,
    'ResNet152x2': models.ResNet152x2,
    'ResNext50_32x4d': models.ResNext50_32x4d,
    'ResNext101_32x8d': models.ResNext101_32x8d,
    'ResNext152_32x4d': models.ResNext152_32x4d,
    'PreResNet18': models.PreActResNet18,
}


def main(argv):
    del argv

    seed = random.Random().randint(0, sys.maxsize)
    rnd_key = jax.random.PRNGKey(seed)

    train_info = input_pipeline.get_dataset_info(FLAGS.dataset, 'train', examples_per_class=None)
    batches_per_train = int(np.ceil(train_info['num_examples'] / FLAGS.batch_size))

    test_info = input_pipeline.get_dataset_info(FLAGS.dataset, 'test', examples_per_class=None)
    batches_per_test = int(np.ceil(test_info['num_examples'] / FLAGS.batch_size))

    train_steps = batches_per_train * FLAGS.number_epochs
    test_steps = batches_per_test

    model = models[FLAGS.model]

    # Create dataset
    train_ds = input_pipeline.get_data(dataset=FLAGS.dataset,
                                       mode='train',
                                       repeats=None,
                                       batch_size=FLAGS.batch_size,
                                       resize_size=FLAGS.resize_size,
                                       crop_size=FLAGS.crop_size,
                                       mixup_alpha=None,
                                       examples_per_class=None,
                                       examples_per_class_seed=None,
                                       num_devices=jax.device_count(),
                                       tfds_manual_dir=None)

    test_ds = input_pipeline.get_data(dataset=FLAGS.dataset,
                                      mode='test',
                                      repeats=None,
                                      batch_size=FLAGS.batch_size,
                                      resize_size=FLAGS.resize_size,
                                      crop_size=FLAGS.crop_size,
                                      mixup_alpha=None,
                                      examples_per_class=None,
                                      examples_per_class_seed=None,
                                      num_devices=jax.device_count(),
                                      tfds_manual_dir=None)

    # Build ResNet architecture
    # TODO: add the rest off
    model_creator = model.partial(num_outputs=train_info['num_classes'])
    model, init_state = utils.create_model(rnd_key, FLAGS.batch_size, FLAGS.resize_size, 3, model_creator)
    state_repl = flax_utils.replicate(init_state)

    # Create optimizer and replicate it over all GPUs
    opt = optim.Momentum(beta=0.9, weight_decay=5e-4).create(model)
    opt_repl = flax_utils.replicate(opt)

    # Delete referenes to the objects that are not needed anymore
    del opt
    del init_state

    # Create function for training
    lr_fn = utils.cyclic_lr(FLAGS.lr, train_steps)
    update_fn = jax.pmap(functools.partial(utils.update_stateful,
                                           eps=FLAGS.eps,
                                           alpha=FLAGS.alpha),
                         axis_name='batch')
    p_eval_step = jax.pmap(utils.eval_step, axis_name='batch')
    p_eval_robust_step = jax.pmap(functools.partial(utils.robust_eval_step,
                                                    eps=FLAGS.eps,
                                                    pgd_alpha=FLAGS.pgd_alpha),
                                  axis_name='batch')

    # Metrics
    steps = []
    accs = []
    losses = []

    # Start training
    print('Number of steps', train_steps, flush=True)
    for step, batch in zip(range(1, train_steps + 1), train_ds.as_numpy_iterator()):
        # Generate a PRNG key that will be rolled into the batch
        rng, step_key = jax.random.split(rnd_key)
        # Shard the step PRNG key
        sharded_keys = common_utils.shard_prng_key(step_key)
        opt_repl, state_repl = update_fn(opt=opt_repl,
                                         batch=batch,
                                         state=state_repl,
                                         rnd_key=sharded_keys,
                                         lr=flax_utils.replicate(lr_fn(step)))

        if step % FLAGS.test_every_steps == 0:
            loss, acc = [], []
            r_loss, r_acc = [], []
            for idx, batch in zip(range(test_steps), test_ds.as_numpy_iterator()):
                metrics = p_eval_step(opt_repl.target,
                                      state_repl,
                                      batch)
                r_metrics = None
                for _ in range(FLAGS.pgd_restarts):
                    r_m = p_eval_robust_step(opt=opt_repl,
                                             state=state_repl,
                                             batch=batch)
                    r_m['loss'] = np.array(r_m['loss'])
                    r_m['error_rate'] = np.array(r_m['error_rate'])
                    if r_metrics is None:
                        r_metrics = r_m
                    else:
                        for idx, (loss_new, loss_old) in enumerate(zip(r_m['loss'], r_metrics['loss'])):
                            if loss_new > loss_old:
                                r_metrics['loss'][idx] = loss_new
                                r_metrics['error_rate'][idx] = r_m['error_rate'][idx]

                loss.append(metrics['loss'])
                acc.append(1.0 - metrics['error_rate'])
                r_loss.append(r_metrics['loss'])
                r_acc.append(1.0 - r_metrics['error_rate'])
            print("Test on step", step,
                  "@loss:", np.mean(loss),
                  "@acc:", np.mean(acc),
                  "@r_loss:", np.mean(r_loss),
                  "@r_acc:", np.mean(r_acc),
                  flush=True)

            accs.append(np.mean(acc))
            losses.append(np.mean(loss))
            steps.append(step)


if __name__ == '__main__':
    app.run(main)

