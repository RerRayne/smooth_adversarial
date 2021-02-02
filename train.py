import functools
import jax
import numpy as np
import os
import random
import sys
import wandb

import flax.optim as optim
import flax.jax_utils as flax_utils
from flax.training import common_utils

import input_pipeline
import models
import utils

from absl import app
from absl import flags

from tqdm import tqdm

# Define models that could be used in this script
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

# Define script parameters
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_integer('eval_batch_size', 512, '')
flags.DEFINE_integer('number_epochs', 15, '')
flags.DEFINE_string('dataset', 'cifar10', '')
flags.DEFINE_string('model', 'PreResNet18', 'The architecture to use') # TODO: add choice
flags.DEFINE_integer('test_every_steps', 500, '')
flags.DEFINE_integer('save_every', 100, '')
flags.DEFINE_string('checkpoint_name', './model.npz', '')
flags.DEFINE_integer('crop_size', 32, '')
flags.DEFINE_float('lr', 0.2, '')
flags.DEFINE_float('eps', 8.0 / 255.0, '')
flags.DEFINE_float('alpha', 10.0 / 255.0, '')
flags.DEFINE_float('pgd_alpha', 2.0 / 255.0, '')
flags.DEFINE_integer('pgd_restarts', 10, '')
flags.DEFINE_string('gpu', '-1',
                    "What GPU to use. For example --gpu=-1, "
                    "don't use GPU. --gpu=0 -- use GPU 0."
                    "--gpu=0,1 -- use GPU 0 and 1")
flags.DEFINE_string('wandb_proj_name', 'smooth_adversarial', '')


def main(argv):
    del argv

    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    # Init W&B
    wandb.init(project=FLAGS.wandb_proj_name)

    # Initialize random generator
    seed = random.Random().randint(0, sys.maxsize)
    rnd_key = jax.random.PRNGKey(seed)

    # Create dataset readers
    train_info = input_pipeline.get_dataset_info(FLAGS.dataset, 'train', examples_per_class=None)
    batches_per_train = int(np.ceil(train_info['num_examples'] / FLAGS.batch_size))

    test_info = input_pipeline.get_dataset_info(FLAGS.dataset, 'test', examples_per_class=None)
    batches_per_test = int(np.ceil(test_info['num_examples'] / FLAGS.eval_batch_size))

    train_steps = batches_per_train * FLAGS.number_epochs
    test_steps = batches_per_test

    # Create dataset
    train_ds = input_pipeline.get_data(dataset=FLAGS.dataset,
                                       mode='train',
                                       repeats=None,
                                       batch_size=FLAGS.batch_size,
                                       crop_size=FLAGS.crop_size,
                                       mixup_alpha=None,
                                       examples_per_class=None,
                                       examples_per_class_seed=None,
                                       num_devices=jax.device_count(),
                                       tfds_manual_dir=None)

    test_ds = input_pipeline.get_data(dataset=FLAGS.dataset,
                                      mode='test',
                                      repeats=None,
                                      batch_size=FLAGS.eval_batch_size,
                                      crop_size=FLAGS.crop_size,
                                      mixup_alpha=None,
                                      examples_per_class=None,
                                      examples_per_class_seed=None,
                                      num_devices=jax.device_count(),
                                      tfds_manual_dir=None)

    # Build ResNet architecture
    # TODO: add the rest off
    model = models[FLAGS.model]
    model_creator = model.partial(num_outputs=train_info['num_classes'])
    model, init_state = utils.create_model(rnd_key, FLAGS.batch_size, FLAGS.crop_size, 3, model_creator)
    state_repl = flax_utils.replicate(init_state)

    # Create optimizer and replicate it over all GPUs
    opt = optim.Momentum(beta=0.9, weight_decay=5e-4, learning_rate=FLAGS.lr).create(model)
    opt_repl = flax_utils.replicate(opt)

    # Delete references to the objects that are not needed anymore
    del opt
    del init_state

    # Create function for training
    lr_fn = utils.cyclic_lr(
        base_lr=0.0,
        max_lr=FLAGS.lr,
        step_size_up=train_steps / 2,
        step_size_down=train_steps / 2)

    # Normalize attack parameters
    alpha = FLAGS.alpha / utils.cifar10_std
    eps = FLAGS.eps / utils.cifar10_std
    pgd_alpha = FLAGS.pgd_alpha /utils.cifar10_std

    # Compile train step and evaluate functions with XLA
    # to execute them in parallel on XLA devices.
    update_fn = jax.pmap(functools.partial(utils.update_stateful,
                                           eps=eps,
                                           alpha=alpha),
                         axis_name='batch')
    p_eval_step = jax.pmap(utils.eval_step,
                           axis_name='batch')
    p_eval_robust_step = jax.pmap(functools.partial(utils.robust_eval_step,
                                                    eps=eps,
                                                    pgd_alpha=pgd_alpha),
                                  axis_name='batch')
    # Initialize metrics arrays
    train_acc, train_loss = [], []

    # Train loop
    for step, batch in tqdm(zip(range(1, train_steps + 1),
                                train_ds.as_numpy_iterator()),
                            total=train_steps):
        # Generate a PRNG key that will be rolled into the batch
        rng, step_key = jax.random.split(rnd_key)

        # Shard the step PRNG key over XLA devices
        sharded_keys = common_utils.shard_prng_key(step_key)

        # Shard learning rate over XLA devices
        curr_lr = lr_fn(step - 1)
        sharded_lr = flax_utils.replicate(curr_lr)
        # Log to W&B lr
        wandb.log({"lr": curr_lr},
                  step=step)

        # Train step
        opt_repl, state_repl, delta = update_fn(opt=opt_repl,
                                                batch=batch,
                                                state=state_repl,
                                                rnd_key=sharded_keys,
                                                lr=sharded_lr)

        # Calculate accuracy over a train batch
        input_batch = dict()
        input_batch['image'] = batch['image'] + delta
        input_batch['label'] = batch['label']

        metrics = p_eval_step(opt_repl.target,
                              state_repl,
                              input_batch)
        train_acc.append(1.0 - metrics['error_rate'])
        train_loss.append(metrics['loss'])

        # Evaluate model and submit results to W&B
        if step % FLAGS.test_every_steps == 0 or step == 1 \
                or step == train_steps or step % batches_per_train == 0:
            loss, acc = [], []
            r_loss, r_acc = [], []
            for batch_idx, test_batch in zip(range(test_steps), test_ds.as_numpy_iterator()):
                metrics = p_eval_step(opt_repl.target,
                                      state_repl,
                                      test_batch)
                r_metrics = p_eval_robust_step(opt=opt_repl,
                                               state=state_repl,
                                               rnd_key=sharded_keys,
                                               batch=test_batch)

                loss.append(metrics['loss'])
                acc.append(1.0 - metrics['error_rate'])
                r_loss.append(r_metrics['loss'])
                r_acc.append(1.0 - r_metrics['error_rate'])

                # Log to W&B stats
                wandb.log({
                    "test_loss": np.mean(loss),
                    "test_acc": np.mean(acc),
                    "robus_loss": np.mean(r_loss),
                    "robust_acc": np.mean(r_acc),
                    "train_loss:": np.mean(train_loss),
                    "train_acc:": np.mean(train_acc)})

                # Re-initialize train stats for next epoch
                train_acc = []
                train_loss = []
    # TODO: add commit of final information
    # TODO: add model saving


if __name__ == '__main__':
    app.run(main)
