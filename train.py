# Train FGSM on CIFAR10
# Original PyTorch code: https://github.com/locuslab/fast_adversarial/blob/54f728755e71857b632882ba6d6cef22f56e2172/CIFAR10/train_fgsm.py

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

import constants
import input_pipeline
import models
import utils

from absl import app
from absl import flags

from tqdm import tqdm

# Define models that could be used in this script
models = {
    'resnet50': models.ResNet50,
    'resnet101': models.ResNet101,
    'resnet152': models.ResNet152,
    'resnet50x2': models.ResNet50x2,
    'resnet101x2': models.ResNet101x2,
    'resnet152x2': models.ResNet152x2,
    'resnext50_32x4d': models.ResNext50_32x4d,
    'resnext101_32x8d': models.ResNext101_32x8d,
    'resnext152_32x4d': models.ResNext152_32x4d,
    'preresnet18': models.PreActResNet18,
    'preresnet34': models.PreActResNet34,
    'preresnet50': models.PreActResNet50,
    'preresnet101': models.PreActResNet101,
    'preresnet152': models.PreActResNet152,
}

# Define script parameters
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_integer('eval_batch_size', 512, '')
flags.DEFINE_integer('number_epochs', 15, '')
# flags.DEFINE_string('dataset', 'cifar10', '')
flags.DEFINE_string('model', 'PreResNet18', 'The architecture to use')  # TODO: add choice
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
flags.DEFINE_bool('test_each_epoch', True, 'Test each test_every_steps or only at the end of training.')


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
    train_info = input_pipeline.get_dataset_info('cifar10', 'train', examples_per_class=None)
    batches_per_train = int(np.ceil(train_info['num_examples'] / FLAGS.batch_size))

    test_info = input_pipeline.get_dataset_info('cifar10', 'test', examples_per_class=None)
    batches_per_test = int(np.ceil(test_info['num_examples'] / FLAGS.eval_batch_size))

    train_steps = batches_per_train * FLAGS.number_epochs
    test_steps = batches_per_test

    # Create dataset
    train_ds = input_pipeline.get_data(dataset='cifar10',
                                       mode='train',
                                       repeats=None,
                                       batch_size=FLAGS.batch_size,
                                       data_mean=constants.cifar10_mean,
                                       data_std=constants.cifar10_mean,
                                       crop_size=FLAGS.crop_size,
                                       examples_per_class=None,
                                       examples_per_class_seed=None,
                                       num_devices=jax.device_count(),
                                       tfds_manual_dir=None).as_numpy_iterator()

    test_ds = input_pipeline.get_data(dataset='cifar10',
                                      mode='test',
                                      repeats=None,
                                      batch_size=FLAGS.eval_batch_size,
                                      data_mean=constants.cifar10_mean,
                                      data_std=constants.cifar10_mean,
                                      crop_size=FLAGS.crop_size,
                                      examples_per_class=None,
                                      examples_per_class_seed=None,
                                      num_devices=jax.device_count(),
                                      tfds_manual_dir=None).as_numpy_iterator()

    # Build ResNet architecture
    model_name = FLAGS.model.lower()
    model = models[model_name]
    if model_name == "pyramid":
        model_creator = model.partial(num_outputs=train_info['num_classes'],
                                      pyramid_alpha=FLAGS.pyramid_alpha,
                                      pyramid_depth=FLAGS.pyramid_depth)
    if model_name.startswith("resnet"):
        model_creator = model.partial(num_outputs=train_info['num_classes'])
    if model_name.startswith("wideresnet"):
        model_creator = model.partial(num_outputs=train_info['num_classes'])
    if model_name.startswith("preresnet"):
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
    alpha = FLAGS.alpha / constants.cifar10_std
    eps = FLAGS.eps / constants.cifar10_std
    pgd_alpha = FLAGS.pgd_alpha / constants.cifar10_std

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

    # After which train steps we want evaluate our model
    train_epochs = get_test_epochs(FLAGS.test_each_epoch, FLAGS.test_every_steps, train_steps)

    # The main loop
    for step, batch in tqdm(zip(range(1, train_steps + 1), train_ds),
                            total=train_steps, desc="Train", leave=True):
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
        metrics = evaluate_on_train(batch, delta, opt_repl, p_eval_step, state_repl)
        train_acc.append(1.0 - metrics['error_rate'])
        train_loss.append(metrics['loss'])

        # Evaluate model and submit results to W&B
        if step in train_epochs:
            acc, loss, r_acc, r_loss = evaluate_on_test(opt_repl, p_eval_robust_step,
                                                        p_eval_step, sharded_keys,
                                                        state_repl, test_ds, test_steps)

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

    # Commit final accs
    acc, _, r_acc, _ = evaluate_on_test(opt_repl, p_eval_robust_step,
                                        p_eval_step, sharded_keys,
                                        state_repl, test_ds, test_steps)
    wandb.log({
        "final_test_acc": np.mean(acc),
        "final_robust_acc": np.mean(r_acc)}
    )

    # Model saving
    opt = flax_utils.unreplicate(opt_repl)
    utils.save_ckpt(opt.target.params)


def get_test_epochs(test_each_epoch, test_every_steps, train_steps):
    if test_each_epoch:
        return set(range(1, train_steps + 1, test_every_steps))
    else:
        return {train_steps}


def evaluate_on_train(batch, delta, opt_repl, p_eval_step, state_repl):
    """

    :param batch:
    :param delta:
    :param opt_repl:
    :param p_eval_step:
    :param state_repl:
    :return:
    """
    input_batch = dict()
    input_batch['image'] = batch['image'] + delta
    input_batch['label'] = batch['label']
    metrics = p_eval_step(opt_repl.target,
                          state_repl,
                          input_batch)
    return metrics


def evaluate_on_test(opt_repl, p_eval_robust_step, p_eval_step, sharded_keys, state_repl, test_ds, test_steps):
    """

    :param opt_repl:
    :param p_eval_robust_step:
    :param p_eval_step:
    :param sharded_keys:
    :param state_repl:
    :param test_ds:
    :param test_steps:
    :return:
    """
    loss, acc = [], []
    r_loss, r_acc = [], []

    for batch_idx, test_batch in tqdm(zip(range(test_steps), test_ds), total=test_steps,
                                      desc="Evaluate", leave=True):
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
    return acc, loss, r_acc, r_loss


if __name__ == '__main__':
    app.run(main)
