import functools
import jax
import jax.numpy as jnp
import numpy as np
import random
import sys
import wandb

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

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
flags.DEFINE_integer('eval_batch_size', 1024, '')
flags.DEFINE_integer('number_epochs', 15, '')
flags.DEFINE_string('dataset', 'cifar10', '')
flags.DEFINE_string('model', 'PreResNet18', 'architecture to use')
flags.DEFINE_integer('test_every_steps', 500, '')
flags.DEFINE_integer('save_every', 100, '')
flags.DEFINE_string('checkpoint_name', './model.npz', '')
flags.DEFINE_integer('crop_size', 32, '')
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

cifar10_mean = np.array([0.4914, 0.4822, 0.4465])
cifar10_std = np.array([0.2471, 0.2435, 0.2616])

upper_limit = ((1 - cifar10_mean) / cifar10_std)
lower_limit = ((0 - cifar10_mean) / cifar10_std)


def main(argv):
    del argv

    # On TPUs, use 'mixed_bfloat16' instead
    # policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)
    #
    # print('Compute dtype: %s' % policy.compute_dtype)
    # print('Variable dtype: %s' % policy.variable_dtype)

    # wandb.init(project='smooth_adversarial')

    seed = random.Random().randint(0, sys.maxsize)
    rnd_key = jax.random.PRNGKey(seed)

    train_info = input_pipeline.get_dataset_info(FLAGS.dataset, 'train', examples_per_class=None)
    batches_per_train = int(np.ceil(train_info['num_examples'] / FLAGS.batch_size))

    test_info = input_pipeline.get_dataset_info(FLAGS.dataset, 'test', examples_per_class=None)
    batches_per_test = int(np.ceil(test_info['num_examples'] / FLAGS.eval_batch_size))

    train_steps = batches_per_train * FLAGS.number_epochs
    test_steps = batches_per_test

    model = models[FLAGS.model]

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
    model_creator = model.partial(num_outputs=train_info['num_classes'])
    model, init_state = utils.create_model(rnd_key, FLAGS.batch_size, FLAGS.crop_size, 3, model_creator)
    state_repl = flax_utils.replicate(init_state)

    for block_name, block in model.params.items():
        if block_name == 'Conv_0':
            print(block_name, 'kernel', block['kernel'].shape)
        if block_name == 'Dense_9':
            print(block_name, 'kernel', block['kernel'].shape)
            print(block_name, 'bias', block['bias'].shape)
        if block_name.startswith('PreActBlock_'):
            print(block_name)
            for name, item in block.items():
                if name.startswith('Conv_'):
                    print(name, 'kernel', item['kernel'].shape)
                if name.startswith('BatchNorm_'):
                    # print(name, 'scale', item['scale'].shape)
                    # print(name, 'bias', item['bias'].shape)
                    pass
            print('==================')

    # Create optimizer and replicate it over all GPUs

    # model.params = reset_weights(model.params, 0.01)
    opt = optim.Momentum(beta=0.9, weight_decay=5e-4, learning_rate=FLAGS.lr).create(model)
    opt_repl = flax_utils.replicate(opt)

    # Delete referenes to the objects that are not needed anymore
    del opt
    # del init_state

    # Create function for training
    lr_fn = utils.cyclic_lr(
        base_lr=0.0,
        max_lr=FLAGS.lr,
        step_size_up=train_steps/2,
        step_size_down=train_steps/2)
    alpha = FLAGS.alpha / cifar10_std
    eps = FLAGS.eps / cifar10_std
    pgd_alpha = FLAGS.pgd_alpha / cifar10_std
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

    # Metrics
    steps = []
    accs = []
    losses = []

    # Start training
    mean, std = [], []
    print('Number of steps', train_steps, flush=True)

    for step, batch in zip(range(1, train_steps + 1), train_ds.as_numpy_iterator()):
        # Generate a PRNG key that will be rolled into the batch
        rng, step_key = jax.random.split(rnd_key)
        # Shard the step PRNG key
        sharded_keys = common_utils.shard_prng_key(step_key)

        if step % FLAGS.test_every_steps == 0 or step == 1 or step == train_steps:
            loss, acc = [], []
            r_loss, r_acc = [], []
            for batch_idx, batch in zip(range(test_steps), test_ds.as_numpy_iterator()):
                metrics = p_eval_step(opt_repl.target,
                                      state_repl,
                                      batch)
                # r_metrics = p_eval_robust_step(opt=opt_repl,
                #                                state=state_repl,
                #                                rnd_key=sharded_keys,
                #                                batch=batch)

                loss.append(metrics['loss'])
                acc.append(1.0 - metrics['error_rate'])
                # r_loss.append(r_metrics['loss'])
                # r_acc.append(1.0 - r_metrics['error_rate'])

            print("Test on step", step,
                  "@loss:", np.mean(loss),
                  "@acc:", np.mean(acc),
                  # "@r_loss:", np.mean(r_loss),
                  # "@r_acc:", np.mean(r_acc),
                  flush=True)
            exit(0)

        curr_lr = lr_fn(step - 1)
        sharded_lr = flax_utils.replicate(curr_lr)
        a = np.array(batch['image'])
        mean.append(np.mean(a.reshape((-1, 3)), axis=0))
        std.append(np.std(a.reshape((-1, 3)), axis=0))
        # wandb.log({"lr": curr_lr},
        #           step=step)
        # new_batch = dict()
        # new_batch['image'] = 0.0001 * jnp.ones_like(batch['image'])
        # new_batch['label'] = batch['label']
        # metrics = p_eval_step(opt_repl.target,
        #                       state_repl,
        #                       new_batch)
        # print(metrics['logits'])
        opt_repl, state_repl = update_fn(opt=opt_repl,
                                         batch=batch,
                                         state=state_repl,
                                         rnd_key=sharded_keys,
                                         lr=sharded_lr)

        # if step % FLAGS.test_every_steps == 0 or step == 1 or step == train_steps:
        #     loss, acc = [], []
        #     r_loss, r_acc = [], []
        #     for batch_idx, batch in zip(range(test_steps), test_ds.as_numpy_iterator()):
        #         metrics = p_eval_step(opt_repl.target,
        #                               state_repl,
        #                               batch)
        #         r_metrics = p_eval_robust_step(opt=opt_repl,
        #                                        state=state_repl,
        #                                        rnd_key=sharded_keys,
        #                                        batch=batch)
        #
        #         loss.append(metrics['loss'])
        #         acc.append(1.0 - metrics['error_rate'])
        #         r_loss.append(r_metrics['loss'])
        #         r_acc.append(1.0 - r_metrics['error_rate'])
        #
        #     # wandb.log({"loss": np.mean(loss),
        #     #            "acc": np.mean(acc),
        #     #            "robust_loss": np.mean(r_loss),
        #     #            "robust_acc": np.mean(r_acc)},
        #     #           step=step)
        #     print("Test on step", step,
        #           "@loss:", np.mean(loss),
        #           "@acc:", np.mean(acc),
        #           "@r_loss:", np.mean(r_loss),
        #           "@r_acc:", np.mean(r_acc),
        #           flush=True)

            # accs.append(np.mean(acc))
            # losses.append(np.mean(loss))
            # steps.append(step)
    # print('final', np.mean(mean, axis=0), np.mean(std, axis=0))


if __name__ == '__main__':
    app.run(main)
