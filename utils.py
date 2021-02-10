#

import jax
import jax.numpy as jnp
import flax
import functools
import io
import numpy as np

from jax.experimental import loops

import constants


def save_ckpt(model_params, path):
    """

    :param model_params:
    :param path:
    :return:
    """
    val, _ = jax.tree_flatten(model_params)
    io_buff = io.BytesIO()

    np.savez(io_buff, *val)

    with open(path, 'wb') as f:
        f.write(io_buff.getvalue())


def load_ckpt(tree, path):
    """

    :param tree:
    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        val = np.load(f, allow_pickle=False)
        val = tuple(val.values())

    return tree.unflatten(val)


def cyclic_lr(
        base_lr,
        max_lr,
        step_size_up,
        step_size_down,
):
    """

    :param base_lr:
    :param max_lr:
    :param step_size_up:
    :param step_size_down:
    :return:
    """
    def step_fn(step):
        if step < step_size_up:
            return base_lr + step * ((max_lr - base_lr) / step_size_up)
        else:
            step = step - step_size_up
            return max_lr - step * ((max_lr - base_lr) / step_size_down)

    return step_fn


def load_and_shard_tf_batch(xs):
    """

    :param xs:
    :return:
    """
    local_device_count = jax.local_device_count()

    def _prepare(x):
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_map(_prepare, xs)


def constant_lr(base_lr, num_steps, warmup=0):
    """

    :param base_lr:
    :param num_steps:
    :param warmup:
    :return:
    """
    def step_fn(step):
        del step
        return base_lr

    del num_steps
    del warmup
    return step_fn


def cross_entory_loss_vec(logits, labels):
    """

    :param logits:
    :param labels:
    :return:
    """
    logp = jax.nn.log_softmax(logits)
    return -jnp.sum(logp * labels, axis=1)


def cross_entropy_loss(logits, labels):
    """

    :param logits:
    :param labels:
    :return:
    """
    return jnp.mean(cross_entory_loss_vec(logits, labels))


def model_loss(params, images, labels, delta, model_fn):
    """

    :param params:
    :param images:
    :param labels:
    :param delta:
    :param model_fn:
    :return:
    """
    logits = model_fn(params, images + delta)
    return cross_entropy_loss(logits=logits, labels=labels)


def attack_pgd(opt, state, rnd_key, batch, eps, alpha, attack_iters, restarts):
    """

    :param opt:
    :param state:
    :param rnd_key:
    :param batch:
    :param eps:
    :param alpha:
    :param attack_iters:
    :param restarts:
    :return:
    """
    # def loss_fn(model, d, input_batch):
    #     with flax.nn.stateful(state) as new_state:
    #         with flax.nn.stochastic(rnd_key):
    #             logits = model(input_batch['image'] + d, train=False)
    #     loss = cross_entropy_loss(logits, input_batch['label'])
    #     return loss

    def loss_vec_fn(model, delta):
        with flax.nn.stateful(state) as new_state:
            with flax.nn.stochastic(rnd_key):
                logits = model(batch['image'] + delta, train=False)
        loss = cross_entory_loss_vec(logits, batch['label'])
        return loss

    X, y = batch['image'], batch['label']
    with loops.Scope() as s:
        s.max_loss = jnp.zeros((y.shape[0]))
        s.max_delta = jnp.zeros_like(X)
        s.delta = jnp.zeros_like(X)
        for _ in s.range(restarts):
            s.delta = init_delta(rnd_key, eps, batch)
            for _ in s.range(attack_iters):
                s.delta = attack_iter(X, alpha, batch, eps, opt, s.delta, rnd_key, state)
            loss = loss_vec_fn(opt.target, s.delta)
            indexes = (s.max_loss < loss).reshape(-1, 1, 1, 1)
            s.max_delta = s.max_delta * (1 - indexes) + s.delta * indexes
            s.max_loss = jnp.maximum(s.max_loss, loss)

        return s.max_delta


@jax.jit
def attack_iter(X, alpha, batch, eps, opt, delta, rnd_key, state):
    """

    :param X:
    :param alpha:
    :param batch:
    :param eps:
    :param opt:
    :param delta:
    :param rnd_key:
    :param state:
    :return:
    """
    def loss_fn(model, d, input_batch):
        with flax.nn.stateful(state) as new_state:
            with flax.nn.stochastic(rnd_key):
                logits = model(input_batch['image'] + d, train=False)
        loss = cross_entropy_loss(logits, input_batch['label'])
        return loss

    with flax.nn.stateful(state) as new_state:
        with flax.nn.stochastic(rnd_key):
            logits = opt.target(batch['image'] + delta, train=False)
    hit = (jnp.argmax(logits, -1) == jnp.argmax(batch['label'], -1)).reshape(-1, 1, 1, 1)
    g_d = jax.grad(loss_fn, 1)(opt.target, delta, batch)
    return update_delta(X, delta, g_d, eps, alpha, X.shape[-1], hit)


def calculate_delta(opt, state, rnd_key, batch, eps, alpha):
    """

    :param opt:
    :param state:
    :param rnd_key:
    :param batch:
    :param eps:
    :param alpha:
    :return:
    """
    def loss_fn(model, delta):
        with flax.nn.stateful(state) as new_state:
            with flax.nn.stochastic(rnd_key):
                logits = model(batch['image'] + delta, train=False)
        loss = cross_entropy_loss(logits, batch['label'])
        return loss

    rnd_key, rnd_input = jax.random.split(rnd_key)
    x = batch['image']
    _, _, _, channels_number = x.shape
    delta = init_delta(rnd_input, eps, batch)
    _, g_d = jax.value_and_grad(loss_fn, 1)(opt.target, delta)

    delta = update_delta(x, delta, g_d, eps, alpha, channels_number, 1)
    return delta


def update_delta(x, delta, g_d, eps, alpha, channels_number, mask):
    """

    :param x:
    :param delta:
    :param g_d:
    :param eps:
    :param alpha:
    :param channels_number:
    :param mask:
    :return:
    """
    deltas = []
    saved_delta = delta
    for channel_idx in range(channels_number):
        new_d = delta[..., channel_idx] + alpha[channel_idx] * jnp.sign(g_d)[..., channel_idx]
        new_d = jax.lax.clamp(-eps[channel_idx], new_d, eps[channel_idx])

        input = x[..., channel_idx]
        lower = constants.lower_limit[channel_idx] - input
        upper = constants.upper_limit[channel_idx] - input
        new_d = jax.lax.clamp(lower, new_d, upper)

        deltas.append(new_d)
    delta = jnp.stack(deltas, axis=-1)
    delta = delta * mask + saved_delta * (1 - mask)
    return delta


def init_delta(rnd_input, eps, batch):
    """

    :param rnd_input:
    :param eps:
    :param batch:
    :return:
    """
    deltas = []
    x = batch['image']
    b, h, w, channels_number = x.shape
    for channel_idx in range(channels_number):
        d = jax.random.uniform(key=rnd_input,
                               shape=(b, h, w),
                               minval=-eps[channel_idx],
                               maxval=eps[channel_idx])

        lower = constants.lower_limit[channel_idx] - x[..., channel_idx]
        upper = constants.upper_limit[channel_idx] - x[..., channel_idx]

        d = jnp.maximum(jnp.minimum(d, upper), lower)

        deltas.append(d)
    delta = jnp.stack(deltas, axis=-1)
    return delta


def update_stateful(opt, state, rnd_key, lr, batch, eps, alpha):
    """

    :param opt:
    :param state:
    :param rnd_key:
    :param lr:
    :param batch:
    :param eps:
    :param alpha:
    :return:
    """
    def loss_fn(model, delta):
        with flax.nn.stateful(state) as new_state:
            with flax.nn.stochastic(rnd_key):
                logits = model(batch['image'] + delta)
        loss = cross_entropy_loss(logits, batch['label'])
        return loss, (new_state, logits)

    _, key = jax.random.split(rnd_key)

    # Calculate distortion
    delta = calculate_delta(opt, state, key, batch, eps, alpha)

    # Optimization step
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_state, logits)), grad = grad_fn(opt.target, delta)
    # loss = jax.experimental.host_callback.id_print(jnp.mean(loss), result=loss)
    grad = jax.lax.pmean(grad, 'batch')
    new_opt = opt.apply_gradient(grad, learning_rate=lr)
    return new_opt, new_state, delta


def robust_eval_step(opt, eps, rnd_key, pgd_alpha, state, batch):
    """

    :param opt:
    :param eps:
    :param rnd_key:
    :param pgd_alpha:
    :param state:
    :param batch:
    :return:
    """
    state = jax.lax.pmean(state, 'batch')
    delta = attack_pgd(opt, state, rnd_key, batch, eps, pgd_alpha, 50, 10)
    with flax.nn.stateful(state, mutable=False):
        logits = opt.target(batch['image'] + delta, train=False)
    metrics = compute_metrics(logits, batch['label'])
    return metrics


def eval_step(model, state, batch):
    """

    :param model:
    :param state:
    :param batch:
    :return:
    """
    state = jax.lax.pmean(state, 'batch')
    with flax.nn.stateful(state, mutable=False):
        logits = model(batch['image'], train=False)
    return compute_metrics(logits, batch['label'])


total_weights = 0


def reset_weights(params, val):
    """

    :param params:
    :param val:
    :return:
    """
    return params
    global total_weights
    if isinstance(params, dict):
        for k in params:
            params[k] = reset_weights(params[k], val)
    else:
        import itertools
        total_weights += list(itertools.accumulate(params.shape, lambda x, y: x * y))[-1]
        print("AFTER ANNIHILATING PARAMS HAVE", total_weights, "WEIGHTS")
        return params * 0.0 + val
    return params


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def create_model(prng_key, batch_size, image_size, channels_number, model_def):
    """

    :param prng_key:
    :param batch_size:
    :param image_size:
    :param channels_number:
    :param model_def:
    :return:
    """
    input_shape = (batch_size, image_size, image_size, channels_number)
    with flax.nn.stateful() as init_state:
        with flax.nn.stochastic(jax.random.PRNGKey(0)):
            _, initial_params = model_def.init_by_shape(
                prng_key, [(input_shape, jnp.float32)])
            initial_params = reset_weights(initial_params, 0.1)
            model = flax.nn.Model(model_def, initial_params)
    return model, init_state


def compute_metrics(logits, labels):
    """

    :param logits:
    :param labels:
    :return:
    """
    loss = cross_entropy_loss(logits, labels)
    error_rate = jnp.mean(jnp.argmax(logits, -1) != jnp.argmax(labels, -1))
    metrics = {
        'loss': loss,
        'error_rate': error_rate,
        'logits': logits,
    }
    return metrics


def eval_step(model, state, batch):
    """

    :param model:
    :param state:
    :param batch:
    :return:
    """
    state = jax.lax.pmean(state, 'batch')
    with flax.nn.stateful(state, mutable=False):
        logits = model(batch['image'], train=False)
    return compute_metrics(logits, batch['label'])
