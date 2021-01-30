import jax
import jax.numpy as jnp
import flax
import functools
import io
import numpy as np

from jax.experimental import loops


cifar10_mean = np.array([0.4914, 0.4822, 0.4465])
cifar10_std = np.array([0.2471, 0.2435, 0.2616])

upper_limit = ((1 - cifar10_mean) / cifar10_std)
lower_limit = ((0 - cifar10_mean) / cifar10_std)


def save_ckpt(model_params, path):
    val, _ = jax.tree_flatten(model_params)
    io_buff = io.BytesIO()

    np.savez(io_buff, *val)

    with open(path, 'wb') as f:
        f.write(io_buff.getvalue())


def load_ckpt(tree, path):
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
    def step_fn(step):
        if step < step_size_up:
            return base_lr + step * ((max_lr - base_lr) / step_size_up)
        else:
            step = step - step_size_up
            return max_lr - step * ((max_lr - base_lr) / step_size_down)
        # return max_lr

    return step_fn


def load_and_shard_tf_batch(xs):
    local_device_count = jax.local_device_count()

    def _prepare(x):
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_map(_prepare, xs)


def constant_lr(base_lr, num_steps, warmup=0):
    def step_fn(step):
        del step
        return base_lr

    del num_steps
    del warmup
    return step_fn


# def cross_entory_loss_vec(logits, labels):
#     logp = jax.nn.log_softmax(logits)
#     return jnp.sum(logp * labels, axis=1)

def cross_entory_loss_vec(logits, labels):
    logp = jax.nn.log_softmax(logits)
    return -jnp.sum(logp * labels, axis=1)


def cross_entropy_loss(logits, labels):
    # logp = jax.nn.log_softmax(logits)
    # return -jnp.mean(jnp.sum(logp * labels, axis=1))
    return jnp.mean(cross_entory_loss_vec(logits, labels))


def model_loss(params, images, labels, delta, model_fn):
    logits = model_fn(params, images + delta)
    return cross_entropy_loss(logits=logits, labels=labels)


def evaluate(opt, images, labels, model_fn):
    logits = model_fn(opt.target, images)
    loss = cross_entropy_loss(logits=logits, labels=labels)
    acc = jnp.mean(jnp.argmax(logits, axis=1) == jnp.argmax(labels, axis=1))
    return loss, acc


def attack_pgd(opt, state, rnd_key, batch, eps, alpha,  attack_iters, restarts):
    def loss_fn(model, d, input_batch):
        with flax.nn.stateful(state) as new_state:
            with flax.nn.stochastic(rnd_key):
                logits = model(input_batch['image'] + d)
        loss = cross_entropy_loss(logits, input_batch['label'])
        return loss

    def loss_vec_fn(model, delta):
        with flax.nn.stateful(state) as new_state:
            with flax.nn.stochastic(rnd_key):
                logits = model(batch['image'] + delta)
        loss = cross_entory_loss_vec(logits, batch['label'])
        return loss

    X, y = batch['image'], batch['label']
    # max_loss = jnp.zeros((y.shape[0]))
    # max_delta = jnp.zeros_like(X)
    # eps = eps * 0.0 + 3.0
    restarts = 1
    with loops.Scope() as s:
        s.max_loss = jnp.zeros((y.shape[0]))
        s.max_delta = jnp.zeros_like(X)
        s.delta = jnp.zeros_like(X)
        for _ in s.range(restarts):
            s.delta = init_delta(rnd_key, eps, batch)
            for _ in s.range(attack_iters):
                with flax.nn.stateful(state) as new_state:
                    with flax.nn.stochastic(rnd_key):
                        logits = opt.target(batch['image'] + s.delta, train=False)
                hit = (jnp.argmax(logits, -1) == jnp.argmax(batch['label'], -1)).reshape(-1, 1, 1, 1)
                # hit = jax.experimental.host_callback.id_print(jnp.sum(hit), result=hit, a='HIT')
                for _ in s.cond_range(jnp.sum(hit) > 0):
                    g_d = jax.grad(loss_fn, 1)(opt.target, s.delta, batch)
                    # g_d = jax.experimental.host_callback.id_print(jnp.mean(l), result=g_d, a='loss')
                    s.delta = update_delta(X, s.delta, g_d, eps, alpha, X.shape[-1], hit)

            loss = loss_vec_fn(opt.target, s.delta)

            indexes = (s.max_loss < loss).reshape(-1, 1, 1, 1) #jnp.argmax(joint_loss, axis=0).reshape(-1, 1, 1, 1)
            # indexes = jax.experimental.host_callback.id_print(jnp.sum(indexes), result=indexes, a='WILL REPLACE')
            s.max_delta = s.max_delta * (1 - indexes) + s.delta * indexes
            s.max_loss = jnp.maximum(s.max_loss, loss)
            with flax.nn.stateful(state) as new_state:
                with flax.nn.stochastic(rnd_key):
                    logits = opt.target(batch['image'] + s.max_delta, train=False)
            hit = (jnp.argmax(logits, -1) == jnp.argmax(batch['label'], -1)).reshape(-1, 1, 1, 1)
            # s.max_delta = jax.experimental.host_callback.id_print(jnp.sum(hit), result=s.max_delta, a='FINAL HIT')

            # s.max_loss = jax.experimental.host_callback.id_print(jnp.mean(max_loss), result=max_loss, a='max_loss')

            # max_delta = delta
        # max_delta = jax.experimental.host_callback.id_print(jnp.max(max_delta), result=max_delta, a='MAX max_delta')
        # max_delta = jax.experimental.host_callback.id_print(jnp.min(max_delta), result=max_delta, a='MIN max_delta')

        return s.max_delta

    # return max_delta


def calculate_delta(opt, state, rnd_key, batch, eps, alpha):
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
    with flax.nn.stateful(state) as new_state:
        with flax.nn.stochastic(rnd_key):
            logits = opt.target(batch['image'] + delta)

    delta = update_delta(x, delta, g_d, eps, alpha, channels_number, 1)
    return delta


def update_delta(x, delta, g_d, eps, alpha, channels_number, mask):
    deltas = []
    saved_delta=delta
    for channel_idx in range(channels_number):
        new_d = delta[..., channel_idx] + alpha[channel_idx] * jnp.sign(g_d)[..., channel_idx]
        new_d = jax.lax.clamp(-eps[channel_idx], new_d, eps[channel_idx])

        input = x[..., channel_idx]
        lower = lower_limit[channel_idx] - input
        upper = upper_limit[channel_idx] - input
        new_d = jax.lax.clamp(lower, new_d, upper)
        # new_d = jnp.maximum(jnp.minimum(new_d, upper), lower)

        deltas.append(new_d)
    delta = jnp.stack(deltas, axis=-1)
    delta = delta * mask + saved_delta * (1 - mask)
    return delta


def init_delta(rnd_input, eps, batch):
    deltas = []
    x = batch['image']
    b, h, w, channels_number = x.shape
    for channel_idx in range(channels_number):
        d = jax.random.uniform(key=rnd_input,
                               shape=(b, h, w),
                               minval=-eps[channel_idx],
                               maxval=eps[channel_idx])

        lower = lower_limit[channel_idx] - x[..., channel_idx]
        upper = upper_limit[channel_idx] - x[..., channel_idx]

        d = jnp.maximum(jnp.minimum(d, upper), lower)

        deltas.append(d)
    delta = jnp.stack(deltas, axis=-1)
    return delta


def update_stateful(opt, state, rnd_key, lr, batch, eps, alpha):
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


# def compute_metrics(logits, labels):
#     loss = cross_entropy_loss(logits, labels)
#     error_rate = jnp.mean(jnp.argmax(logits, -1) != jnp.argmax(labels, -1))
#     metrics = {
#         'loss': loss,
#         'error_rate': error_rate,
#         'logits': logits,
#     }
#     # metrics = jax.lax.pmean(metrics, 'batch')
#     return metrics


def robust_eval_step(opt, eps, rnd_key, pgd_alpha, state, batch):
    state = jax.lax.pmean(state, 'batch')
    # adv_x = pgd(opt.target, batch['image'], eps, pgd_alpha, 50, np.inf)
    delta = attack_pgd(opt, state, rnd_key, batch, eps, pgd_alpha, 50, 10)
    # delta = jax.experimental.host_callback.id_print(jnp.std(delta), result=delta)
    with flax.nn.stateful(state, mutable=False):
        logits = opt.target(batch['image'] + delta, train=False)
    metrics = compute_metrics(logits, batch['label'])
    return metrics


def eval_step(model, state, batch):
    state = jax.lax.pmean(state, 'batch')
    with flax.nn.stateful(state, mutable=False):
        logits = model(batch['image'], train=False)
    return compute_metrics(logits, batch['label'])

total_weights = 0
def reset_weights(params, val):
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
    input_shape = (batch_size, image_size, image_size, channels_number)
    with flax.nn.stateful() as init_state:
        with flax.nn.stochastic(jax.random.PRNGKey(0)):
            _, initial_params = model_def.init_by_shape(
                prng_key, [(input_shape, jnp.float32)])
            initial_params = reset_weights(initial_params, 0.1)
            model = flax.nn.Model(model_def, initial_params)
    return model, init_state


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    error_rate = jnp.mean(jnp.argmax(logits, -1) != jnp.argmax(labels, -1))
    metrics = {
        'loss': loss,
        'error_rate': error_rate,
        'logits': logits,
    }
    # metrics = jax.lax.pmean(metrics, 'batch')
    return metrics


def eval_step(model, state, batch):
    state = jax.lax.pmean(state, 'batch')
    with flax.nn.stateful(state, mutable=False):
        logits = model(batch['image'], train=False)
    return compute_metrics(logits, batch['label'])
