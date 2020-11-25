import jax
import jax.numpy as jnp
import flax
import functools
import io
import numpy as np


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


def cyclic_lr(base_lr, num_steps):
  def step_fn(step):
    N = num_steps//2
    if step < N:
      lr = base_lr * step/N
    else:
      lr = base_lr * (1.0 - (step%N)/N)
    return lr

  return step_fn


def load_and_shard_tf_batch(xs):
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    # x = x.numpy()  # pylint: disable=protected-access
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def constant_lr(base_lr, num_steps, warmup=0):
    def step_fn(step):
        return base_lr

    return step_fn


def cross_entropy_loss(logits, labels):
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(logp * labels, axis=1))


def model_loss(params, images, labels, delta, model_fn):
    logits = model_fn(params, images + delta)
    return cross_entropy_loss(logits=logits, labels=labels)


# # Update step, replicated over all GPUs
# @partial(jax.pmap, axis_name='batch')
def update(opt, rnd_key, lr, batch, loss_fn, eps, alpha):
    rnd_key, rnd_input = jax.random.split(rnd_key)
    delta = jax.random.uniform(key=rnd_input,
                               shape=batch['image'].shape,
                               minval=-eps,
                               maxval=eps)
    l, g_d = jax.value_and_grad(loss_fn, 3)(opt.target,
                                            batch['image'],
                                            batch['label'],
                                            delta)
    delta = jax.lax.clamp(-eps, delta + alpha * jnp.sign(g_d), eps)
    l, g_t = jax.value_and_grad(loss_fn, 0)(opt.target,
                                            batch['image'],
                                            batch['label'],
                                            delta)

    g = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), g_t)
    opt = opt.apply_gradient(g, learning_rate=lr)
    return opt, rnd_key


def evaluate(opt, images, labels, model_fn):
    logits = model_fn(opt.target, images)
    loss = cross_entropy_loss(logits=logits, labels=labels)
    acc = jnp.mean(jnp.argmax(logits, axis=1) == jnp.argmax(labels, axis=1))
    return loss, acc


def calculate_delta(opt, state, rnd_key, batch, eps, alpha):
    def loss_fn(model, delta):
        with flax.nn.stateful(state) as new_state:
            with flax.nn.stochastic(rnd_key):
                logits = model(batch['image'])
        loss = cross_entropy_loss(logits, batch['label'])
        return loss, (new_state, logits)

    rnd_key, rnd_input = jax.random.split(rnd_key)
    delta = jax.random.uniform(key=rnd_input,
                               shape=batch['image'].shape,
                               minval=-eps,
                               maxval=eps)
    grad_fn = jax.value_and_grad(loss_fn, 1, has_aux=True)
    _, g_d = grad_fn(opt.target, delta)
    delta = jax.lax.clamp(-eps, delta + alpha * jnp.sign(g_d), eps)
    return delta


def update_stateful(opt, state, rnd_key, lr, batch, eps, alpha):
    def loss_fn(model, delta):
        with flax.nn.stateful(state) as new_state:
            with flax.nn.stochastic(rnd_key):
                logits = model(batch['image'])
        loss = cross_entropy_loss(logits, batch['label'])
        return loss, (new_state, logits)

    # Calculate distortion
    delta = calculate_delta(opt, state, rnd_key, batch, eps, alpha)

    # Optimization step
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (new_state, logits)), grad = grad_fn(opt.target, delta)
    grad = jax.lax.pmean(grad, 'batch')
    new_opt = opt.apply_gradient(grad, learning_rate=lr)

    return new_opt, new_state


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    error_rate = jnp.mean(jnp.argmax(logits, -1) != jnp.argmax(labels, -1))
    metrics = {
        'loss': loss,
        'error_rate': error_rate,
    }
    metrics = jax.lax.pmean(metrics, 'batch')
    return metrics


def robust_eval_step(opt, rnd_key, eps, alpha, state, batch):
    delta = calculate_delta(opt, state, rnd_key, batch, eps, alpha)
    state = jax.lax.pmean(state, 'batch')
    with flax.nn.stateful(state, mutable=False):
        logits = opt.target(batch['image'] + delta, train=False)
    return compute_metrics(logits, batch['label'])


def eval_step(model, state, batch):
    state = jax.lax.pmean(state, 'batch')
    with flax.nn.stateful(state, mutable=False):
        logits = model(batch['image'], train=False)
    return compute_metrics(logits, batch['label'])


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def create_model(prng_key, batch_size, image_size, channels_number, model_def):
    input_shape = (batch_size, image_size, image_size, channels_number)
    with flax.nn.stateful() as init_state:
        with flax.nn.stochastic(jax.random.PRNGKey(0)):
            _, initial_params = model_def.init_by_shape(
                prng_key, [(input_shape, jnp.float32)])
            model = flax.nn.Model(model_def, initial_params)
    return model, init_state


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    error_rate = jnp.mean(jnp.argmax(logits, -1) != jnp.argmax(labels, -1))
    metrics = {
        'loss': loss,
        'error_rate': error_rate,
    }
    metrics = jax.lax.pmean(metrics, 'batch')
    return metrics


def eval_step(model, state, batch):
    state = jax.lax.pmean(state, 'batch')
    with flax.nn.stateful(state, mutable=False):
        logits = model(batch['image'], train=False)
    return compute_metrics(logits, batch['label'])


