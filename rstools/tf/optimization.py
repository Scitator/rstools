import tensorflow as tf


def update_varlist(loss, optimizer, var_list, grad_clip=1.0, global_step=None, global_clip=False):
    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    if not global_clip:
        grads_and_vars = [(tf.clip_by_norm(grad, grad_clip), var) for grad, var in grads_and_vars]
    else:
        gradients, variables = zip(*grads_and_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=grad_clip)
        grads_and_vars = list(zip(clipped_gradients, variables))
    update_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    return update_step


def build_scope_optimization(
        var_list, optimization_params=None, loss=None, optimizer=None, lr=None, global_step=None,
        **kwargs):
    assert loss is not None
    optimization_params = optimization_params or {}

    initial_lr = optimization_params.get("initial_lr", 1e-4)
    decay_steps = int(optimization_params.get("decay_steps", 100000))
    lr_decay = optimization_params.get("lr_decay", 0.999)
    grad_clip = optimization_params.get("grad_clip", 10.0)

    # @TODO: need to test other lrs
    lr = lr or tf.train.exponential_decay(
        initial_lr,
        global_step,
        decay_steps,
        lr_decay,
        staircase=True)

    optimizer = optimizer or tf.train.AdamOptimizer(lr)

    train_op = update_varlist(
        loss, optimizer,
        var_list=var_list,
        grad_clip=grad_clip,
        global_step=global_step,
        **kwargs)
    return optimizer, train_op


def build_model_optimization(
        model, optimization_params=None,
        loss=None, optimizer=None, lr=None,
        **kwargs):
    model.loss = model.loss if model.loss is not None else loss
    model.optimizer, model.train_op = build_scope_optimization(
        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope=model.scope),
        optimization_params=optimization_params,
        loss=model.loss, optimizer=optimizer, lr=lr,
        global_step=model.global_step,
        **kwargs)
