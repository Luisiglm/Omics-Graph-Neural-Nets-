import tensorflow as tf


def c_index_loss(surv, f):
    """ Returns the minus C-index for survival.
    Args:
       surv: a tensor with two columns corresponding to the times to last followup and events (1 if an event
              has occured at followup). tensor of shape [batch size x 2]
       f: predices risks. Tensor of shape [batch size]
    """
    t = tf.gather(surv, [0], axis=1)
    d = tf.cast(1, dtype="float32") - tf.transpose(tf.gather(surv, [1], axis=1))
    # first check for ti>tj and fi<tj we will call this case 1 (c1)
    greater_times = get_greater_times(t)
    greater_risks = get_greater_times(-f)
    c_loss_c1 = tf.math.multiply(greater_times, greater_risks)
    # use only the rows that aren't censored.
    c_loss_c1 = tf.transpose(c_loss_c1)
    c_loss_c1 = tf.math.reduce_sum(tf.matmul(d, c_loss_c1))
    # check for tied times.
    equal_times = get_equal_times(t)
    equal_risks = get_equal_times(f)
    lower_risks = get_greater_times(f)
    # check for ti==tj and fi>fj and di ==1 and dj == 0, we will call this case 2 (c2)
    c_loss_c2 = tf.math.multiply(equal_times, greater_risks)
    c_loss_c2 = tf.transpose(c_loss_c2)
    c_loss_c2 = tf.math.reduce_sum(tf.matmul(d, c_loss_c2))
    # check for ti==tj and fi<fj and di ==1 and dj == 0, we will call this case 3 (c3)
    c_loss_c3 = tf.math.multiply(equal_times, lower_risks)
    c_loss_c3 = tf.transpose(c_loss_c3)
    c_loss_c3 = tf.math.reduce_sum(tf.matmul(tf.math.multiply(d, tf.cast(0.5, 'float32')), c_loss_c3))
    #  check for ti==tj and fi==fj and di ==1 and dj == 1, we will call this case 4 (c4)
    c_loss_c4 = tf.math.multiply(equal_times, equal_risks)
    c_loss_c4 = tf.transpose(c_loss_c4)
    c_loss_c4 = tf.math.reduce_sum(tf.matmul(tf.math.multiply(d, tf.cast(0.5, 'float32')), c_loss_c4))
    #  check for ti>tj and fi==fj and di ==1 and dj == 0, we will call this case 5 (c5)
    c_loss_c5 = tf.math.multiply(greater_times, equal_risks)
    c_loss_c5 = tf.transpose(c_loss_c5)
    c_loss_c5 = tf.math.reduce_sum(tf.matmul(tf.math.multiply(d, tf.cast(0.5, 'float32')), c_loss_c5))
    c_loss = -c_loss_c1 - c_loss_c2 - c_loss_c3 - c_loss_c4 - c_loss_c5
    # divide by all permissible pairs.
    no = tf.math.reduce_sum(tf.matmul(d, tf.transpose(greater_times))) + tf.math.reduce_sum(
        tf.matmul(tf.math.multiply(d, tf.cast(0.5, 'float32')), tf.transpose(equal_times)))
    return (c_loss / no)


def get_equal_times(t):
    """ Returns the minus C-index for survival.
    Args:
       t: times to last followup or risks. Tensor fo shape [batch size]
    Example:
       t = [1, 2, 2]

       t_order = [[0, 0, 0,],
             [0, 0, 1,],
             [0, 1, 0]]
    """
    batch_size = tf.shape(t)[0]
    rep_t1 = tf.tile(t, (1, batch_size))
    rep_t2 = tf.transpose(rep_t1)
    t_order = tf.equal(rep_t1, rep_t2)
    t_order = tf.cast(t_order, dtype="float32")
    # delete the diagonal element.
    t_order = tf.linalg.set_diag(t_order, tf.zeros(batch_size, dtype='float32'))
    return (t_order)


def get_greater_times(t):
    """ Returns the minus C-index for survival.
    Args:
       t: times to last followup or risks. Tensor fo shape [batch size]
    Example:
       t = [1, 3, 2]

       t_order = [[0, 1, 1,],
             [0, 0, 0,],
             [0, 1, 0]]
    """
    batch_size = tf.shape(t)[0]
    rep_t1 = tf.tile(t, (1, batch_size))
    rep_t2 = tf.transpose(rep_t1)
    t_order = tf.greater(rep_t1, rep_t2)
    t_order = tf.cast(t_order, dtype="float32")
    return (t_order)




