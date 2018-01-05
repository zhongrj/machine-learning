import tensorflow as tf


def init_sess(saver, checkpoint_dir):
    sess = tf.InteractiveSession()
    try:
        # if not os.path.exists(dir):
        #     raise BaseException()
        saver.restore(sess, checkpoint_dir)
        print('Loading session from ', checkpoint_dir)
    except:
        print('Loading session Exception, initializing all parameters...')
        sess.run(tf.global_variables_initializer())
    return sess
