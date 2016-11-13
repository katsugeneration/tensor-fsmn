import tensorflow as tf
import numpy as np
from fsmn import *
import time


def main():
    batch = 20
    memory = 10
    input = 200
    steps = 30
    output = 300

    with tf.Session() as sess:
        model = FSMN(memory, input, output)
        model._memory_weights = tf.Variable(np.arange(memory), dtype=tf.float32)
        tf.initialize_all_variables().run()
        w1 = model._W1.eval()
        w2 = model._W2.eval()
        bias = model._bias.eval()
        memory_weights = model._memory_weights.eval()
        inputs = np.random.rand(batch, steps, input).astype(np.float32)
        start = time.time()
        ret = sess.run(model(tf.constant(inputs, dtype=tf.float32)))
        print(str(time.time() - start), "(s)")

    expect_first_batch = []
    for i in range(steps):
        hidden = np.sum([memory_weights[j] * inputs[0][i - j] for j in range(0, min(memory, i + 1))], axis=0)
        expect_first_batch.append(np.dot(w1.T, inputs[0][i]) + np.dot(w2.T, hidden) + bias)

    expect_first_batch = np.array(expect_first_batch)
    real_first_batch = ret[0].reshape(-1, output)
    assert (np.absolute(expect_first_batch - real_first_batch) < 0.0001).all()
    tf.reset_default_graph()

if __name__ == '__main__':
    main()
