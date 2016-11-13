import tensorflow as tf


class FSMN(object):
    def __init__(self, memory_size, input_size, output_size, dtype=tf.float32):
        self._memory_size = memory_size
        self._output_size = output_size
        self._input_size = input_size
        self._dtype = dtype
        self._build_graph()

    def _build_graph(self):
        self._W1 = tf.get_variable("fsmnn_w1", [self._input_size, self._output_size], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
        self._W2 = tf.get_variable("fsmnn_w2", [self._input_size, self._output_size], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
        self._bias = tf.get_variable("fsmnn_bias", [self._output_size], initializer=tf.constant_initializer(0.0, dtype=self._dtype))
        self._memory_weights = tf.get_variable("memory_weights", [self._memory_size], initializer=tf.constant_initializer(1.0, dtype=self._dtype))

    def __call__(self, input_data):
        batch_size = input_data.get_shape()[0].value
        num_steps = input_data.get_shape()[1].value

        memory_matrix = []
        for step in range(num_steps):
            left_num = tf.maximum(0, step + 1 - self._memory_size)
            right_num = num_steps - step - 1
            mem = self._memory_weights[tf.minimum(step, self._memory_size)::-1]
            d_batch = tf.diag(tf.pad(mem, [[left_num, right_num]]))
            memory_matrix.append([tf.concat(0, [[d_batch]] * batch_size)])
        memory_matrix = tf.concat(0, memory_matrix)

        h_hatt = tf.reduce_sum(tf.batch_matmul(memory_matrix, tf.concat(0, [[input_data]] * num_steps)), 2)
        h_hatt = tf.transpose(h_hatt, perm=[1, 0, 2])
        h = tf.batch_matmul(input_data, tf.concat(0, [[self._W1]] * batch_size))
        h += tf.batch_matmul(h_hatt, tf.concat(0, [[self._W2]] * batch_size)) + self._bias
        return h
