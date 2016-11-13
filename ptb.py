import numpy as np
import tensorflow as tf
import reader
import time
import sys
from fsmn import *


class PTBModel:

    @property
    def optimizer(self):
        return self._optimizer

    def __init__(self):
        # internal setting
        self._optimizer = tf.train.AdamOptimizer()

        # config
        self._batch_size = 200
        self._num_steps = 50
        self._hidden_size = 200
        self._vocab_size = 10000
        self._keep_prob = 1.0
        self._max_grad_norm = 5  # parameters L2Norm sum limits
        self._memory_size = 20

        # input and output variables
        self._input_data = tf.placeholder(tf.int32, [self._batch_size, self._num_steps])
        self._targets = tf.placeholder(tf.int32, [self._batch_size, self._num_steps])
        self._cost = None
        self._train_op = None
        self._logits = None

        self.train_writer = None
        self._build_graph(True)

    def _build_graph(self, is_training):
        # Load predefined layer "embedding"
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [self._vocab_size, 200])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        # Add dropout after embedding layer
        if is_training:
            inputs = tf.nn.dropout(inputs, self._keep_prob)

        # Claculate FSMN Layer
        #  FSMN
        with tf.variable_scope('fsmn1'):
            fsmn = FSMN(self._memory_size, 200, 400)
            outputs = fsmn(inputs)
            # Relu
            outputs = tf.nn.relu(outputs)
            # Dropout
            if is_training:
                outputs = tf.nn.dropout(outputs, self._keep_prob)
        # with tf.variable_scope('fsmn2'):
        #     fsmn = FSMN(self._memory_size, self._hidden_size, self._hidden_size)
        #     outputs = fsmn(outputs)
        #     # Relu
        #     outputs = tf.nn.relu(outputs)
        #     # Dropout
        #     if is_training:
        #         outputs = tf.nn.dropout(outputs, self._keep_prob)
        outputs = tf.reshape(outputs, [-1, 400])

        # Final output layer for getting word label
        # input shape is (batch_size x num_steps, hidden_size)
        # data style [sequence1-1, sequence1-2, sequence1-3, ... , sequenceN-M]
        softmax_w = tf.get_variable("softmax_w", [400, self._vocab_size])
        softmax_b = tf.get_variable("softmax_b", [self._vocab_size])
        self._logits = tf.matmul(outputs, softmax_w) + softmax_b

        # loss function
        # logits shape is (batch_size x num_steps, vocab_size)
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [self._logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([self._batch_size * self._num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / self._batch_size
        tf.scalar_summary("Cost", self._cost)
        self._summaries = tf.merge_all_summaries()

        if not is_training:
            return

        # Gradient calculator
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self._max_grad_norm)
        self._train_op = self._optimizer.apply_gradients(zip(grads, tvars))

    def _one_loop_setup(self, eval_op):
        fetches = []
        fetches.append(self._cost)
        fetches.append(eval_op)
        fetches.append(self._summaries)
        feed_dict = {}
        return fetches, feed_dict

    def _run_epoch(self, session, data, eval_op, verbose=False):
        epoch_size = ((len(data) // self._batch_size) - 1) // self._num_steps
        start_time = time.time()
        costs = 0.0
        iters = 0

        for step, (x, y) in enumerate(reader.ptb_iterator(data, self._batch_size, self._num_steps)):
            fetches, feed_dict = self._one_loop_setup(eval_op)
            feed_dict[self._input_data] = x
            feed_dict[self._targets] = y

            res = session.run(fetches, feed_dict)
            self.train_writer.add_summary(res[2], step / 13)
            cost = res[0]

            costs += cost
            iters += self._num_steps

            if verbose and step % (epoch_size // 10) == 10:
                sys.stdout.write("%.3f perplexity: %.3f speed: %.0f wps\n" %
                    (step * 1.0 / epoch_size, np.exp(costs / iters),
                    iters * self._batch_size * self._num_steps / (time.time() - start_time)))
                sys.stdout.flush()

        return np.exp(costs / iters)

    def train(self, session, data):
        return self._run_epoch(session, data, self._train_op, verbose=True)

    def evaluate(self, session, data):
        return self._run_epoch(session, data, tf.no_op())

    def predict(self, session, data, word_to_id):
        def _get_word_fromid(word_to_id, search_id):
            for word, wid in word_to_id.items():
                if wid == search_id:
                    return word

        for step, (x, y) in enumerate(reader.ptb_iterator(data, self._batch_size, self._num_steps)):
            fetches, feed_dict = self._one_loop_setup(self._logits)
            feed_dict[self._input_data] = x
            feed_dict[self._targets] = y

            res = session.run(fetches, feed_dict)
            label = res[1]
            label = np.argmax(label, 1)
            y = np.reshape(y, (self._batch_size * self._num_steps))
            for pre, real in zip(label, y):
                sys.stdout.write("Predict %s : Real %s\n" % (_get_word_fromid(word_to_id, pre), _get_word_fromid(word_to_id, real)))


def main():
    sys.stdout.write("start ptb")
    raw_data = reader.ptb_raw_data("")
    train_data, valid_data, test_data, word_to_id = raw_data

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-0.04, 0.04)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = PTBModel()

        saver = tf.train.Saver()
        tf.initialize_all_variables().run()
        model.train_writer = tf.train.SummaryWriter('./train', graph=session.graph)

        for i in range(13):
            sys.stdout.write("Epoch: %d\n" % (i + 1))
            train_perplexity = model.train(session, train_data)
            sys.stdout.write("Epoch: %d Train Perplexity: %.3f\n" % (i + 1, train_perplexity))
            valid_perplexity = model.evaluate(session, valid_data)
            sys.stdout.write("Epoch: %d Valid Perplexity: %.3f\n" % (i + 1, valid_perplexity))
            test_perplexity = model.evaluate(session, test_data)
            sys.stdout.write("Epoch: %d Test Perplexity: %.3f\n" % (i + 1, test_perplexity))

        # model.predict(session, test_data, word_to_id)
        saver.save(session, 'model.ckpt')

if __name__ == '__main__':
    main()
