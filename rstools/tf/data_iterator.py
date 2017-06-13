import threading
import tensorflow as tf


class IteratorQueue(object):
    def __init__(self,
                 iterator_fn,
                 coord,
                 placeholders,
                 queue_size=1024,
                 pad_sequence=False):
        self.iterator_fn = iterator_fn
        self.coord = coord
        self.threads = []

        self.placeholders = placeholders

        # or typical queue
        if pad_sequence:
            queue_fn = tf.PaddingFIFOQueue
        else:
            queue_fn = tf.FIFOQueue

        self.queue = queue_fn(
            queue_size,
            list(map(lambda x: x.dtype, self.placeholders)),
            shapes=list(map(lambda x: x.shape.as_list(), self.placeholders)))

        self.enqueue = self.queue.enqueue(self.placeholders)

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def thread_main(self, sess, i_thread):
        stop = False
        while not stop:
            thread_iterator = self.iterator_fn(i_thread)
            for data in thread_iterator:
                if self.coord.should_stop():
                    self.stop_threads()
                    stop = True
                    break
                sess.run(
                    self.enqueue,
                    feed_dict=dict(zip(self.placeholders, data)))

    def stop_threads(self):
        for t in self.threads:
            t.stop()

    def start_threads(self, sess, n_threads=1):
        for i_thread in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess, i_thread))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
