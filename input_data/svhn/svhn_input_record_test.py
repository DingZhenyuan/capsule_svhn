""" Tests for svhn input_record. """

import numpy as np
import tensorflow as tf

import svhn_input_record

SVHN_DATA_DIR = '../testdata/svhn/'

class SVHNInputRecordTest(tf.test.TestCase):

    def testTrain(self):
        with self.test_session(graph=tf.Graph()) as sess:
            features = svhn_input_record.inputs(
                data_dir=SVHN_DATA_DIR,
                batch_size=1,
                split='train',
                batch_capacity=2
            )
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            images, labels, recons_image = sess.run(
                [features['images'], features['labels'], features['recons_image']]
            )
            print(images)
            print(labels)
            print(recons_image)

            coord.request_stop()
            for thread in threads:
                thread.join()


if __name__ == '__main__':
    tf.test.main()

