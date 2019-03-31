"""Author: Brandon Trabucco, Copyright 2019
Extract region features for images."""

import tensorflow as tf
import numpy as np
import pickle as pkl
import os
import time
import threading
import FasterRCNN.inference
from FasterRCNN.inference import create_r101fpn_mask_rcnn_model_graph as create_model

tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_string("image_folder", "./train_images/", "Path to the image files.")
tf.flags.DEFINE_string("file_type", "jpg", "The type of file to look for.")
tf.flags.DEFINE_string("output_dir", "./train_features/", "Output data directory.")
tf.flags.DEFINE_integer("num_threads", 4, "Number of threads to use when serializing the dataset.")
tf.flags.DEFINE_integer("queue_size", 64, "The number of examples to extract at once.")
tf.flags.DEFINE_integer("image_size", 800, "The size of the image larger dimension.")
tf.flags.DEFINE_integer("start_at_file_index", 0, "The number of tfrecords to skip.")
tf.flags.DEFINE_string("gpu", "0", "Which GPU to use.")
tf.flags.DEFINE_float("memory", 1.0, "How much GPU memory to use.")
FLAGS = tf.flags.FLAGS

def rescale_edge(images, new_edge, shorter_edge=False):
    def compute_longer_edge(height, width, new_edge):
        return tf.cast(width * new_edge / height, tf.int32)
    height, width = tf.shape(images)[0], tf.shape(images)[1]
    height_smaller_than_width = tf.less_equal(height, width) if shorter_edge else tf.less_equal(width, height)
    new_height_and_width = tf.cond(
        height_smaller_than_width,
        lambda: (new_edge, compute_longer_edge(height, width, new_edge)),
        lambda: (compute_longer_edge(width, height, new_edge), new_edge))
    return tf.image.resize_images(images, new_height_and_width)

def build_pipeline(image_folder, file_type, image_size):
    dataset = tf.data.Dataset.list_files(os.path.join(image_folder, "*." + file_type))
    def lazy_load_image(image_filename):
        with tf.gfile.GFile(image_filename, "rb") as f:
            return f.read()
    def filename_to_tensor(image_filename):
        image_bytes = tf.py_func(lazy_load_image, [image_filename], tf.string)
        image_tensor = tf.image.decode_jpeg(image_bytes, channels=3)
        return {"image": rescale_edge(image_tensor, image_size),
        "filename": image_filename}
    dataset = dataset.map(filename_to_tensor, num_parallel_calls=FLAGS.num_threads)
    dataset = dataset.apply(tf.contrib.data.ignore_errors()).batch(1)
    def prepare_final_batch(x):
        return {"image": tf.transpose(tf.cast(x["image"], tf.float32), [0, 3, 1, 2]),
            "filename": x["filename"]}
    dataset = dataset.map(prepare_final_batch, num_parallel_calls=FLAGS.num_threads)
    dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=2))
    return dataset.make_one_shot_iterator().get_next()

class Extractor(object):
    def __init__(self, image_folder, file_type, image_size, gpu_id=0, memory_fraction=1):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = build_pipeline(image_folder, file_type, image_size)
            self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(visible_device_list=str(gpu_id),
                    per_process_gpu_memory_fraction=memory_fraction)))
            self.fetch_dict = create_model(self.x["image"], self.sess)
    def extract(self):
        return self.sess.run([self.x["filename"], self.fetch_dict])

def write_all_features(data_queue, output_dir, is_finished):
    while len(data_queue) > 0 or not is_finished[0]:
        try:
            x = data_queue.pop(0)
        except Exception as e:
            time.sleep(0.5)
            continue
        target_file = os.path.join(output_dir, os.path.basename(
            x[0][0].decode('utf8'))) + ".pkl"
        print("saved: " + target_file)
        with open(target_file, "wb") as f:
            pkl.dump(x[1], f)

if __name__ == "__main__":
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    data_queue, is_finished = [], [False]
    extractor = Extractor(FLAGS.image_folder, FLAGS.file_type, FLAGS.image_size, 
        gpu_id=FLAGS.gpu, memory_fraction=FLAGS.memory)
    serializer_threads = [
        threading.Thread(target=write_all_features, args=(data_queue, FLAGS.output_dir, 
        is_finished)) for i in range(FLAGS.num_threads)]
    for thread in serializer_threads:
        thread.start()
    while True:
        try:
            if len(data_queue) < FLAGS.queue_size:
                data_queue.append(extractor.extract())
            else:
                time.sleep(0.5)
        except Exception as e:
            print(e)
            break
    is_finished[0] = True
    tf.train.Coordinator().join(serializer_threads)