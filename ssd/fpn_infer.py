import os
import tensorflow as tf
import numpy as np
import time

model_file = './fpn_resnet50_transform.pb'
input_node = 'Placeholder:0'
output_node = 'postprocess_FPN/GatherV2_1:0'

def do_infer(image):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(model_file, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            graph = tf.get_default_graph()
            out_tensor = graph.get_tensor_by_name(output_node)
            input_tensor = graph.get_tensor_by_name(input_node)
            result = sess.run(out_tensor, feed_dict={input_tensor:image})
            print(result.shape)

if __name__ == '__main__':
    image = np.random.randint(0, 255, (224, 224, 3), np.uint8)
    do_infer(image)
