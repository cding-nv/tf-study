This can work.

ssd model is from http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz

frozen_inference_graph.pb can work directly. Don't need the below freeze_graph command.
freeze_graph --input_saved_model_dir=saved_model --input_checkpoint=model.ckpt --output_graph=output.pb --output_node_names='detection_boxes,detection_scores,num_detections,detection_classes' --input_binary=true





./test ../../models/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb detection_boxes,detection_scores,num_detections,detection_classes ./dog.png
