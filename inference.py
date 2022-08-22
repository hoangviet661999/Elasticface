from cv2 import threshold
import tensorflow as tf
import align.detect_face
import cv2

minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
input_image_size = 160

def detect_face(img):
    img = cv2.resize(img, (input_image_size, input_image_size))

    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, 'align')
            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            img_cropped = img[int(bounding_boxes[0, 1]):int(bounding_boxes[0, 3]), int(bounding_boxes[0, 0]):int(bounding_boxes[0, 2]), :]
            img_cropped = cv2.resize(img_cropped, (input_image_size, input_image_size))


            cv2.rectangle(img, (int(bounding_boxes[0, 0]), int(bounding_boxes[0, 1])), (int(bounding_boxes[0, 2]), int(bounding_boxes[0, 3])), (0, 255, 0), 2)
            cv2.putText(img, 'best_name', (int(bounding_boxes[0, 1]), int(bounding_boxes[0, 3])+10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
            cv2.imshow('img', img)
            cv2.waitKey(0)
        
