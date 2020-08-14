import tensorflow as tf
from yolo.yolo_v3_test import YOLOV3
from configuration import test_file_dir, test_picture_dir, save_model_dir, CHANNELS, CATEGORY_NUM, IMAGE_HEIGHT, \
    IMAGE_WIDTH

import h5py

def resize_image_with_pad(image):
    image_tensor = tf.image.resize_with_pad(image=image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
    image_tensor = tf.cast(image_tensor, tf.float32)
    image_tensor = image_tensor / 255.0
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    return image_tensor

image = tf.io.decode_jpeg(contents=tf.io.read_file("test_data/JPEGImages/0009.jpg"), channels=CHANNELS)
img_tensor = resize_image_with_pad(image)
print("shape is {}, type is {}".format(img_tensor.shape, img_tensor.dtype))


yolo_v3 = YOLOV3(out_channels=3 * (CATEGORY_NUM + 5))
print("-----------------1-")
yolo_v3.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
print("-----------------2-")
yolo_v3.load_weights(filepath=save_model_dir + "epoch-135")
print("-----------------3-")
yolo_output = yolo_v3(img_tensor, training=False)
print(yolo_output)
print("-----------------4-")
yolo_v3.summary()
print(type(yolo_output[1]))
# 保存模型结构及权重
# tf.keras.utils.plot_model(yolo_v3, "my_first_model_with_shape_info.png", show_shapes=True)

#yolo_v3.save('./model/yolo_v3_epoch_135_3.h5')
yolo_v3.save('./model-8-8/yolo_v3_epoch_135_12', save_format="tf")



# converter = tf.lite.TFLiteConverter.from_keras_model(yolo_v3)
# #converter.allow_custom_ops = True
# converter.experimental_new_converter = True
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                        tf.lite.OpsSet.SELECT_TF_OPS]
# tflite_model = converter.convert()
# open(save_model_dir + "yolo_v3_epoch_135_3.tflite", "wb").write(tflite_model)