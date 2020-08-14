import tensorflow as tf


converter = tf.lite.TFLiteConverter.from_saved_model("D:\桌面文件夹\大二下\大创大数据机器视觉\模型转换\model-8-8\yolo_v3_epoch_135_12")
# tflite_model = converter.convert()

# converter = tf.lite.TFLiteConverter.from_keras_model(yolo_v3)

converter.allow_custom_ops = True
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open("./yolo_v3_epoch_135_12.tflite", "wb").write(tflite_model)