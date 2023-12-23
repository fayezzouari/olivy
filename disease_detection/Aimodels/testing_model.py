
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

input_image= plt.imread('C:\code\Olive_Leaf_Disease_Datasets_After_Preprocessing\__training\Healthy\B-2_jpg.rf.52f17683407e0ffc2b5d9912ea47cc7b.jpg')
model = tf.keras.models.load_model('C:\code\Olive_Leaf_Disease_Datasets_After_Preprocessing\saved_models\,2')
class_names=['Aculos','Healthy','olive peacock']


img_array = tf.keras.preprocessing.image.img_to_array(input_image)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)

predicted_class = class_names[np.argmax(predictions[0])]
confidence = round(100 * (np.max(predictions[0])), 2)

print (predicted_class)
print ('confidence:',confidence)