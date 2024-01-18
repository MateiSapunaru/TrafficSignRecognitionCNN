from keras.preprocessing.image import load_img
import numpy as np
from keras.models import load_model

model = load_model('model_saved.h5')
print(model.summary())

image = load_img('v_data/OCT2017/train/NORMAL/NORMAL-1384-1.jpeg', target_size=(224, 224))
img = np.array(image)
img = img / 255.0
img = img.reshape(1, 224, 224, 3)
label = model.predict(img)
print(label[0])
