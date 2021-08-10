import streamlit as st
import tensorflow as tf

import cv2
from PIL import Image, ImageOps
import numpy as np
st.balloons()

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('model.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

  st.title("Binary Classifier")
  st.markdown('The model will predict wheather the image contain a ship or a truck')
  file = st.file_uploader("Please upload an image of ship or truck", type=["jpg", "png"])

  st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    size = (32,32)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis,...]
    
    prediction = model.predict(img_reshape)

    return prediction
if file is None:
    st.text("Please upload an image file")

else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    st.balloons()
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    names = ['ship','truck']
    str ="The image seems to be contain a "+ names[np.argmax(predictions)]
    st.success(str)
    st.write('The model accuracy found out to be -87%')

    #st.write(np.argmax(predictions))
    
    #st.balloons()


