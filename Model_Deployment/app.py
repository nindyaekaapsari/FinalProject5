from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.cm as cm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import Image, display
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('Modela2_Best.h5')

def predict_label(img_path):
    x = load_img(img_path, target_size=(224,224))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    answer = model.predict(x)
    answer =  np.argmax(answer, axis=1)
    return answer

def get_img_array(img_path):
    path = img_path
    img = image.load_img(path, target_size=(224,224,3))
    img = image.img_to_array(img)
    img = np.expand_dims(img , axis= 0 )
    return img

def pieplot(img_path):
    img = get_img_array(img_path)
    label = ['COVID-19', 'Normal', 'Pneumonia']
    a = model.predict(img)[0][0]*100
    b = model.predict(img)[0][1]*100
    c = model.predict(img)[0][2]*100
    data = [a, b,c]
    plt.figure(figsize =(10, 7))
    plt.title("Probabilitas Klasifikasi", fontsize = 20)
    plt.pie(data, labels = label, autopct='%1.1f%%', shadow=True, textprops={'fontsize': 20})
    plt.tight_layout()
    plt.savefig('./static/images/pieplot.png')
    imgs = './static/images/pieplot.png'
    return render_template('index.html', name = imgs )

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path , heatmap, cam_path="./static/images/cam_path.png", alpha=0.4):
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)

    display(Image(cam_path))

def image_prediction_and_visualization(path,last_conv_layer_name = "top_conv", model = model):
    img_array = get_img_array(path)

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    print("Citra dengan Heatmap")

    save_and_display_gradcam(path, heatmap)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            result = predict_label(img_path)
            if result == 0:
                prediction = 'COVID-19'
            elif result == 1:
                prediction = 'Normal'			
            elif result == 2:
                prediction = 'Pneumonia'
            pieplot(img_path)
            image_prediction_and_visualization(img_path)
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction)

    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run()