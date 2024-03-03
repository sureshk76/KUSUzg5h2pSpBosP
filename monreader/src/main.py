from features.evaluate_model import evaluate_model, predict_image
from features.data_util import get_data
from PIL import Image
import tensorflow as tf
import gradio as gr
import io
import base64
from transformers import pipeline
import numpy as np


def main():
    
    img_height, img_width = 180, 180
    model = tf.keras.models.load_model('monreader\\src\\models\\fd_model.h5')

    def predict_image_wrapper(image):
        return predict_image(image, img_height, img_width, model)

    inputs = gr.Image()
    outputs = gr.Textbox()

    # Create a Gradio interface with the predict_image function
    gr.Interface(fn=predict_image_wrapper, 
                 inputs=inputs, 
                 outputs=outputs, 
                 examples=['monreader\\src\\data\\0001_000000017.jpg', 'monreader\\src\\data\\0002_000000027.jpg', 'monreader\\src\\data\\0004_000000013.jpg'], 
                 title="Image Classification"
                 ).launch()


if __name__ == '__main__':
    main()
