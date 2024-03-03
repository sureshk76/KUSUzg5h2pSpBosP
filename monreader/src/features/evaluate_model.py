from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

def evaluate_model(model, test_data):
    results = {}    
    predicted_labels = []
    true_labels = []
    all_images = []
    for images, labels in test_data:
        true_labels.extend(labels.numpy())
        predicted_labels.extend(tf.argmax(model.predict(images), axis=1).numpy())

    # Accuracy
    results['accuracy'] = accuracy_score(true_labels, predicted_labels)
    
    # F1 Score
    results['f1_score'] = f1_score(true_labels, predicted_labels)
        
    cm = confusion_matrix(true_labels, predicted_labels)
    
    class_names = test_data.class_names
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    return results

def predict_image(img, img_height, img_width, model):
    # Preprocess the image
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.resize(img, (img_height, img_width))
    img = np.expand_dims(img, axis=0)
    # img = img / 255.0
    # Predict the class probabilities for the image
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    if predicted_class == 0:
        result = "it's a flip"
    elif predicted_class == 1:
        result = "it's not flipping"
    return f"{result}"