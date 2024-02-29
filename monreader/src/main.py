from features.evaluate_model import evaluate_model
from features.data_util import get_data
# from models.train_model import train_model
# from models.predict_model import predict_model
# from models import fd_model
import tensorflow as tf


def main():
    _, _, data = get_data()
    _, _, test_data = data
    results = evaluate_model(tf.keras.models.load_model('monreader\\src\\models\\fd_model.h5'), test_data)
    print(results)

if __name__ == '__main__':
    main()




