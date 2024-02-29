import tensorflow as tf

def get_data():
    img_height, img_width, img_channels = 180, 180, 3

    batch_size=32

    data_dir = 'monreader\\src\\data\\images\\training'

    test_data_dir = 'monreader\\src\\data\\images\\testing'

    loading_dataste_seed = 0


    train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=loading_dataste_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size

    )
    val_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=loading_dataste_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size

    )


    test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_dir,
    seed=loading_dataste_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size

    )

    classes = train_data.class_names
    shape = (img_height, img_width, img_channels)
    data = (train_data, val_data, test_data)

    return shape, classes, data
