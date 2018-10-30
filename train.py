from tensorflow.keras.layers  import Input, Conv2D, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten
from tensorflow.keras import Model
import pickle
import argparse

def create_model():
    input = Input(shape=(27,27,3))
    x = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(input)
    x = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)

    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(25, activation='linear')(x)

    model = Model(inputs = input, outputs = x)
    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])

    print(model.summary())

    return model

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input data dir")
    args = vars(ap.parse_args())

    IN_DIR = args['input']
    with open(f'{IN_DIR}/train.pkl', 'rb') as f:
        data = pickle.load(f)
        train_x = data['X']
        train_y = data['Y']

    model = create_model()
    history = model.fit(train_x, train_y, batch_size=32, epochs = 10)




