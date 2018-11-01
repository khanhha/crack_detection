from tensorflow.keras.layers  import Input, Conv2D, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten
from tensorflow.keras import Model

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

    return model