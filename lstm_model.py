import tensorflow as tf

class LSTMModel:
    def __init__(self, input_dim, hidden_unit1, hidden_unit2, n_features, dropout_rate, learning_rate):
        self.model = self.build_model(input_dim, hidden_unit1, hidden_unit2, n_features, dropout_rate, learning_rate)

    def load_model(self, save_path):
        self.model = tf.keras.models.load_model(save_path)

    def build_model(self, input_dim, hidden_unit1, hidden_unit2, n_features, dropout_rate, learning_rate):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(units=hidden_unit1, return_sequences=True, input_shape=(input_dim, n_features)))
        model.add(tf.keras.layers.LSTM(units=hidden_unit2, return_sequences=False))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(units=1, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

        # print(model.summary())

        return model


    def train(self, x, y, epochs, batch_size, save_path):
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=0)
        rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0)
        mcp = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_loss', verbose=0,
                                                 save_best_only=True, save_weights_only=True)
        tb = tf.keras.callbacks.TensorBoard('logs')

        history = self.model.fit(x, y, shuffle=True, epochs=epochs, callbacks=[es, rlr, mcp, tb],
                                 validation_split=0.2, verbose=1, batch_size=batch_size)

        self.model.load_weights(save_path)
        self.model.save(save_path)
        print(f'\n===> Save trained model to {save_path}\n')

        return history










