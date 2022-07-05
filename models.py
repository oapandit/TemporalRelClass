from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Embedding, merge, Reshape
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from tensorflow.keras.layers import add, concatenate, multiply, subtract
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, AveragePooling2D
from constants import *
from keras_custom_layer import TemporalMaxPooling, global_loss
from tensorflow.keras import optimizers


class models:

    def __init__(self, word_context_length, word_vector_size, char_vector_size, num_interval_relations,
                 num_point_relations):
        self.word_context_length = word_context_length
        self.word_vector_size = word_vector_size
        self.char_vector_size = char_vector_size
        self.num_interval_relations = num_interval_relations
        self.num_point_relations = num_point_relations

        self.num_word_for_char_emd = NUM_WORD_FOR_CHAR_EMD  # considering only event head word; so 1
        self.num_unique_chars = 28
        self.embedding_dim = 10
        self.char_feature = 64
        self.pos_vec_len = 1

        self.is_fasttext = False
        self.is_tense = False

    def _get_optimizers_with_custom_learning_rates(self, optimizer):

        # ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta']

        if optimizer == "rmsprop":
            return optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)

        elif optimizer == "sgd":
            return optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

        else:
            return optimizer

    def get_neural_model(self, is_point_relation, hyper_params):
        '''
        Based on is_point_relation value return either interval or point relation model
        :param is_point_relation:
        :param hyper_params:
        :return:
        '''
        if is_point_relation:
            return self.get_endpoint_classification_model(*hyper_params)
        else:
            return self.get_interval_relation_classification_model(*hyper_params)

    def get_interval_relation_classification_model(self, interaction,
                                                   is_char=False,
                                                   is_pos=False,
                                                   nn="BGRU",
                                                   input_drop_out=0.2,
                                                   num_rnn_neurons=16,
                                                   cnn_filters=32,
                                                   opt='adam',
                                                   output_dense=32):

        is_return_seq = False
        opt = self._get_optimizers_with_custom_learning_rates(opt)

        event_word_input_1 = Input(shape=(self.word_context_length, self.word_vector_size,), name='event_word_input_1')
        event_word_input_2 = Input(shape=(self.word_context_length, self.word_vector_size,), name='event_word_input_2')

        if is_char:
            emd_td = TimeDistributed(
                Embedding(self.num_unique_chars + 2, self.embedding_dim, input_length=self.char_vector_size))
            gru_td = TimeDistributed(GRU(self.char_feature, dropout=input_drop_out, recurrent_dropout=input_drop_out))
            # gru_td = GRU(self.char_feature, dropout=input_drop_out, recurrent_dropout=input_drop_out)

            event_char_input_1 = Input(shape=(self.num_word_for_char_emd, self.char_vector_size,),
                                       name='event_char_input_1')
            event_char_input_2 = Input(shape=(self.num_word_for_char_emd, self.char_vector_size,),
                                       name='event_char_input_2')

            c1 = emd_td(event_char_input_1)
            c2 = emd_td(event_char_input_2)
            event_char_feat_1 = gru_td(c1)
            event_char_feat_2 = gru_td(c2)

        else:
            pass

        if self.is_fasttext:
            self.num_word_for_char_emd = 1
            self.char_vector_size = 300
            event_char_input_1 = Input(shape=(self.num_word_for_char_emd, self.char_vector_size,),
                                       name='event_char_input_1')
            event_char_input_2 = Input(shape=(self.num_word_for_char_emd, self.char_vector_size,),
                                       name='event_char_input_2')

            event_char_feat_1 = event_char_input_1
            event_char_feat_2 = event_char_input_2
            is_char = True

        event_input_1 = event_word_input_1
        event_input_2 = event_word_input_2

        if (nn == "BRNN"):
            brnn = Bidirectional(SimpleRNN(num_rnn_neurons, dropout=input_drop_out, recurrent_dropout=input_drop_out,
                                           return_sequences=is_return_seq))
            x1 = brnn(event_input_1)
            x2 = brnn(event_input_2)



        elif (nn == "BLSTM"):
            blstm = Bidirectional(LSTM(num_rnn_neurons, dropout=input_drop_out, recurrent_dropout=input_drop_out,
                                       return_sequences=is_return_seq))
            x1 = blstm(event_input_1)
            x2 = blstm(event_input_2)

        elif (nn == "RNN"):
            rnn = SimpleRNN(num_rnn_neurons, dropout=input_drop_out, recurrent_dropout=input_drop_out,
                            return_sequences=is_return_seq)
            x1 = rnn(event_input_1)
            x2 = rnn(event_input_2)

        elif (nn == "LSTM"):
            lstm = LSTM(num_rnn_neurons, dropout=input_drop_out, recurrent_dropout=input_drop_out,
                        return_sequences=is_return_seq)
            x1 = lstm(event_input_1)
            x2 = lstm(event_input_2)
        elif (nn == "GRU"):
            gru = GRU(num_rnn_neurons, dropout=input_drop_out, recurrent_dropout=input_drop_out,
                      return_sequences=is_return_seq)
            x1 = gru(event_input_1)
            x2 = gru(event_input_2)
        else:
            bgru = Bidirectional(GRU(num_rnn_neurons, dropout=input_drop_out, recurrent_dropout=input_drop_out,
                                     return_sequences=is_return_seq))
            x1 = bgru(event_input_1)
            x2 = bgru(event_input_2)

        if is_return_seq:
            # td = TimeDistributed(Dense(num_rnn_neurons))
            # x1 = td(x1)
            # x2 = td(x2)

            x1 = TemporalMaxPooling()(x1)
            x2 = TemporalMaxPooling()(x2)

        if self.is_tense:
            emd_pos = Embedding(10, 5, input_length=1)
            event_tense_input_1 = Input(shape=(self.pos_vec_len,), name='event_tense_input_1')
            event_tense_input_2 = Input(shape=(self.pos_vec_len,), name='event_tense_input_2')

            event_tense_feat_1 = emd_pos(event_tense_input_1)
            event_tense_feat_2 = emd_pos(event_tense_input_2)

            event_tense_feat_1 = Reshape((-1,))(event_tense_feat_1)
            event_tense_feat_2 = Reshape((-1,))(event_tense_feat_2)

            x1 = concatenate([event_tense_feat_1, x1], axis=1)
            x2 = concatenate([event_tense_feat_2, x2], axis=1)

        if is_pos:
            emd_pos = Embedding(10, 5, input_length=1)
            event_pos_input_1 = Input(shape=(self.pos_vec_len,), name='event_pos_input_1')
            event_pos_input_2 = Input(shape=(self.pos_vec_len,), name='event_pos_input_2')

            event_pos_feat_1 = emd_pos(event_pos_input_1)
            event_pos_feat_2 = emd_pos(event_pos_input_2)

            event_pos_feat_1 = Reshape((-1,))(event_pos_feat_1)
            event_pos_feat_2 = Reshape((-1,))(event_pos_feat_2)

            x1 = concatenate([event_pos_feat_1, x1], axis=1)
            x2 = concatenate([event_pos_feat_2, x2], axis=1)

        if is_char:

            if not is_return_seq:
                event_char_feat_1 = Reshape((-1,))(event_char_feat_1)
                event_char_feat_2 = Reshape((-1,))(event_char_feat_2)

                x1 = concatenate([event_char_feat_1, x1], axis=1)
                x2 = concatenate([event_char_feat_2, x2], axis=1)

            else:

                x1 = Reshape((1, -1,))(x1)
                x2 = Reshape((1, -1,))(x2)

                x1 = concatenate([event_char_feat_1, x1], axis=2)
                x2 = concatenate([event_char_feat_2, x2], axis=2)

        if interaction == "CONCATE":
            merged = concatenate([x1, x2])

        elif interaction == "ADDITION":
            merged = add([x1, x2])

        elif interaction == "SUBTRACTION":
            merged = subtract([x1, x2])

        elif interaction == "MULTIPLICATION":
            merged = multiply([x1, x2])

        elif interaction == "MLP":
            merged = concatenate([x1, x2])
            merged = Dense(output_dense, activation='relu')(merged)
            merged = Dense(output_dense, activation='relu')(merged)

        elif interaction == "CNN":

            x1 = Reshape((1, -1,))(x1)
            x2 = Reshape((1, -1,))(x2)
            merged = concatenate([x1, x2], axis=1)
            conv = Conv1D(filters=cnn_filters, kernel_size=3, padding='same', activation='relu')(merged)
            merged = Flatten()(conv)

        else:

            x1 = Reshape((1, -1,))(x1)
            x2 = Reshape((1, -1,))(x2)
            merged = concatenate([x1, x2], axis=1)
            merged = Reshape((2, -1, 1))(merged)

            conv1 = Conv2D(filters=cnn_filters, kernel_size=(2, 4), activation='relu')(merged)
            conv1 = MaxPooling2D(pool_size=(1, 2))(conv1)

            conv2 = Conv2D(filters=cnn_filters, kernel_size=(2, 8), activation='relu')(merged)
            conv2 = MaxPooling2D(pool_size=(1, 2))(conv2)

            conv3 = Conv2D(filters=cnn_filters, kernel_size=(2, 16), activation='relu')(merged)
            conv3 = MaxPooling2D(pool_size=(1, 2))(conv3)

            conv4 = Conv2D(filters=cnn_filters, kernel_size=(2, 32), activation='relu')(merged)
            conv4 = MaxPooling2D(pool_size=(1, 2))(conv4)

            conv1 = Reshape((-1,))(conv1)
            conv2 = Reshape((-1,))(conv2)
            conv3 = Reshape((-1,))(conv3)
            conv4 = Reshape((-1,))(conv4)
            merged = concatenate([conv1, conv2, conv3, conv4], axis=1)

        main_loss = Dense(self.num_interval_relations, activation='softmax')(merged)

        if self.is_tense:

            if is_pos:
                if is_char:
                    model = Model(
                        inputs=[event_word_input_1, event_char_input_1, event_pos_input_1, event_tense_input_1,
                                event_word_input_2,
                                event_char_input_2, event_pos_input_2, event_tense_input_2],
                        outputs=[main_loss])
                else:
                    model = Model(
                        inputs=[event_word_input_1, event_pos_input_1, event_tense_input_1, event_word_input_2,
                                event_pos_input_2, event_tense_input_2],
                        outputs=[main_loss])

            else:
                if is_char:
                    model = Model(
                        inputs=[event_word_input_1, event_char_input_1, event_word_input_2, event_char_input_2],
                        outputs=[main_loss])
                else:
                    model = Model(inputs=[event_word_input_1, event_word_input_2], outputs=[main_loss])

        else:

            if is_pos:
                if is_char:
                    model = Model(inputs=[event_word_input_1, event_char_input_1, event_pos_input_1, event_word_input_2,
                                          event_char_input_2, event_pos_input_2],
                                  outputs=[main_loss])
                else:
                    model = Model(inputs=[event_word_input_1, event_pos_input_1, event_word_input_2, event_pos_input_2],
                                  outputs=[main_loss])

            else:
                if is_char:
                    model = Model(
                        inputs=[event_word_input_1, event_char_input_1, event_word_input_2, event_char_input_2],
                        outputs=[main_loss])
                else:
                    model = Model(inputs=[event_word_input_1, event_word_input_2], outputs=[main_loss])

        print(model.summary())

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def get_word_emb_cnn(self, num_relations, opt="adam", output_dense=32, cnn_filters=32):

        event_input_1 = Input(shape=(self.word_vector_size,), name='event_input_1')
        event_input_2 = Input(shape=(self.word_vector_size,), name='event_input_2')

        x1 = Reshape((1, -1,))(event_input_1)
        x2 = Reshape((1, -1,))(event_input_2)
        merged = concatenate([x1, x2], axis=1)
        conv = Conv1D(filters=cnn_filters, kernel_size=3, padding='same', activation='relu')(merged)

        # Deep CNN
        conv = Conv1D(filters=cnn_filters / 2, kernel_size=3, padding='same', activation='relu')(conv)
        conv = Conv1D(filters=cnn_filters / 4, kernel_size=3, padding='same', activation='relu')(conv)

        merged = Flatten()(conv)
        merged = Dense(output_dense, activation='relu')(merged)
        main_loss = Dense(num_relations, activation='softmax')(merged)

        model = Model(inputs=[event_input_1, event_input_2], outputs=[main_loss])

        print(model.summary())

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        print('compilation complete')

        return model

    def get_endpoint_classification_model(self, interaction, is_char=False, nn="BGRU", input_drop_out=0.2,
                                          num_rnn_neurons=16, cnn_filters=32, opt='adam', output_dense=32):

        event_word_input_1 = Input(shape=(self.word_context_length, self.word_vector_size,), name='event_word_input_1')
        event_word_input_2 = Input(shape=(self.word_context_length, self.word_vector_size,), name='event_word_input_2')

        if is_char:
            event_char_input_1 = Input(shape=(self.word_context_length, self.char_vector_size,),
                                       name='event_char_input_1')
            x = TimeDistributed(
                Embedding(self.num_unique_chars, self.embedding_dim, input_length=self.char_vector_size))(
                event_char_input_1)
            event_char_feat_1 = TimeDistributed(
                GRU(self.char_feature, dropout=input_drop_out, recurrent_dropout=input_drop_out))(x)

            event_char_input_2 = Input(shape=(self.word_context_length, self.char_vector_size,),
                                       name='event_char_input_2')
            x = TimeDistributed(
                Embedding(self.num_unique_chars, self.embedding_dim, input_length=self.char_vector_size))(
                event_char_input_2)
            event_char_feat_2 = TimeDistributed(
                GRU(self.char_feature, dropout=input_drop_out, recurrent_dropout=input_drop_out))(x)

            event_input_1 = concatenate([event_char_feat_1, event_word_input_1], axis=2)
            event_input_2 = concatenate([event_char_feat_2, event_word_input_2], axis=2)

        else:
            event_input_1 = event_word_input_1
            event_input_2 = event_word_input_2

        if (nn == "BRNN"):
            brnn = Bidirectional(SimpleRNN(num_rnn_neurons, dropout=input_drop_out, recurrent_dropout=input_drop_out))
            x1 = brnn(event_input_1)
            x2 = brnn(event_input_2)

        elif (nn == "BLSTM"):
            blstm = Bidirectional(LSTM(num_rnn_neurons, dropout=input_drop_out, recurrent_dropout=input_drop_out))
            x1 = blstm(event_input_1)
            x2 = blstm(event_input_2)

        elif (nn == "RNN"):
            rnn = SimpleRNN(num_rnn_neurons, dropout=input_drop_out, recurrent_dropout=input_drop_out)
            x1 = rnn(event_input_1)
            x2 = rnn(event_input_2)

        elif (nn == "LSTM"):
            lstm = LSTM(num_rnn_neurons, dropout=input_drop_out, recurrent_dropout=input_drop_out)
            x1 = lstm(event_input_1)
            x2 = lstm(event_input_2)
        elif (nn == "GRU"):
            gru = GRU(num_rnn_neurons, dropout=input_drop_out, recurrent_dropout=input_drop_out)
            x1 = gru(event_input_1)
            x2 = gru(event_input_2)
        else:
            bgru = Bidirectional(
                GRU(num_rnn_neurons, dropout=input_drop_out, recurrent_dropout=input_drop_out, return_sequences=True))
            td = TimeDistributed(Dense(num_rnn_neurons))
            x1 = bgru(event_input_1)
            x1 = td(x1)
            x2 = bgru(event_input_2)
            x2 = td(x2)

        if interaction == "CONCATE":
            merged = concatenate([x1, x2])

        elif interaction == "ADDITION":
            merged = add([x1, x2])

        elif interaction == "SUBTRACTION":
            merged = subtract([x1, x2])

        elif interaction == "MULTIPLICATION":
            merged = multiply([x1, x2])

        elif interaction == "MLP":
            merged = concatenate([x1, x2])
            merged = Dense(output_dense, activation='relu')(merged)
            merged = Dense(output_dense, activation='relu')(merged)

        elif interaction == "CNN":

            x1 = Reshape((1, -1,))(x1)
            x2 = Reshape((1, -1,))(x2)
            merged = concatenate([x1, x2], axis=1)
            merged = Reshape((2, -1, 1))(merged)
            conv1 = Conv2D(filters=cnn_filters, kernel_size=(2, 5), activation='relu')(merged)
            conv1 = AveragePooling2D(pool_size=(1, 2))(conv1)
            conv2 = Conv2D(filters=cnn_filters, kernel_size=(2, 10), activation='relu')(merged)
            conv3 = Conv2D(filters=cnn_filters, kernel_size=(2, 15), activation='relu')(merged)
            conv4 = Conv2D(filters=cnn_filters, kernel_size=(2, 20), activation='relu')(merged)

            conv1 = Reshape((-1,))(conv1)
            conv2 = Reshape((-1,))(conv2)
            conv3 = Reshape((-1,))(conv3)
            conv4 = Reshape((-1,))(conv4)
            merged = concatenate([conv1, conv2, conv3, conv4], axis=1)

        else:

            x1 = Reshape((1, -1,))(x1)
            x2 = Reshape((1, -1,))(x2)
            merged = concatenate([x1, x2], axis=1)
            conv = Conv1D(filters=cnn_filters * 2, kernel_size=5, padding='causal', activation='relu')(merged)

            # Deep CNN
            conv = Conv1D(filters=cnn_filters / 2, kernel_size=3, padding='same', activation='relu')(conv)
            conv = Conv1D(filters=cnn_filters / 4, kernel_size=3, padding='same', activation='relu')(conv)

            merged = Flatten()(conv)

        end_point_1 = Dense(self.num_point_relations, activation='softmax')(merged)
        end_point_2 = Dense(self.num_point_relations, activation='softmax')(merged)
        end_point_3 = Dense(self.num_point_relations, activation='softmax')(merged)
        end_point_4 = Dense(self.num_point_relations, activation='softmax')(merged)

        if is_char:
            model = Model(inputs=[event_word_input_1, event_char_input_1, event_word_input_2, event_char_input_2],
                          outputs=[end_point_1, end_point_2, end_point_3, end_point_4])
        else:
            model = Model(inputs=[event_input_1, event_input_2],
                          outputs=[end_point_1, end_point_2, end_point_3, end_point_4])

        print(model.summary())

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model


if __name__ == "__main__":
    word_vector_size = WORD_VECTOR_SIZE
    char_vector_size = CHAR_VECTOR_SIZE
    word_context_length = WORD_CONTEXT_LENGTH
    num_interval_relations = NUM_INTERVAL_RELATIONS
    num_point_relations = NUM_POINT_RELATIONS
    vec_files_path = VEC_FILES_PATH

    m = models(word_context_length, word_vector_size, char_vector_size, num_interval_relations, num_point_relations)
    m.get_interval_relation_classification_model("DEEP_CNN", is_char=True)
