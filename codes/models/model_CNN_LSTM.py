from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Flatten, BatchNormalization, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
import tensorflow.keras

def Model_CNN_LSTM(input_shape, class_num):
    (W,X,Y,Z) = input_shape # 10, 3, 30, 30 
    # Model_Acc #
    Model_Acc_Input = Input(shape=(W, X, Y, Z)) # (batch, steps, channels)

    # Conv block 1
    Model_Acc = (TimeDistributed(Conv2D(32, kernel_size=(3, 3), padding = 'same', activation='relu', kernel_regularizer=regularizers.l2(0.00001), data_format='channels_first')))(Model_Acc_Input)
    Model_Acc = (TimeDistributed(MaxPooling2D(pool_size=(3, 3))))(Model_Acc)
    Model_Acc = (TimeDistributed(BatchNormalization()))(Model_Acc)
    
    # Conv block 2
    Model_Acc = (TimeDistributed(Conv2D(64, kernel_size=(3, 3), padding = 'same', activation='relu', kernel_regularizer=regularizers.l2(0.00001))))(Model_Acc)
    Model_Acc = (TimeDistributed(BatchNormalization()))(Model_Acc)    
    Model_Acc = (TimeDistributed(Conv2D(64, kernel_size=(3, 3), padding = 'same', activation='relu', kernel_regularizer=regularizers.l2(0.00001))))(Model_Acc)
    Model_Acc = (TimeDistributed(BatchNormalization()))(Model_Acc)
    Model_Acc = (TimeDistributed(MaxPooling2D(pool_size=(2, 2))))(Model_Acc)
    Model_Acc = (TimeDistributed(BatchNormalization()))(Model_Acc)
    
    # Conv block 3
    Model_Acc = (TimeDistributed(Conv2D(128, kernel_size=(3, 3), padding = 'same', activation='relu', kernel_regularizer=regularizers.l2(0.00001))))(Model_Acc)
    Model_Acc = (TimeDistributed(BatchNormalization()))(Model_Acc)    
    Model_Acc = (TimeDistributed(Conv2D(128, kernel_size=(3, 3), padding = 'same', activation='relu', kernel_regularizer=regularizers.l2(0.00001))))(Model_Acc)
    Model_Acc = (TimeDistributed(BatchNormalization()))(Model_Acc)
    Model_Acc = (TimeDistributed(MaxPooling2D(pool_size=(2, 2))))(Model_Acc)
    Model_Acc = (TimeDistributed(BatchNormalization()))(Model_Acc)
    
    Model_Acc = (TimeDistributed(Flatten()))(Model_Acc) # output shape -> 512
    
    
    # Model_Ang #
    Model_Ang_Input = Input(shape=(W,X,Y,Z))

    Model_Ang = (TimeDistributed(Conv2D(32, kernel_size=(3, 3), padding = 'same', activation='relu', kernel_regularizer=regularizers.l2(0.00001), data_format='channels_first')))(Model_Ang_Input)
    Model_Ang = (TimeDistributed(MaxPooling2D(pool_size=(3, 3))))(Model_Ang)
    Model_Ang = (TimeDistributed(BatchNormalization()))(Model_Ang)
    
    Model_Ang = (TimeDistributed(Conv2D(64, kernel_size=(3, 3), padding = 'same', activation='relu', kernel_regularizer=regularizers.l2(0.00001))))(Model_Ang)
    Model_Ang = (TimeDistributed(BatchNormalization()))(Model_Ang)    
    Model_Ang = (TimeDistributed(Conv2D(64, kernel_size=(3, 3), padding = 'same', activation='relu', kernel_regularizer=regularizers.l2(0.00001))))(Model_Ang)
    Model_Ang = (TimeDistributed(BatchNormalization()))(Model_Ang)
    Model_Ang = (TimeDistributed(MaxPooling2D(pool_size=(2, 2))))(Model_Ang)
    Model_Ang = (TimeDistributed(BatchNormalization()))(Model_Ang)
    
    Model_Ang = (TimeDistributed(Conv2D(128, kernel_size=(3, 3), padding = 'same', activation='relu', kernel_regularizer=regularizers.l2(0.00001))))(Model_Ang) 
    Model_Ang = (TimeDistributed(BatchNormalization()))(Model_Ang)    
    Model_Ang = (TimeDistributed(Conv2D(128, kernel_size=(3, 3), padding = 'same', activation='relu', kernel_regularizer=regularizers.l2(0.00001))))(Model_Ang)
    Model_Ang = (TimeDistributed(BatchNormalization()))(Model_Ang)
    Model_Ang = (TimeDistributed(MaxPooling2D(pool_size=(2, 2))))(Model_Ang)
    Model_Ang = (TimeDistributed(BatchNormalization()))(Model_Ang) ## Trainable
    
    Model_Ang = (TimeDistributed(Flatten()))(Model_Ang)
    
    
    ## Merge ##
    Merge = concatenate([Model_Acc, Model_Ang])
    
    
    # Weight Classifier #
    Model_Weight = (LSTM(512))(Merge) ## Trainable
    Model_Weight = (BatchNormalization())(Model_Weight) ## Trainable
    Model_Weight = (Dense(class_num, activation='softmax', name='Model_Weight'))(Model_Weight) ## Trainable

    # Final Model #
    Model_Final = Model(inputs=[Model_Acc_Input, Model_Ang_Input], outputs=[Model_Weight])
    Model_Final.summary()
    Model_Final.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.00001), loss={'Model_Weight': 'categorical_crossentropy'}, metrics=['accuracy'])

    return (Model_Final)