rom keras.layers import Convolution2D
from keras.layers import  MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
def mdl():
    epoch=25
    model=Sequential()
    model.add(Convolution2D(filters=32,kernel_size=(3,3),strides=(1,1),activation="relu", input_shape=(64,64,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=525,activation="relu"))
    model.add(Dense(units=250,activation="relu"))
    model.add(Dense(units=1,activation="sigmoid"))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    from keras_preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_set = train_datagen.flow_from_directory('/root/cnndog/cnn_dataset/training_set/',target_size=(64,64),class_mode='binary')
    test_set = test_datagen.flow_from_directory('/root/cnndog/cnn_dataset/test_set/',target_size=(64, 64),class_mode='binary')
    r=model.fit(train_set,epochs=epoch,validation_data=test_set)
    accu=r.history["accuracy"][epoch-1]*100
    a=int(accu)
    with open('/root/data/accu.txt',"w" ) as f:
        data = str(a)
	f.write(data)
mdl()