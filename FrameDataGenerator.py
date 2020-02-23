import numpy as np
import keras
import os
import cv2
import config

class FrameDataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size=config.BATCH_SIZE, img_size=(config.IMAGE_HEIGHT,config.IMAGE_WIDTH), ratio=1, shuffle=True):
        self.base_path = config.DATA_PATH
        self.get_dataset()

        self.batch_size = batch_size
        self.img_size = img_size
        self.ratio = ratio
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_dataset) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_batch_img_paths = [self.list_dataset[k] for k in indexes]
        # # Generate data
        X, y = self.__data_generation(list_batch_img_paths)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_dataset))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_batch_img_paths):
        resize_shape = (int(self.img_size[0]/self.ratio), int(self.img_size[1]/self.ratio))

        # Initialization
        X = np.empty((self.batch_size, *resize_shape, 3))
        y = np.empty((self.batch_size, *self.img_size, 3))

        # Generate data
        for i, img_file_path in enumerate(list_batch_img_paths):
            img = cv2.imread(self.base_path + "/" +img_file_path)
            X[i] = cv2.resize(img, resize_shape)
            y[i] = cv2.resize(img, self.img_size)

        return X, y

    def get_dataset(self) :
        self.list_dataset = []
        files = os.listdir(self.base_path)
        for file in files :
            if (file.endswith(".jpg")) :
                self.list_dataset.append(file)

# frame_data_generator = FrameDataGenerator()
# X, y = frame_data_generator.__getitem__(0)
# print("X.shape",X.shape)


