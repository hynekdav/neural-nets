from keras.callbacks import Callback
import numpy as np
from preprocessing.loading import DataLoader


class FlippingCallback(Callback):
    def __init__(self, x_data, y_data, num_objects, num_epochs):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.num_objects = num_objects
        self.num_epochs = num_epochs

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.x_data)
        flipped_train_y = np.array(self.y_data)

        for sample, (pred_bboxes, exp_bboxes) in enumerate(zip(preds, flipped_train_y)):
            pred_bboxes = pred_bboxes.reshape(self.num_objects, -1)
            exp_bboxes = exp_bboxes.reshape(self.num_objects, -1)

            mses = np.zeros((self.num_objects, self.num_objects))
            for i, exp_bbox in enumerate(exp_bboxes):
                for j, pred_bbox in enumerate(pred_bboxes):
                    mses[i, j] = np.mean(np.square(exp_bbox - pred_bbox))

            new_order = np.zeros(self.num_objects, dtype=int)

            for i in range(self.num_objects):
                ind_exp_bbox, ind_pred_bbox = np.unravel_index(mses.argmin(), mses.shape)
                mses[ind_exp_bbox] = np.inf
                mses[:, ind_pred_bbox] = np.inf
                new_order[ind_pred_bbox] = ind_exp_bbox

            self.y_data[sample] = exp_bboxes[new_order].flatten()


class ArchitectureTrainer(object):
    def __init__(self, model, num_objects=1):
        self.model = model
        self.num_objects = num_objects

    def train(self, training_data_folder, output_model_save_path, validation_split=0.1):
        images, labels = DataLoader.load_data(training_data_folder, self.num_objects)

        i = int((1 - validation_split) * len(images))
        train_imgs = images[:i]
        test_imgs = images[i:]
        train_boxes = labels[:i]
        test_boxes = labels[i:]

        epochs = 50

        from keras.callbacks import EarlyStopping
        self.model.fit(train_imgs, train_boxes, epochs=epochs, validation_data=(test_imgs, test_boxes), batch_size=8,
                       callbacks=[EarlyStopping(patience=3),
                                  FlippingCallback(train_imgs, train_boxes, self.num_objects, epochs),
                                  FlippingCallback(test_imgs, test_boxes, self.num_objects, epochs)],
                       verbose=2)

        self.model.save(output_model_save_path)
