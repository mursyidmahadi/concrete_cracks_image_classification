# %% Import
from module import dataset_load, model_pred, show_predictions
from keras.models import load_model
import os

MODEL_PATH = os.path.join(os.getcwd(), 'model', 'no_aug_model.h5')
loaded_model = load_model(MODEL_PATH)

DATASET_PATH = os.path.join(os.getcwd(), 'dataset')
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# %% Data Loading
dataset, pf_train, pf_val, pf_test = dataset_load(DATASET_PATH, BATCH_SIZE, IMG_SIZE)
class_names = dataset.class_names

# %% Model Prediction
image_batches, label_batches, y_pred_batches = model_pred(loaded_model, pf_test)

# %%
show_predictions(class_names, image_batches, label_batches, y_pred_batches)

# %%
