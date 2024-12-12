import pickle
from picker_model import TargetModel
from dataprocessor import load_data

import tensorflow as tf
from tensorflow.data import Dataset
from tqdm.notebook import tqdm

from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
import keras 

def prepare_data(data):
  image_links = list()
  labels = list()

  for item in data:
    image_links.extend([k for k, v in item.items()])
    labels.extend([v for k, v in item.items()])

  return image_links, labels


def get_data(file_path='/content/correct_answers (3).pkl'): 
  with open(file_path, 'rb') as f:
    data1 = pickle.load(f)
  
  image_links, labels = prepare_data(data1)
  return image_links, labels

def create_dataset(image_links, labels, batch_size=8):
  images = [load_data(l) for l in tqdm(image_links)]

  dataset = Dataset.from_tensor_slices((images,labels))

  dataset = dataset.shuffle(64)
  dataset = dataset.prefetch(64)

  return dataset.batch(batch_size)

def train_model(model, ds, num_epoch = 100): 
  early_stopping = EarlyStopping(monitor='accuracy',
                                 patience=25,
                                 mode='max',
                                 restore_best_weights=True
                                 )
  model.compile(optimizer=AdamW(learning_rate=1e-6, weight_decay=0.001),
                # binary crossentropy
                loss='binary_crossentropy',
                metrics=['accuracy'])
  model.fit(ds, epochs=num_epoch)
  return model

def save_model(model, save_path="checkpoint.weights.h5"): 
  try: 
    model.save_weights(save_path, overwrite=True)
    print("succesfully saved :>")
  except Exception as e: 
    print("Caugh error while saving:", e)
  

def run(data_path, 
        batch_size=8, 
        chunk_size=300, 
        num_epoch=100, 
        save=True, 
       ): 
  img_links, labels = get_data(data_path)

  model = TargetModel().model
  
  for i in range(len(img_links) // chunk_size): 
    chunk_start = i * chunk_size
    chunk_end = i * chunk_size + chunk_size
    
    ds = create_dataset(img_links[chunk_start:chunk_end], 
                        labels[chunk_start:chunk_end], 
                        batch_size = batch_size)
    model = train_model(model, ds, num_epoch)

    if save: 
      save_model(model)
  return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the model with specified parameters.")

    parser.add_argument('--data-path', type=str, help="Path to the dataset file")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--chunk-size', type=int, default=300, help="Chunk size for dividing the dataset")
    parser.add_argument('--num-epoch', type=int, default=100, help="Number of epochs to train the model")
    parser.add_argument('--save', action='store_true', help="Save the model after training")

    args = parser.parse_args()

    model = run(data_path=args.data_path,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                num_epoch=args.num_epoch,
                save=args.save)

# Example run
# python script_name.py /path/to/data.csv --batch_size 16 --chunk_size 500 --num_epoch 50 --save
