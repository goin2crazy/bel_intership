from config import * 
from picker_model import build_model
from utils.data import Processor

import numpy as np 

def get_stuff(): 
    model = build_model(1)
    model.load_weights(cfg.model_path)
    
    processor=Processor(cfg.image_size, cfg.batch_size)
    return model, processor

def do_inference_minimodel(processor, model, image_links):
    dataset = processor(image_links)
    predictions = model.predict(dataset)

    predictions = predictions.flatten()
    predictions = np.argmax(predictions, axis=-1)
    return image_links[predictions]

def find_target_image(processor, model, page_link,):
    image_links = processor.parse_images_from_page(page_link)
    return do_inference_minimodel(processor, model, image_links=image_links)

if __name__ == "__main__": 
    model = build_model(1)
    model.load_weights(cfg.model_path)
    