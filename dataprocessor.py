from config import Config as cfg 
from config import RuntimeMeta

import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tqdm.notebook import tqdm

import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup

def load_image(image_link):
    if type(image_link) == str:
      if image_link.startswith("http"):
          try:
              headers = {
                  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
              }

              response = requests.get(image_link, headers = headers)
              # response = requests.get(image_link
              img = Image.open(BytesIO(response.content))
          except Exception as e:
              print(image_link)
              print(e)
              return None
      else:
          img = Image.open(image_link)
    elif type(image_link) == np.ndarray:
      img = Image.fromarray(image_link)
    else:
      raise Exception("Unknown image type")

    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def encode_image(img):
    img = img.resize(cfg.image_size)
    img = np.array(img)
    img = img.astype('float32')
    img /= 255.0
    return img

def load_data(image_link):
    img = load_image(image_link)
    if img is None:
        return None
    img = encode_image(img)

    # convert data to tf.tensor
    img = tf.convert_to_tensor(img)
    return img


class Processor(metaclass=RuntimeMeta):
    def __init__(self, image_size, batch_size):
        self.image_size = image_size
        self.batch_size = batch_size

    def get_page_content(self, url, verbose = 0 ):
    # Send a request to fetch the webpage content
      response = requests.get(url)

      # Check if the request was successful
      if response.status_code == 200:
          # Parse the content with BeautifulSoup
          soup = BeautifulSoup(response.content, 'html.parser')

          # Find all image tags
          images = soup.find_all('img')

          # Loop through each image and find its parent link if available
          for img in images:
              # Check if the image is inside a link tag
              parent_a = img.find_parent('a')
              if parent_a and parent_a.has_attr('href'):
                  # print(f'Image src: {img.get("src")}')
                  # print(f'Link: {parent_a.get("href")}')
                  yield img.get("src"), parent_a.get("href")
              else:
                  if verbose:
                    print(f'Image src: {img.get("src")} has no associated link.')

                  yield img.get("src"), 'None'
      else:
          print(f'Failed to retrieve the webpage. Status code: {response.status_code}')

    def load_product_info(self, url): 
    # Send a request to fetch the webpage content
      response = requests.get(url)

      # Check if the request was successful
      if response.status_code == 200:
          # Parse the content with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
        
            return_data = {} 
            return_data['price'] = soup.find('span', {'id': 'spanInfoPrice'}).text
            return return_data
      else:
            print(f'Failed to retrieve the webpage. Status code: {response.status_code}')

    def build_dataset(self, image_links):
        images = [load_data(image_link) for image_link in tqdm(image_links)]
        images = [img for img in images if img is not None]

        dataset = Dataset.from_tensor_slices(images)
        try:
            next(iter(dataset))
        except Exception as e:
            print(e)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def __call__(self, *args, **kwargs):
        return self.build_dataset(*args, **kwargs)

    def parse_images_from_page(self, page_url):
      image_links = [i for i, _ in  self.get_page_content(page_url) if type(i)==str and (i.startswith("https://auctions.c.yimg.jp/images.auctions.yahoo.co.jp/image/"))]

      return list(set(image_links))

    def take_newest(self, idx=10, *args, **kwargs):
      pages = [f"https://injapan.ru{i}" for _, i in self.get_page_content(cfg.mainpage_url) if i.startswith("/auction/")]

      page_url = pages[idx]
      return self.parse_images_from_page(page_url)
