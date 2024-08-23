from picker_model import TargetModel 

import argparse

import json 
import os 
from tqdm import tqdm
# import clear_output 
from IPython.display import clear_output

def collect_links(t, first_page_link, verbose=0) -> list: 
  prodicts_links = list() 
  for i in range(1): 
    i=i+1

    try: 
      main_link = first_page_link.split('page-')[0] + str(i) + first_page_link.split('page-')[-1][1:]
    except: 
      break
    # main_link = f'https://injapan.ru/category/2084017018/currency-USD/mode-1/condition-used/page-{i}/sort-enddate/order-ascending.html'
  
    pages = [f"https://injapan.ru{i}" for _, i in t.processor.get_page_content(main_link) if i.startswith("/auction/")]
  
    # checkout the links the clear output 
    if verbose: 
      print('\n'.join(pages))
      clear_output(wait=True)
  
    prodicts_links.extend(pages)
  return prodicts_links


def encode_images(t, page_link): 
  image_links = t.processor.parse_images_from_page(page_link)
  image_links = list(set(image_links))

  target_link = t.do_inference_minimodel(image_links)

  print(target_link)
  clear_output(wait=True)
  return {
      **{k: 0 for k in [link for link in image_links if link != target_link]},
      **{target_link: 1}, 
  }

def map_fn(t, target_folder, page_link): 

  if not os.path.exists(target_folder):
    os.mkdir(target_folder)

  if os.path.exists(f'{target_folder}/{page_link.split("/")[-1]}.json'):
    return

  predicted_data = encode_images(t, page_link)

  # save predicted_data with name page.split('/')[-1] by json 
  # if it already exists dont save 
  
  with open(f'{target_folder}/{page_link.split("/")[-1]}.json', 'w') as f:
    json.dump(predicted_data, f)


def main(main_page_link, target_folder_name) -> None : 
  t = TargetModel()

  products_links = collect_links(t, main_page_link) 
  
  # remove dupblicates 
  products_links = list(set(products_links))

  for i, page_link in tqdm(enumerate(products_links)):
    print(f'page {i+1}/{len(products_links)}')
    map_fn(t, page_link)

def parse_args():
    """
    Main usage Example: 
    
        python script.py --page-link "https://example.com/page-link" --folder-name "target_folder"
        
    """
  
    parser = argparse.ArgumentParser(description="Arguments for running the image encoding and link collection script")
    
    parser.add_argument('--page-link', type=str, required=True, help="The main page link to start collecting product links from")
    parser.add_argument('--folder-name', type=str, required=True, help="The target folder name where the JSON files will be saved")
    
    args = parser.parse_args()
    
    return args.page_link, args.folder_name

if __name__ == '__main__': 
  page_link, folder_name = parse_args()

  main(page_link, folder_name)
