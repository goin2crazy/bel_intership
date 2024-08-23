from config import * 
from picker_model import TargetModel
from collect_data import collect_links, encode_images

import argparse
from tqdm.auto import tqdm

import telebot
import numpy as np
import pandas as pd 
import requests

from io import BytesIO
from PIL import Image

cfg = Config
logs = Logs()

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for running the Telegram bot")
    
    parser.add_argument('--model', type=str, required=True, help="The name of the model to use, e.g., 'gemini'")
    parser.add_argument('--gemini-api', type=str, required=True, help="API key for the Gemini model")
    parser.add_argument('--bot-id', type=str, required=True, help="API key for the Telegram bot")
    
    args = parser.parse_args()
    
    return args.model, args.gemini_api, args.bot_id

def main(picker, model, link): 
    target_image_link = picker.processor.parse_images_from_page(link)
    target_image_link = picker(target_image_link)
    
    return model(target_image_link), target_image_link


def bot_setup(bot_api_key, main_fn, models = None): 
    bot = telebot.TeleBot(bot_api_key)

    
    @bot.message_handler(func=lambda message: True)
    def echo_message(message):
        if message.text.strip().startswith("/extra"): 
            link = message.text.replace("/extra", "").strip() 

            picker= models['picker'] 
            model = models['main']
            
            urls = list() 
            predicted = list() 
            predicted_detail_ids = list() 

            all_links = collect_links(picker, link)
            all_links = list(set(all_links))[:50]
            bot.reply_to(message, f"Found {len(all_links)} pages...")


            bot.reply_to(message, f"Trying to find correct image on {len(all_links)} images")
            
            for l in tqdm(all_links): 
                try: 
                    encoded = encode_images(picker, page_link=l)
                except Exception as e: 
                    print(e)
                    continue
                urls.append(l)
                predicted.append(encoded)

            bot.reply_to(message, f"Done :)")
            bot.reply_to(message, f"Trying to find a detail id...")

            for p in predicted: 
                image_url = "".join([k for k, v in p.items() if v == 1])
                try: 
                    predicted_detail_ids.append(model(image_url))
                except Exception as e: 
                    print(e)
                    predicted_detail_ids.append(None)

            bot.reply_to(message, f"Done :). There is the first: {predicted_detail_ids[0]}")
            dataframe = pd.DataFrame({
                "url": urls, 
                "label_1": [" ".join([k for k, v in p.items() if v==1]) for p in predicted], 
                "label_0": [" ".join([k for k, v in p.items() if v==0]) for p in predicted], 
                "predicted_detail_id": predicted_detail_ids
            })
            # Save DataFrame to Excel
            excel_path = 'output.xlsx'
            dataframe.to_excel(excel_path, index=False)

            # Send the Excel file to chat
            with open(excel_path, 'rb') as file:
                bot.send_document(message.chat.id, file)

            # Optional: Confirmation message
            bot.reply_to(message, "Done :). The file has been sent.")
        else: 
            if message.text.startswith("https://injapan.ru/auction/"):
                bot.reply_to(message, f"Parsing the page...")

                label, image_path = main_fn(message.text)

                responce = f"""Classification result: {label}"""
                # Send the classification result as text
                bot.reply_to(message, responce)

                bot.send_photo(message.chat.id, open(image_path, 'rb'))

            else:
                bot.reply_to(message, f"Wrong links! links have to start with https://injapan.ru/auction/")

    # and here we actually run it
    bot.polling()

if __name__ == "__main__": 
    # Parse important variables
    model_name, gemini_api, bot_id = parse_args() 

    # Initalize models
    assert model_name in ['gemini'], "There no available model you lookin for"

    if model_name == 'gemini': 
        from gemini_model import GeminiInference

        model = GeminiInference(api_key =gemini_api)
    else: 
        model = None 

    picker = TargetModel()

    models = {"picker": picker, "main": model} 
    # Load model
    bot_setup(
        bot_api_key=bot_id, 
        main_fn = lambda link: main(link=link, picker=picker, model=model), 
        models = models,         
        )    
    
