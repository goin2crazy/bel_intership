from config import * 

from picker_model import TargetModel

import argparse

import telebot
import numpy as np
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

    target_image_link = picker.parse_images_from_page(link)
    target_image_link = picker(target_image_link)
    
    return model(target_image_link), target_image_link

def bot_setup(bot_api_key, main_fn): 
    bot = telebot.TeleBot(bot_api_key)

    @bot.message_handler(func=lambda message: True)
    def echo_message(message):

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

    # Load model
    bot_setup(
        bot_api_key=bot_id, 
        main_fn = lambda link: main(link=link, picker=picker, model=model), 
        )    
    
