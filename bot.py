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

    parser.add_argument('--gemini-api-model', type=str, default='gemini-1.5-pro', required=False, help="Gemini model u going to use")
    parser.add_argument('--prompt', type=str, default=None, required=False, help="source to txt file write prompt written inside")
    
    args = parser.parse_args()
    
    if (args.prompt) == None: 
        prompt = None
    else: 
        try: 
            print(f"Reading file in {args.prompt}")
            with open(args.prompt, 'r') as f: 
                prompt = f.read() 
                print("Readed Prompt: ", prompt)
        except Exception as e: 
            print(f"Error while reading the {args.prompt}:", e)
            prompt = None 

    return (
        args.model, args.gemini_api, args.bot_id, 
        {
            'gemini_model': args.gemini_api_model, 
            'prompt': prompt
         },)

def main(picker, model, link): 
    target_image_link = picker.processor.parse_images_from_page(link)
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
    model_name, gemini_api, bot_id, addictional_data = parse_args() 

    # Initalize models
    assert model_name in ['gemini'], "There no available model you lookin for"

    if model_name == 'gemini': 
        from gemini_model import GeminiInference

        model = GeminiInference(api_key =gemini_api, model_name =addictional_data['gemini_model'], prompt=addictional_data['prompt'])
    else: 
        model = None 

    picker = TargetModel()

    # Load model
    bot_setup(
        bot_api_key=bot_id, 
        main_fn = lambda link: main(link=link, picker=picker, model=model), 
        )    
    
