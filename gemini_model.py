# ! pip install -q google-generativeai

import google.generativeai as genai
from pathlib import Path
from time import sleep

from PIL import Image
import requests

class GeminiInference():
  def __init__(self, api_key,  model_name = 'gemini-1.5-flash'):
    self.gemini_key = api_key

    genai.configure(api_key=self.gemini_key)
    generation_config = {
        "temperature": 1,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
    }
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    self.model = genai.GenerativeModel(model_name=model_name,
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)


  def get_response(self, img):
    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": img.read_bytes()
        },
    ]
    prompt_parts = [
        image_parts[0],
        """ My grandma has problems with vision, she works in japan machine details fabric, can you please her to figure out the detail main id in this images?

There is some hints that my grandma uses for recognize detail number:
"
For Example number have to looks like 1A2 345 678 B. The target number id have to include the 4 parts, first three parts is for example "1A2 345 678" (each includes 3 chars) and last part, four part, can include in range from 1 char to 3 char, in this example last part is "B"

General:
- Remember the main prefixes: For example, 1K0 for Volkswagen Golf, A for Mercedes-Benz, 95B for Porsche Macan.
- Pay attention to letter suffixes: They can indicate different modifications of the same part."

Please think about recognizing the detail number id step by step
Write total detail number at the end
Please Write the <START> to beginning of number, and <END> to detail number end:
Your Answer Example have to look like:

"
{reasoning about correct detail number}

<START> 1A2 345 678 B <END>
"
If you dont any number on image just write <START> None <END>
        """,
    ]
    response = self.model.generate_content(prompt_parts)
    return response.text

  def extract_number(self, response):
    try:
      return response.split('<START>')[1].split('<END>')[0]
    except:
      return 'None'

  def __call__(self, image_path):

      # Validate that an image is present
      if image_path.startswith('http'):
        # read remote img bites
        img = Image.open(requests.get(image_path, stream=True).raw)
        # save image to local "example_image.jpg"
        img.save("example_image.jpg")
        image_path = "example_image.jpg"

      if not (img := Path(image_path)).exists():
        raise FileNotFoundError(f"Could not find image: {img}")

      answer = self.get_response(img)
      return self.extract_number(answer)
