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
### 1. Audi and Volkswagen (VAG Group)
- Number format: Typically, the part number is 9 or 11 characters long, divided into groups of 3, 3, and 3/5 characters (e.g. **1K0 615 601 A**).
- Number structure:
- First group: The first 3 characters often identify the model or type of vehicle (e.g. 1K0 for Golf 5).
- Second group: The second group of characters identifies the part category (e.g. 615 might identify brake components).
- Third group: The last 3-5 characters usually identify the specific part and its version or modification.
- Special features: Porsche, as part of VAG, often has numbers similar to the Audi/Volkswagen system, but may sometimes include unique prefixes, such as 95B for Macan models.

### 2. Mercedes-Benz
- Number format: Typically, part numbers have a 10-character format, divided into 3 groups (e.g. **A 204 820 11 02**).
- Number structure:
- First group: Often indicates the part category or series (e.g. 204 for the C-Class W204).
- Second group: The second group indicates the part itself (e.g. 820 may indicate the electrical system).
- Third group: The last group indicates the part modification.
- Prefixes: Often, numbers start with the letter A, which indicates that this is a genuine Mercedes-Benz part.

### 3. BMW
- Number format: BMW part numbers consist of 11 digits with no spaces or other separators (e.g. **11 12 7 838 797**).
- Number structure:
- First group (first 2 digits): Indicates the main category of the part (e.g. 11 - engine).
- Second group (next 2 digits): Indicates the subcategory (e.g. 12 - cooling system).
- Third group (remaining 7 digits): Unique part code.
- Special features: Some parts may have letters at the end of the number indicating the version or revision.

### 4. Porsche
- Number format: Porsche part numbers are 9 or 12 characters long and can be divided into 3 groups (e.g. **955 351 503 10**).
- Number structure:
- First group: The first 3 digits usually indicate the model (e.g. 955 for Cayenne).
- Second group: Indicates the system or category (e.g. 351 for the brake system).
- Third group: Unique part number and modification.
- Features: In some cases, suffixes are used to indicate color or material (e.g. C for black).

### General tips:
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