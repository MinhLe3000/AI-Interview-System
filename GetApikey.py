import os
from dotenv import load_dotenv

def loadapi():

    # Load biến môi trường từ file .env
    load_dotenv()

    API_KEY = os.getenv("GOOGLE_API_KEY")
    return API_KEY