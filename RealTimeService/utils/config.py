from dotenv import load_dotenv
import os

load_dotenv()

def get_env(key: str, default=None):
    return os.getenv(key, default)

# Common email configuration constants (read from environment)
EMAIL_FROM = os.getenv('EMAIL_FROM')
EMAIL_TO = os.getenv('EMAIL_TO')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
