from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any

def create_message(subject: str, body: str, from_addr: str, to_addr: str) -> MIMEMultipart:
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg.attach(MIMEText(body, 'plain'))
    return msg

def format_detection_body(detection_data: Dict[str, Any]) -> str:
    parts = []
    for k, v in detection_data.items():
        parts.append(f"{k}: {v}")
    return "\n".join(parts)
