import os
from pathlib import Path

from dotenv import load_dotenv
import uvicorn

root=Path(__file__).resolve().parents[1]
load_dotenv(root/'.env')
uvicorn.run('backend.app.main:app',host=os.getenv('APP_HOST','127.0.0.1'),port=int(os.getenv('APP_PORT','8092')))
