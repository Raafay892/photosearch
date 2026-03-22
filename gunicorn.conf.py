import os

bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
workers = 1
threads = 1
timeout = 180
