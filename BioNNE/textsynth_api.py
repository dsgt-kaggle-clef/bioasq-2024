import sys
import requests
import json
import time
import datetime
from tokens import * 

URL = 'https://api.textsynth.com'
def make_textsynth_request(path, query_param, llm_log_file=None):
    response = requests.post(URL + path,
        headers = { "Authorization": f"Bearer {TEXTSYNTH_TOKEN}"},
        json = query_param)
    if llm_log_file:
        llm_log_file.write(f"\n# {datetime.datetime.now()}\n")
        llm_log_file.write(f'{URL + path}\n')
        json.dump(query_param, llm_log_file, indent=2)
        llm_log_file.write('\nResponse\n')
        json.dump(response.json(), llm_log_file, indent=2)
    if response.status_code != 200:
        print("Request error:", response.text)
    return response.json()
