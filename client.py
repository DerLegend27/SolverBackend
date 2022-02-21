import requests
from requests.exceptions import HTTPError

files = {'image': open(r'C:\Users\Schlu\OneDrive\Desktop\Finalversion\test-images\math-equation_0001.png', 'rb')}
url = 'http://localhost:5000/api'

try:
    response = requests.post(url, files=files)
    response.raise_for_status()
    jsonResponse = response.json()
    print("Entire JSON response")
    print(jsonResponse)

except HTTPError as http_err:
    print(f'HTTP error occurred: {http_err}')
except Exception as err:
    print(f'Other error occurred: {err}')