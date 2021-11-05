import requests
from requests.exceptions import HTTPError

files = {'image': open('/Users/nicolai/Documents/SolverBackend/images/math-equation.png', 'rb')}
url = 'http://localhost:8080/'

try:
    response = requests.post(url, files=files)
    response.raise_for_status()
    # access JSOn content
    jsonResponse = response.json()
    print("Entire JSON response")
    print(jsonResponse)

except HTTPError as http_err:
    print(f'HTTP error occurred: {http_err}')
except Exception as err:
    print(f'Other error occurred: {err}')