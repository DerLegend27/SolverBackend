import requests

url = 'http://localhost:8080/'
files = {'data': open('test.jpeg', 'rb')}
requests.post(url, files=files)