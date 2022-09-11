import requests

url = 'http://ttp://127.0.0.1:5000/predict_api'
r = requests.post(url,json={'SepalLengthCm':1.0, 'SepalWidthCm':2.1, 'PetalLengthCm':1.2,'PetalWidthCm':1.0})

print(r.json())