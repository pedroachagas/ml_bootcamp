import requests
host = 'churn-serving-env.eba-fmv7b9qz.sa-east-1.elasticbeanstalk.com.'
url = f'http://{host}/predict'
customer = {'gender': 'female',
 'seniorcitizen': 'no',
 'partner': 'yes',
 'dependents': 'yes',
 'phoneservice': 'no',
 'multiplelines': 'no',
 'internetservice': 'no',
 'onlinesecurity': 'no internet service',
 'onlinebackup': 'no internet service',
 'deviceprotection': 'no internet service',
 'techsupport': 'no internet service',
 'streamingtv': 'no internet service',
 'streamingmovies': 'no internet service',
 'contract': 'two year',
 'paperlessbilling': 'no',
 'paymentmethod': 'mailed check',
 'tenure': 12,
 'monthlycharges': 19.7,
 'totalcharges': 258.35}
r = requests.post(url, json=customer).json()
print(r)