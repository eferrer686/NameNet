import pandas as pd
from sklearn import preprocessing


# Read SUR_NAMES DB
db = pd.read_csv("data/Common_surnames.csv")

db = db.loc[:,['name','prop100k']]

print(db)

prop = db.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(db['prop100k'])
db['prop100k'] = x_scaled