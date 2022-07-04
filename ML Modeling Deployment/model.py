import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# read the csv file

iris_df = pd.read_csv("iris.csv")
print(iris_df.head())

# select independent and dependent variable
x = iris_df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y = iris_df["Species"]

# split the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=58)

# feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# instantiate the model
classifier =  RandomForestClassifier()

# fit the model
classifier.fit(x_train,y_train)

# make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))




