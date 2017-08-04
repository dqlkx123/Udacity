import numpy as np
import pandas as pd
from IPython.display import display


def main():

    #load file to data
    fileName = "titanic_data.csv"
    data = pd.read_csv(fileName)

    #add outcome
    outcome = data['Survived']

    #drop survival column from data
    data = data.drop('Survived',axis=1)

    accuracy(outcome, prediction(data))


    # print(data.head())


def accuracy(truth, pred):
    if len(truth) != len(pred):
        print "truth and pred don't match"

    else:
        print "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean() * 100)

def prediction(data):
    prediction = []
    for _,passenger in data.iterrows():
        if passenger['Sex'] == 'female':
            prediction.append(1)
        elif passenger['Age'] <= 10:
            prediction.append(1)
        else:
            prediction.append(0)

    return pd.Series(prediction)


if __name__ == '__main__':

    main()

