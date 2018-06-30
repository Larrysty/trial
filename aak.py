from math import sqrt
from skimage.feature import blob_dog
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics
from try1 import classification_model
from sklearn.ensemble import RandomForestClassifier
import glob

def find_blob():
    for img in glob.glob("img/*.jpg"):
            # Read the image
            image_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

            #cv2.imshow('img',image_gray)
            # Gaussian filter
            blobs_dog = blob_dog(image_gray, max_sigma=150, threshold=.5)

            # Calculate the radii
            blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

            # List for blob
            blobs_list = [blobs_dog]
            color = ['lime']
            title = ['Difference of Gaussian']
            sequence = zip(blobs_list, color, title)
            find_parameters(sequence)


def find_parameters(sequence):
    for idx, (blobs, color, title) in enumerate(sequence):
        a = [0]
        b = [0]
        d = [0]
        for blob in blobs:
            y, x, r = blob
            if r not in a:
                a.append(r)
                b.append(x)
                d.append(y)
            maximum_radius = sum(a) / len(a)
            max_rad = max(a)
            location = a.index(max_rad)
            x_coordinate = b[location]
            y_coordinate = d[location]
            area = 3.14 * maximum_radius * maximum_radius
            perimeter = 2 * 3.14 * maximum_radius
            compactness = (perimeter * perimeter) / (area - 1)
            row = str(x_coordinate) + "," + str(y_coordinate) + "," + str(maximum_radius) + "," + str(
                perimeter) + "," + str(area) + "," + str(compactness) + "\n"

            # h=plt.Circle((x_coordinate, y_coordinate), maximum_radius, color=color, linewidth=2, fill=False)
            # plt.close('all')
        # csv.write(row)
        write_into_csv(row)


def write_into_csv(row):
    with open('file.csv', 'a') as newFile:
        newFile.write(row)


def prepare():
    with open('file.csv', 'r') as original: data = original.read()
    with open('file.csv', 'w') as modified: modified.write("x,y,radius,perimeter,area,compactness\n" + data)
    t = pd.read_csv('file.csv')
    mean = t['area'].mean()
    print(mean)
    df = pd.DataFrame(t, columns=['x', 'y', 'radius', 'perimeter', 'area', 'compactness'])
    df['diagnosis'] = np.where(df['area'] > mean, 'M', 'B')
    print(df)
    df.to_csv('file.csv', sep=',', index=False)


def split_data():
    from sklearn.linear_model import LogisticRegression

    data = pd.read_csv('file.csv')
    df=pd.DataFrame(data,columns=['x', 'y', 'radius', 'perimeter', 'area', 'compactness','diagnosis'])

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    data_m = data[['x', 'y', 'radius', 'perimeter', 'area', 'compactness']]
    data_mw = ['x', 'y', 'radius', 'perimeter', 'area', 'compactness']
    predictors = data_m.columns[1:6]
    target = 'diagnosis'

    X = data.loc[:, predictors]
    y = np.ravel(data.loc[:, [target]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    predict(X_train, y_train, X_test, y_test)

    train, test = train_test_split(df, test_size=0.25)
    classification_model(model=LogisticRegression(),data=train,predictors=data_mw,outcome=target)

    heatmap(data)
def predict(X_train, y_train, X_test, y_test):
    # Initiating the model:

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    scores = cross_val_score(rf, X_train, y_train, scoring='accuracy', cv=10).mean()

    print("The mean accuracy with 10 fold cross validation is %s" % round(scores * 100, 2))

    predicted = rf.predict(X_test)

    acc_test = metrics.accuracy_score(y_test, predicted)

    print('The accuracy on test data is %s' % (round(acc_test, 2)))




def heatmap(data1):
    plt.figure(figsize=(14, 14))
    sns.heatmap(data1.corr(), vmax=1, square=True, annot=True)
    plt.show()


if __name__ == "__main__":
    #find_blob()
    #prepare()
    split_data()