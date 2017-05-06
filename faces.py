import re
import os
import math
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def read_image(file_name):
    """Read PGM file as numpy array."""
    with open(file_name, 'rb') as f:
        buf = f.read()
    header, width, height, max_val = re.search(
        b"(^P5\s(?:\s*#.*[\r\n])*(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buf).groups()
    return np.frombuffer(buf, dtype='u1' if int(max_val) < 256 else '>u2',
                         count=int(width)*int(height), offset=len(header)
                         ).reshape((int(height), int(width)))


def get_images():
    """Get all images in dataset that can be read.
       returns 2D array of [image_name, image_object]"""
    all_images = []
    images = os.listdir('data/')
    all_images += [i for i in images if i[-4:] == '.pgm']
    filtered = []
    # for some reason some of th pgm files can't be read, so filter them out
    for images in all_images:
        try:
            filtered.append([images, read_image('data/{}'.format(images))])
        except:
            continue
    return filtered


def fill_bin(x, y, mo, spaces, bn):
    """Fill the "bins" in cell. This creates the histogram. """
    for i in range(*y):
        for j in range(*x):
            orientation = mo[i][j][1]

            # find closest bin
            closest = 0
            val = abs(orientation - spaces[0])
            for s in range(1, len(spaces)):
                if abs(orientation - spaces[s]) < val:
                    closest = s
                    val = abs(orientation - spaces[s])
            # Accumulate magnitude to closest bin
            bn[closest] += mo[i][j][0]


def compute_HOG_features(image):
    """Generate features necessary for face detection"""
    # Find horizontal (gx) and vertical (gy) gradient of image
    gx, gy = np.gradient(image)

    # Each pixel to be a vector of magnitude and orientation of the gradient
    mo = []
    for y in range(len(gx)):
        # makes it easier to keep it 2D
        mo.append([])
        for x in range(len(gx[y])):
            mo[y].append([
                math.sqrt(math.pow(gx[y][x], 2) + math.pow(gy[y][x], 2)),
                math.atan2(gy[y][x], gx[y][x])])

    # Generate histogram for each cell
    # We will use 12 bins. Our orientaion ranges from [-pi, pi]
    spaces = [-math.pi + ((math.pi * 2)/12) * i for i in range(12)]

    # Split image into 9 cells, here are their ranges in each direction
    csize = image.shape[0] // 3
    ycells = [(0, csize), (csize, csize*2), (csize*2, image.shape[0])]
    xcells = [(0, csize), (csize, csize*2), (csize*2, image.shape[1])]

    cbins = []
    for y in ycells:
        for x in xcells:
            cbins.append([0 for _ in range(12)])
            # Go through each pixel in this cell
            fill_bin(x, y, mo, spaces, cbins[-1])
    del mo

    # Normalization, normalize each vector (histograms)
    cbins = [bn / np.linalg.norm(bn) for bn in cbins]

    return np.array(cbins).flatten()


def get_feature_labels(images, label=0):
    """Images is a 2D array of [image_name, image_object]. From this,
       compute the features and labels for image"""
    # Testing for recognizing individuals
    if label == 0:  # name
        vals = ['megak', 'night', 'glickman', 'cheyer', 'an2i', 'bpm',
                'saavik', 'kk49', 'tammo', 'steffi', 'boland', 'mitchell',
                'sz24', 'danieln', 'karyadi', 'ch4f', 'kawamura', 'phoebe',
                'at33', 'choon']
    elif label == 1:  # pose
        vals = ['left', 'right', 'up', 'straight']
    elif label == 2:  # expression
        vals = ['neutral', 'happy', 'sad', 'angry']
    elif label == 3:  # eyes
        vals = ['open', 'sunglasses']

    labels = [vals.index(i[0].split('_')[label]) for i in images]
    features = [compute_HOG_features(i[1]) for i in images]
    return features, labels


if __name__ == "__main__":
    images = get_images()
    feature, label = get_feature_labels(images)
    xtrain, xtest, ytrain, ytest = train_test_split(feature, label,
                                                    test_size=.5)
    clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
    clf.fit(xtrain, ytrain)
    predicted = clf.predict(xtest)
    print(accuracy_score(ytest, predicted))
