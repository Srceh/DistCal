import numpy

import sklearn.datasets


def load_data(dataset, model_class):

    if dataset == 0:

        (x, y) = sklearn.datasets.load_diabetes(return_X_y=True)

        y = numpy.expand_dims(y, axis=1)

    elif dataset == 1:

        (x, y) = sklearn.datasets.load_boston(return_X_y=True)

        y = numpy.expand_dims(y, axis=1)

    elif dataset == 2:

        x, y = load_airfoil()

    elif dataset == 3:

        x, y = load_forest_fire()

    elif dataset == 4:

        x, y = load_strength()

    elif dataset == 5:

        x, y = load_app_energy()

    else:

        raise NotImplementedError('A valid dataset needs to be given.')

    per_idx = numpy.random.permutation(numpy.shape(x)[0])

    x = x[per_idx]

    y = y[per_idx]

    if model_class == 'gp':

        x = x[:1000, :]

        y = y[:1000, 0].reshape(-1, 1)

    return x, y


def filter_nan(raw):

    idx = (numpy.sum(numpy.isnan(raw), axis=1) == 0)

    raw = raw[idx, :]

    return raw


def load_airfoil():

    raw = numpy.loadtxt('./Dataset/Airfoil_Self-Nois.csv', delimiter=',')

    x = raw[:, :-1]

    y = numpy.expand_dims(raw[:, -1], axis=1)

    return x, y


def load_app_energy():

    raw = numpy.genfromtxt('./Dataset/Appliances_energy_prediction.csv', delimiter=',', skip_header=True)

    raw = filter_nan(raw)

    x = raw[:, 1:]

    y = numpy.expand_dims(raw[:, 0], axis=1)

    return x, y


def load_forest_fire():

    raw = numpy.genfromtxt('./Dataset/Forest_Fires.csv', delimiter=',', skip_header=True)

    raw = filter_nan(raw)

    y = numpy.log(numpy.expand_dims(raw[:, -1], axis=1) + 1)

    x = raw[:, :-1]

    return x, y


def load_strength():

    raw = numpy.genfromtxt('./Dataset/Concrete_Compressive_Strength.csv', delimiter=',', skip_header=False)

    raw = filter_nan(raw)

    y = numpy.expand_dims(raw[:, -1], axis=1)

    x = raw[:, :-1]

    return x, y