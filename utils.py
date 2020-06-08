import sklearn.linear_model

import sklearn.gaussian_process

import sklearn.isotonic

import numpy

import scipy.stats

import scipy.integrate

import scipy.optimize

import tensorflow

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Dropout

import tensorflow.keras as keras

import sklearn.linear_model

decay_list = [1e-4]

ls_list = [1.0]

eps = numpy.finfo(numpy.random.randn(1).dtype).eps

def get_mdl(x, y, model_class):

    if model_class == 'olr':

        mdl = sklearn.linear_model.LinearRegression()

        mdl.fit(x, y)

        mu = mdl.predict(X=x)

        mdl.sigma = numpy.std(mu - y)

        mdl.model_class = 'olr'

    elif model_class == 'br':

        mdl = sklearn.linear_model.BayesianRidge()

        mdl.fit(x, numpy.squeeze(y))

        mdl.model_class = 'br'

    elif model_class == 'gp':

        k = numpy.std(y) * sklearn.gaussian_process.kernels.RBF(length_scale=1.0) + \
            sklearn.gaussian_process.kernels.WhiteKernel(noise_level=1)

        mdl = sklearn.gaussian_process.GaussianProcessRegressor(normalize_y=False, kernel=k,
                                                                n_restarts_optimizer=32, alpha=1e-2)

        mdl.fit(x, y)

        mdl.model_class = 'gp'

    elif model_class == 'deep':

        tensorflow.compat.v1.disable_eager_execution()

        mdl_list = []

        ll_list = []

        batch_size = int(numpy.shape(x)[0] / 2)

        epochs = int(1e5)

        Nfeat = numpy.shape(x)[1]

        for decay in decay_list:

            L_input = Input(shape=(Nfeat,), name="input")

            h_1 = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(decay),
                        name="dense_1")(L_input)

            h_1_drop = Dropout(0.5)(h_1)

            h_2 = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(decay),
                        name="dense_2")(h_1_drop)

            h_2_drop = Dropout(0.5)(h_2)

            L_out = Dense(1, activation="linear", kernel_regularizer=keras.regularizers.l2(decay),
                          name="output")(h_2_drop)

            mdl = Model(inputs=[L_input], outputs=L_out)

            mdl.summary()

            mdl.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adadelta())

            mdl.fit(x, y, batch_size=batch_size, epochs=epochs,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='loss',
                                                             min_delta=0,
                                                             patience=256,
                                                             verbose=0, mode='auto')])

            mdl.model_class = 'deep'

            mdl.decay = decay

            ll_ls_list = []

            mdl.ls = ls_list[0]

            tmp_mu, tmp_sigma = get_prediction(x, mdl)

            ll_ls_list.append(numpy.sum(scipy.stats.norm.logpdf(y.reshape(-1, 1), loc=tmp_mu, scale=tmp_sigma)))

            mdl.ls = ls_list[numpy.argmax(ll_ls_list)]

            mdl_list.append(mdl)

            ll_list.append(numpy.max(ll_ls_list))

        mdl = mdl_list[numpy.argmax(ll_list)]

        tensorflow.compat.v1.enable_eager_execution()

    else:
        mdl = None

        print('not implemented: ' + model_class)

    return mdl


def get_prediction(x, mdl):

    if mdl.model_class == 'olr':

        mu = mdl.predict(x)

        sigma = numpy.ones(numpy.shape(mu)) * mdl.sigma

    elif mdl.model_class == 'br':

        mu = mdl.predict(x)

        sigma = numpy.ones(numpy.shape(mu)) * numpy.sqrt(1 / mdl.alpha_)

    elif mdl.model_class == 'gp':

        mu, sigma = mdl.predict(x, return_std=True)

    elif mdl.model_class == 'deep':

        tensorflow.compat.v1.disable_eager_execution()

        nb_MC_samples = 128

        MC_output = K.function([mdl.layers[0].input, K.learning_phase()], [mdl.layers[-1].output])

        learning_phase = True

        MC_y_hat = [MC_output([x, learning_phase])[0] for _ in range(nb_MC_samples)]

        MC_y_hat = numpy.array(MC_y_hat)

        MC_y_hat = MC_y_hat[:, :, 0]

        mu = numpy.mean(MC_y_hat, axis=0).reshape(1, -1)

        var = numpy.var(MC_y_hat, axis=0)

        decay = mdl.decay

        ls = mdl.ls

        tau = 0.5 * (ls ** 2) / (2 * numpy.shape(x)[0] * decay)

        sigma = (tau**-1 + var) ** 0.5

        tensorflow.compat.v1.enable_eager_execution()

    mu = numpy.squeeze(mu).reshape(-1, 1)

    sigma = numpy.squeeze(sigma).reshape(-1, 1)

    return mu, sigma


def get_cal_table(y, mu, sigma, t_list, tau_list):

    n_y = numpy.shape(y)[0]

    s = numpy.ones((2, n_y, 3))

    ss = numpy.ones((2, n_y, 3, 3))

    ssc = numpy.ones((n_y, 3, 3))

    sss = numpy.ones((2, n_y, 3, 3, 3))

    sssc = numpy.ones((6, n_y, 3, 3, 3))

    up_idx = numpy.argmax(t_list > y, axis=1)

    low_idx = up_idx - 1

    s[0, :, 0] = scipy.stats.norm.logcdf(t_list[0, low_idx].reshape(-1, 1), loc=mu, scale=sigma).ravel()

    s[0, :, 1] = scipy.stats.norm.logsf(t_list[0, low_idx].reshape(-1, 1), loc=mu, scale=sigma).ravel()

    s[1, :, 0] = scipy.stats.norm.logcdf(t_list[0, up_idx].reshape(-1, 1), loc=mu, scale=sigma).ravel()

    s[1, :, 1] = scipy.stats.norm.logsf(t_list[0, up_idx].reshape(-1, 1), loc=mu, scale=sigma).ravel()

    for i in range(0, 3):
        for j in range(0, 3):
            ss[0, :, i, j] = s[0, :, i] * s[0, :, j]
            ss[1, :, i, j] = s[1, :, i] * s[1, :, j]
            ssc[:, i, j] = s[0, :, i] * s[1, :, j]
            for k in range(0, 3):
                sss[0, :, i, j, k] = ss[0, :, i, j] * s[0, :, k]
                sss[1, :, i, j, k] = ss[1, :, i, j] * s[1, :, k]
                sssc[0, :, i, j, k] = s[0, :, i] * s[0, :, j] * s[1, :, k]
                sssc[1, :, i, j, k] = s[0, :, i] * s[1, :, j] * s[0, :, k]
                sssc[2, :, i, j, k] = s[0, :, i] * s[1, :, j] * s[1, :, k]
                sssc[3, :, i, j, k] = s[1, :, i] * s[0, :, j] * s[0, :, k]
                sssc[4, :, i, j, k] = s[1, :, i] * s[0, :, j] * s[1, :, k]
                sssc[5, :, i, j, k] = s[1, :, i] * s[1, :, j] * s[0, :, k]

    return up_idx, low_idx, s, ss, ssc, sss, sssc


def get_iso_cal_table(y, mu, sigma):

    q_raw = scipy.stats.norm.cdf(y, loc=mu.reshape(-1, 1), scale=sigma.reshape(-1, 1))

    q_list, idx = numpy.unique(q_raw, return_inverse=True)

    q_hat_list = numpy.zeros_like(q_list)

    for i in range(0, len(q_list)):
        q_hat_list[i] = numpy.mean(q_raw <= q_list[i])

    q_hat = q_hat_list[idx]

    return q_raw.ravel(), q_hat.ravel()


def get_beta_cal(q_hat, y, t_list):

    q_y = numpy.zeros(len(y))

    n_y = numpy.shape(q_hat)[0]

    for i in range(0, len(y)):
        t_loc = numpy.argmax(t_list > y[i])
        q_y[i] = q_hat[i, t_loc]

    q_list = numpy.linspace(0, 1, 21)[1:-1]

    q_y = q_y.reshape(-1, 1).repeat(19, axis=1)

    z = numpy.zeros_like(q_y)

    for i in range(0, 19):
        z[:, i] = (q_y[:, i] <= q_list[i])

    z = z.reshape(-1, 1)

    q_y = q_y.reshape(-1, 1)

    s = numpy.hstack([numpy.log(q_y), numpy.log(1-q_y), numpy.ones((numpy.shape(q_y)[0], 1))])

    beta_mdl = sklearn.linear_model.LogisticRegression(C=1e8)

    beta_mdl.fit(s, z.ravel())

    return beta_mdl


def get_q_raw(y, mu, sigma):

    q_raw = scipy.stats.norm.cdf(y, loc=mu.reshape(-1, 1), scale=sigma.reshape(-1, 1))

    return q_raw.ravel()


def get_cal_table_test(mu, sigma, t_list_test):

    n_t = numpy.shape(t_list_test)[1]

    n_y = numpy.shape(mu)[0]

    t = t_list_test.repeat(n_y, axis=1).reshape(-1, 1)

    mu_cal = mu.reshape(1, -1).repeat(n_t, axis=0).reshape(-1, 1)

    sigma_cal = sigma.reshape(1, -1).repeat(n_t, axis=0).reshape(-1, 1)

    ln_s = scipy.stats.norm.logcdf(t, loc=mu_cal, scale=sigma_cal)

    ln_ns = scipy.stats.norm.logsf(t, loc=mu_cal, scale=sigma_cal)

    n = numpy.shape(ln_s)[0]

    s = numpy.hstack([ln_s, ln_ns, numpy.ones([n, 1])])

    return s


def get_norm_q(mu, sigma, t_list):

    q = numpy.zeros([len(mu), len(t_list)])

    s = numpy.zeros([len(mu), len(t_list)])

    for j in range(0, len(t_list)):
        q[:, j] = numpy.squeeze(scipy.stats.norm.cdf(t_list[j], loc=mu, scale=sigma))
        s[:, j] = numpy.squeeze(scipy.stats.norm.pdf(t_list[j], loc=mu, scale=sigma))

    return q, s


def get_density(t_list, q_hat):

    t_list = t_list.ravel()

    n, m = numpy.shape(q_hat)

    diff_t = (t_list[1:] - t_list[:-1]).ravel().reshape(1, -1).repeat(n, axis=0)

    s_hat = numpy.abs(numpy.diff(q_hat, axis=1)) / diff_t

    s_hat[s_hat <= 0] = 1e-256

    return s_hat


def get_density_test(t_list, t_list_test, q_hat, q_default):

    t_list_test = t_list_test.ravel()

    n, m = numpy.shape(q_hat)

    neg_idx = (numpy.sum(numpy.diff(q_hat, axis=1) < 0, axis=1) > 0)

    q_hat[neg_idx, :] = q_default[neg_idx, :]

    flat_idx = (numpy.mean(q_hat, axis=1) >= 0.99)

    q_hat[flat_idx, :] = q_default[flat_idx, :]

    flat_idx = (numpy.mean(q_hat, axis=1) <= 0.01)

    q_hat[flat_idx, :] = q_default[flat_idx, :]

    flat_idx = (q_hat[:, -1] <= 0.99)

    q_hat[flat_idx, :] = q_default[flat_idx, :]

    diff_t = (t_list_test[1:] - t_list_test[:-1]).ravel().reshape(1, -1).repeat(n, axis=0)

    s_hat = numpy.diff(q_hat, axis=1) / diff_t

    return s_hat, q_hat


def get_log_loss(y, t_list, density_hat):

    t_list_hat = (t_list[0:-1] + t_list[1:]) / 2

    ll = numpy.zeros(len(y))

    for i in range(0, len(y)):
        t_loc = numpy.argmin(numpy.abs(y[i] - t_list_hat))
        if density_hat[i, t_loc] <= 0:
            ll[i] = -numpy.log(eps)
        else:
            ll[i] = -numpy.log(density_hat[i, t_loc])

    return ll


def get_y_hat(t_list, density_hat):

    n_y, n_t = numpy.shape(density_hat)

    t_list_hat = (t_list[0:-1] + t_list[1:]) / 2

    y_hat = numpy.zeros(n_y)

    if len(t_list_hat) == n_t:

        for i in range(0, n_y):

            y_py = t_list_hat * density_hat[i, :]

            y_hat[i] = scipy.integrate.trapz(y_py, t_list_hat)

    else:
        for i in range(0, n_y):

            y_py = t_list * density_hat[i, :]

            y_hat[i] = scipy.integrate.trapz(y_py, t_list)

    return y_hat


def get_se(y, y_hat):

    se = (numpy.squeeze(y) - numpy.squeeze(y_hat))**2

    return se


def get_q_y(y, q, t_list):

    q_y = numpy.zeros(len(y))

    for i in range(0, len(y)):
        t_loc = numpy.argmax(t_list > y[i])
        q_y[i] = q[i, t_loc]

    return q_y


def get_cal_error(q_y):

    ce = numpy.zeros(20)

    q_list = numpy.linspace(0, 1, 21)[1:-1]

    q_hat = numpy.zeros_like(q_list)

    for i in range(0, len(q_list)):
        q_hat[i] = numpy.mean(q_y <= q_list[i])

    ce[1:20] = (q_list.ravel() - q_hat.ravel())**2

    ce[0] = numpy.mean(ce[1:20])

    return ce


def get_pin_ball_loss(y, q_hat, t_test):
    """
    Compute pinball loss for quantile levels tau = 0.05, 0.1, ..., 0.95.

    Returns the averaged pinball loss and the pinball loss for every quantile
    level as a concatenated array.
    """
    tau = numpy.linspace(0, 1, 21)[1:-1].reshape(-1, 1)

    # obtain approximate quantiles for all predictions and quantile levels
    t_loc = numpy.argmin(numpy.abs(tau[:, :, numpy.newaxis] - q_hat), axis=2)
    y_hat = t_test[t_loc]

    # compute pinball loss for all quantile levels
    tmp_z = y.ravel() - y_hat
    loss = numpy.mean(numpy.where(tmp_z >= 0, tau, tau-1) * tmp_z, axis=1)

    # compute averaged pinball loss as well
    pbl = numpy.concatenate(([numpy.mean(loss)], loss))

    return pbl


def get_train_ll(y, q_hat, t_list_test):

    n, m = numpy.shape(q_hat)

    t_list_test = t_list_test.ravel()

    diff_t = (t_list_test[1:] - t_list_test[:-1]).ravel().reshape(1, -1).repeat(n, axis=0)

    s_hat = numpy.diff(q_hat, axis=1) / diff_t

    return numpy.mean(get_log_loss(y, t_list_test, s_hat))

