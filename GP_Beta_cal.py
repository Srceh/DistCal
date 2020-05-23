import numpy

import scipy.stats

import tensorflow as tf

import tensorflow.keras.optimizers as optimizers

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot

tf.compat.v1.enable_eager_execution()

numpy.set_printoptions(precision=2)


class GP_Beta:

    def __init__(self, length_scale=None, std=None,
                 omega=None, kappa=None, jitter=1e-2):

        self.jitter = jitter

        self.n = None

        self.y = None

        self.mu = None

        self.sigma = None

        self.q = None

        self.s = None

        self.ln_q = None

        self.ln_1_q = None

        self.ln_s = None

        self.B = None

        self.K_p = None

        self.K_y = None

        self.C = None

        self.theta = None

        self.mu_w_test = None

        self.cov_w_test = None

        self.gradient = None

        self.NLL = None

        self.A = None

        self.A_u = None

        self.L_u = None

        self.mu_u = None

        self.sigma_u = None

        self.n_u = None

        self.C_u = None

        self.C_u_inv = None

        self.mu_shift = None

        if length_scale is None:
            self.length_scale = 1
        else:
            self.length_scale = length_scale

        if std is None:
            self.std = 1
        else:
            self.std = std

        if omega is None:
            self.omega = numpy.random.randn(3)
        else:
            self.omega = omega

        if kappa is None:
            self.kappa = numpy.ones(3)
        else:
            self.kappa = kappa

    def fit(self, y, mu, sigma, n_u, lr=1e-3, optimizer_choice='adam'):

        self.n = numpy.shape(y)[0]

        self.y = y

        self.mu = mu

        self.sigma = sigma

        self.q = scipy.stats.norm.cdf(y, loc=mu, scale=sigma)

        self.s = scipy.stats.norm.pdf(y, loc=mu, scale=sigma)

        self.ln_q = scipy.stats.norm.logcdf(y, loc=mu, scale=sigma)

        self.ln_1_q = scipy.stats.norm.logsf(y, loc=mu, scale=sigma)

        self.ln_s = scipy.stats.norm.logpdf(y, loc=mu, scale=sigma)

        c_theta = numpy.ones(8)

        c_theta[1] = 1.0

        c_theta[2:5] = numpy.random.randn(3)

        c_theta[5:] = 1.0

        self.n_u = n_u

        self.mu_u = (numpy.random.choice(mu.ravel(), self.n_u, replace=False)).reshape(-1, 1) + numpy.random.randn(self.n_u, 1)

        # self.sigma_u = numpy.abs(numpy.random.choice(sigma.ravel(), self.n_u, replace=False)).reshape(-1, 1)

        self.sigma_u = numpy.ones((self.n_u, 1))
        
        C_u = kernel(c_theta, self.mu_u, self.sigma_u, tf.constant(self.jitter, dtype='float64'))

        A_u = numpy.random.randn(self.n_u, 3)

        L_u = scipy.linalg.cholesky(C_u)

        mu_shift = numpy.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) 

        theta = numpy.hstack([c_theta.ravel(), A_u.ravel(), L_u.ravel(),
                              self.mu_u.ravel(), self.sigma_u.ravel(), mu_shift])

        theta = parameter_update(theta, self.q, self.ln_q, self.ln_1_q, self.ln_s, self.mu, self.sigma,
                                 self.n_u, self.n, self.jitter, lr=lr, plot_loss=True, optimizer_choice=optimizer_choice)

        c_theta = theta[:8]

        u = theta[8:]

        self.length_scale = c_theta[0]

        self.std = c_theta[1]

        self.omega = c_theta[2:5].reshape(3, 1)

        self.kappa = c_theta[5:8].reshape(3, 1)

        self.A_u = u[:3*self.n_u].reshape(self.n_u, 3)

        self.L_u = u[3*self.n_u:3*self.n_u+9*self.n_u**2].reshape(3*self.n_u, 3*self.n_u)

        self.mu_u = u[3*self.n_u+9*self.n_u**2:3*self.n_u+9*self.n_u**2+self.n_u].reshape(-1, 1)

        self.sigma_u = u[3*self.n_u+9*self.n_u**2+self.n_u:-6].reshape(-1, 1)

        self.mu_shift = u[-6:]

        self.theta = c_theta

        self.B = coregion(self.omega, self.kappa).numpy()

        C_u = kernel(c_theta, self.mu_u, self.sigma_u, 
                    jitter=tf.constant(self.jitter, dtype='float64')).numpy()

        C_inv_u = numpy.linalg.inv(C_u)

        self.C_u = C_u

        self.C_u_inv = C_inv_u

    def predict(self, t_test, mu_test, sigma_test):

        if self.mu_w_test is None:
            mu_w_test, cov_w_test = predict_new_w(self, mu_test, sigma_test)

            self.mu_w_test = mu_w_test.copy()

            self.cov_w_test = cov_w_test.copy()

        s_hat, q_hat = get_calibration(t_test, mu_test, sigma_test, self.mu_w_test, self.cov_w_test, self.mu_shift)

        return s_hat, q_hat


def predict_new_w(mdl, mu_test, sigma_test):

    y_train = mdl.y

    n_y = numpy.shape(y_train)[0]

    n_test = numpy.shape(mu_test)[0]

    mu_w_hat = numpy.zeros((n_test, 3))

    cov_w_hat = numpy.zeros((n_test, 3, 3))

    theta = mdl.theta

    mu_train = mdl.mu_u

    C_u = mdl.C_u

    C_u_inv = mdl.C_u_inv

    sigma_train = mdl.sigma_u

    C_test = kernel_test(theta, mu_test, sigma_test, mu_train, sigma_train).numpy()

    C_upper = C_test[:3*mdl.n_u, :]

    C_lower = C_test[3*mdl.n_u:3*mdl.n_u+3, :]

    V_u = numpy.matmul(mdl.L_u.transpose(), mdl.L_u)

    D_u = V_u - C_u

    for j in range(0, n_test):

        C_wu = C_upper[:, j*3:(j+1)*3].transpose()

        C_diag_w = C_lower[:, j*3:(j+1)*3]

        T_wu = numpy.matmul(C_wu, C_u_inv)

        mu_w_hat[j, :] = numpy.matmul(C_wu, mdl.A_u.reshape(-1, 1)).ravel()

        cov_w_hat[j, :, :] = (C_diag_w + numpy.matmul(numpy.matmul(T_wu, D_u), T_wu.transpose()))

    return mu_w_hat, cov_w_hat


def get_calibration(t_test, mu_test, sigma_test, mu_w, cov_w, mu_shift):

    n_y = numpy.shape(mu_test)[0]

    n_t = numpy.shape(t_test)[1]

    q_hat = numpy.zeros((n_y, n_t))

    s_hat = numpy.zeros((n_y, n_t))

    for i in range(0, n_y):

        ln_s = scipy.stats.norm.logpdf(x=t_test, loc=mu_test[i, :], scale=sigma_test[i, :]).reshape(-1, 1)

        feature_q = numpy.hstack([scipy.stats.norm.logcdf(x=t_test, loc=mu_test[i, :],
                                                          scale=sigma_test[i, :]).reshape(-1, 1),
                                  scipy.stats.norm.logsf(x=t_test, loc=mu_test[i, :],
                                                         scale=sigma_test[i, :]).reshape(-1, 1),
                                  numpy.ones((n_t, 1))])

        w_sample = scipy.stats.multivariate_normal.rvs(size=4096, mean=mu_w[i, :], cov=cov_w[i, :, :])

        w_sample[:, 0] = -numpy.exp(w_sample[:, 0] / mu_shift[0] + mu_shift[1])

        w_sample[:, 1] = numpy.exp(w_sample[:, 1] / mu_shift[2] + mu_shift[3])

        w_sample[:, 2] = w_sample[:, 2] / mu_shift[4] + mu_shift[5]

        raw_prod = numpy.matmul(feature_q, w_sample.transpose())

        MAX = raw_prod.copy()

        MAX[MAX < 0] = 0

        q_hat[i, :] = numpy.mean(numpy.exp(-MAX) / (numpy.exp(-MAX) + numpy.exp(raw_prod - MAX)), axis=1).ravel()

        tmp_de = numpy.where(raw_prod <= 0,
                             2 * numpy.log(1 + numpy.exp(raw_prod)),
                             2 * (raw_prod + numpy.log(1 + 1 / numpy.exp(raw_prod))))

        ln_s_hat = (raw_prod + numpy.log((w_sample[:, 0] + w_sample[:, 1]) * numpy.exp(feature_q[:, 0].reshape(-1, 1)) -
                                         w_sample[:, 0]) - feature_q[:, 0].reshape(-1, 1) -
                    feature_q[:, 1].reshape(-1, 1) - tmp_de) + ln_s

        mc_s_hat = numpy.exp(ln_s_hat)

        mc_s_hat[numpy.isnan(mc_s_hat)] = 0

        mc_s_hat[numpy.isinf(mc_s_hat)] = 0

        s_hat[i, :] = numpy.mean(mc_s_hat, axis=1).ravel()

    return s_hat, q_hat


def mc_link_lik(w, mu_shift, q, ln_q, ln_1_q, ln_s):

    w_a = w[:, 0::3]

    w_b = w[:, 1::3]

    a = -tf.math.exp(w_a / mu_shift[0] + mu_shift[1])

    b = tf.math.exp(w_b / mu_shift[2] + mu_shift[3])

    c = w[:, 2::3] / mu_shift[4] + mu_shift[5]

    tmp_sum = a * tf.squeeze(ln_q) + b * tf.squeeze(ln_1_q) + c

    tmp_de = tf.where(tmp_sum <= 0,
                      2 * tf.math.log(1 + tf.math.exp(tmp_sum)),
                      2 * (tmp_sum + tf.math.log(1 + 1 / (tf.math.exp(tmp_sum)))))

    ln_s_hat = (tmp_sum + tf.math.log((a + b) * tf.squeeze(q) - a) -
                tf.squeeze(ln_q) - tf.squeeze(ln_1_q) - tmp_de) + tf.squeeze(ln_s)

    mean_exp = tf.reduce_mean(tf.math.exp(ln_s_hat), axis=0)

    ln_mean_s_hat = tf.where(mean_exp > 0, tf.math.log(mean_exp),
                             tf.math.log(tf.zeros_like(mean_exp, dtype='float64') +
                                         tf.constant(1e-16, dtype='float64')))

    link_ll = tf.reduce_sum(ln_mean_s_hat)

    return link_ll


def get_sample_w_step(A_u, Q_w, C_wu, raw_sample_w):

    n_u = int(len(A_u)/3)

    A_u = tf.reshape(A_u, [-1, 1])

    Q_w = tf.reshape(Q_w, [3, 3])

    C_wu = tf.reshape(C_wu, [-1, 3 * n_u])

    C_wu = tf.reshape(C_wu, [-1, 3 * n_u])

    mu_w = tf.reshape(tf.linalg.matmul(C_wu, A_u), [1, -1])

    return tf.squeeze(tf.reshape((tf.linalg.matmul(raw_sample_w, tf.linalg.cholesky(Q_w)) + mu_w), [1, -1]))


def get_Q_w(L_u, C_u, C_wu, C_diag_w, n_u):

    C_u = tf.reshape(C_u, [3*n_u, 3*n_u])

    C_wu = tf.reshape(C_wu, [-1, 3*n_u])

    C_diag_w = tf.reshape(C_diag_w, [3, 3])

    T_wu = tf.linalg.matmul(C_wu, tf.linalg.inv(C_u))

    L_u_hat = tf.reshape(L_u, [3*n_u, 3*n_u])

    V_u = tf.linalg.matmul(tf.transpose(L_u_hat), L_u_hat)

    D_u = V_u - C_u

    return tf.squeeze(tf.reshape((C_diag_w + tf.linalg.matmul(tf.linalg.matmul(T_wu, D_u), tf.transpose(T_wu))),
                                 [1, -1]))


def get_sample_w(u, C_u, C_wu, C_diag_w, raw_sample_w, n_u, n_y):

    A_u = u[:3*n_u]

    L_u = u[3*n_u:]

    sample_w_list = []

    Q_w = []

    for i in range(0, n_y):

        Q_w.append(get_Q_w(L_u, C_u, C_wu[i*(9*n_u):(i+1)*(9*n_u)], C_diag_w[i*9:(i+1)*9], n_u))

        sample_w_list.append(tf.reshape(get_sample_w_step(A_u, tf.reshape(Q_w[-1], [1, -1]),
                                                          C_wu[i*(9*n_u):(i+1)*(9*n_u)],
                                                          tf.reshape(raw_sample_w[:, i*3:(i+1)*3], [-1, 3])), [-1, 3]))

    return tf.concat(sample_w_list, axis=1)


def get_noraml_kl(u, C_u, n_u):

    C_u = tf.reshape(C_u, [3*n_u, 3*n_u])

    A_u = tf.reshape(u[:3*n_u], [-1, 1])

    mu_u = tf.squeeze(tf.reshape(tf.linalg.matmul(C_u, A_u), [1, -1]))

    L_u = tf.reshape(u[3*n_u:], [3*n_u, 3*n_u])

    V_u = tf.linalg.matmul(tf.transpose(L_u), L_u)

    kl = -0.5 * tf.linalg.slogdet(V_u)[1] + \
        0.5 * tf.linalg.slogdet(C_u)[1] + \
        0.5 * tf.linalg.matmul(tf.reshape(mu_u, [1, -1]), A_u) + \
        0.5 * tf.linalg.trace(tf.linalg.matmul(tf.linalg.inv(C_u), V_u))

    return tf.squeeze(kl)


def vi_obj_link_only(theta, q, ln_q, ln_1_q, ln_s, mu, sigma,
           n_u, n_y, raw_sample_w, jitter):

    q = tf.convert_to_tensor(q)

    ln_q = tf.convert_to_tensor(ln_q)

    ln_1_q = tf.convert_to_tensor(ln_1_q)

    ln_s = tf.convert_to_tensor(ln_s)

    mu = tf.convert_to_tensor(mu)

    sigma = tf.convert_to_tensor(sigma)

    c_theta = theta[:8]

    u = theta[8:8+3*n_u+9*n_u**2]

    mu_u = tf.reshape(theta[8+3*n_u+9*n_u**2:8+3*n_u+9*n_u**2+n_u], [-1, 1])

    sigma_u = tf.reshape(theta[8+3*n_u+9*n_u**2+n_u:-6], [-1, 1])

    C_u = kernel(c_theta, mu_u, sigma_u, jitter=tf.constant(jitter, dtype='float64'))

    C_wu = kernel_test(c_theta, mu, sigma, mu_u, sigma_u)

    C_wu = tf.transpose(C_wu[:3*n_u, :])

    C_diag_w = kernel_diag(c_theta, mu, sigma, jitter=tf.constant(jitter, dtype='float64'))

    sample_w = \
        get_sample_w(u,
                     tf.squeeze(tf.reshape(C_u, [1, -1])),
                     tf.squeeze(tf.reshape(C_wu, [1, -1])),
                     tf.squeeze(tf.reshape(C_diag_w, [1, -1])),
                     raw_sample_w, n_u, n_y)

    mu_shift = theta[-6:]

    link_ll = mc_link_lik(sample_w, mu_shift, q, ln_q, ln_1_q, ln_s)

    return - link_ll


def vi_obj(theta, q, ln_q, ln_1_q, ln_s, mu, sigma,
           n_u, n_y, raw_sample_w, jitter):

    q = tf.convert_to_tensor(q)

    ln_q = tf.convert_to_tensor(ln_q)

    ln_1_q = tf.convert_to_tensor(ln_1_q)

    ln_s = tf.convert_to_tensor(ln_s)

    mu = tf.convert_to_tensor(mu)

    sigma = tf.convert_to_tensor(sigma)

    c_theta = theta[:8]

    u = theta[8:8+3*n_u+9*n_u**2]

    mu_u = tf.reshape(theta[8+3*n_u+9*n_u**2:8+3*n_u+9*n_u**2+n_u], [-1, 1])

    sigma_u = tf.reshape(theta[8+3*n_u+9*n_u**2+n_u:-6], [-1, 1])

    C_u = kernel(c_theta, mu_u, sigma_u, jitter=tf.constant(jitter, dtype='float64'))

    C_wu = kernel_test(c_theta, mu, sigma, mu_u, sigma_u)

    C_wu = tf.transpose(C_wu[:3*n_u, :])

    C_diag_w = kernel_diag(c_theta, mu, sigma, jitter=tf.constant(jitter, dtype='float64'))

    sample_w = \
        get_sample_w(u,
                     tf.squeeze(tf.reshape(C_u, [1, -1])),
                     tf.squeeze(tf.reshape(C_wu, [1, -1])),
                     tf.squeeze(tf.reshape(C_diag_w, [1, -1])),
                     raw_sample_w, n_u, n_y)

    mu_shift = theta[-6:]

    link_ll = mc_link_lik(sample_w, mu_shift, q, ln_q, ln_1_q, ln_s)

    kl = get_noraml_kl(u, tf.squeeze(tf.reshape(C_u, [1, -1])), n_u)

    obj = - link_ll + kl

    return obj


def parameter_update(theta_0, q, ln_q, ln_1_q, ln_s, mu, sigma, n_u, n_y, jitter,
                     sample_size_w=4096, batch_size=None, optimizer_choice='adam',
                     lr=1e-3, max_batch=int(1024), factr=1e-8, plot_loss=True):

    batch_L = []

    if optimizer_choice == 'adam':
        optimizer = optimizers.Adam(lr=lr)
    elif optimizer_choice == 'adadelta':
        optimizer = optimizers.Adadelta(lr=lr)
    elif optimizer_choice == 'adagrad':
        optimizer = optimizers.Adagrad(lr=lr)
    elif optimizer_choice == 'adamax':
        optimizer = optimizers.Adamax(lr=lr)
    elif optimizer_choice == 'ftrl':
        optimizer = optimizers.Ftrl(lr=lr)
    elif optimizer_choice == 'nadam':
        optimizer = optimizers.Nadam(lr=lr)
    elif optimizer_choice == 'rmsprop':
        optimizer = optimizers.RMSprop(lr=lr)
    elif optimizer_choice == 'sgd':
        optimizer = optimizers.SGD(lr=lr)

    theta = tf.Variable(theta_0)

    fin_theta = theta_0.copy()

    if batch_size is None:
        batch_size = int(numpy.floor(n_y / 2))

    batch_idx = numpy.arange(0, n_y, batch_size)

    batch_num = len(batch_idx) - 1

    converge = False

    for i in range(0, int(1e8)):

        for j in range(0, batch_num):

            raw_sample_w = tf.random.normal((sample_size_w, 3 * (batch_idx[j + 1] - batch_idx[j])), dtype='float64')

            _, g_t = get_obj_g(theta,
                               q[batch_idx[j]:batch_idx[j + 1]],
                               ln_q[batch_idx[j]:batch_idx[j + 1]],
                               ln_1_q[batch_idx[j]:batch_idx[j + 1]],
                               ln_s[batch_idx[j]:batch_idx[j + 1]],
                               mu[batch_idx[j]:batch_idx[j + 1]],
                               sigma[batch_idx[j]:batch_idx[j + 1]],
                               n_u, (batch_idx[j + 1] - batch_idx[j]),
                               raw_sample_w, jitter)

            if numpy.isfinite(numpy.sum(g_t.numpy())):

                valid_theta = theta.numpy()

                optimizer.apply_gradients(zip([g_t], [theta]))

                theta = theta.numpy()

                theta[:2] = numpy.abs(theta[:2])

                theta[:2][theta[:2] <= 1e-8] = 1e-8

                theta[5:8][theta[5:8] <= 1e-8] = 1e-8

            else:

                theta = valid_theta.copy()

                # print('NaN in gradient.')

            raw_sample_w = tf.random.normal((sample_size_w, 3 * numpy.shape(q)[0]), dtype='float64')

            L_t = vi_obj(theta,
                         q,
                         ln_q,
                         ln_1_q,
                         ln_s,
                         mu,
                         sigma,
                         n_u,
                         numpy.shape(q)[0],
                         raw_sample_w, jitter)

            tmp_L = (L_t.numpy() / numpy.shape(q)[0])

            if len(batch_L) >= 2:
                if tmp_L < numpy.min(batch_L[:-1]):
                    fin_theta = theta.copy()

            theta = tf.Variable(theta)

            if numpy.mod(len(batch_L), 16) == 0:

                print('=============================================================================')

                print(theta[:8])
                print(theta[-6:])

                print('Batch: ' + str(len(batch_L)) + ', optimiser: ' + optimizer_choice + ', Loss: ' + str(tmp_L))

                print('=============================================================================')

            batch_L.append(numpy.min(tmp_L))

            if plot_loss:
                fig = matplotlib.pyplot.figure(figsize=(16, 9))

                matplotlib.pyplot.plot(numpy.arange(0, len(batch_L)),
                                       numpy.array(batch_L))

                matplotlib.pyplot.xlabel('Batches')

                matplotlib.pyplot.ylabel('Loss')

                matplotlib.pyplot.title('Learning Rate: ' + str(lr))

                matplotlib.pyplot.grid(True)

                matplotlib.pyplot.ylim([numpy.min(batch_L), numpy.median(batch_L)])

                try:
                    fig.savefig('./' + str(n_u) + '_' + optimizer_choice + '_' + str(lr) + '.png', bbox_inches='tight')
                except PermissionError:
                    pass
                except OSError:
                    pass

                matplotlib.pyplot.close(fig)

            if len(batch_L) > batch_num*16:
                previous_opt = numpy.min(batch_L.copy()[:-batch_num*16])

                current_opt = numpy.min(batch_L.copy()[-batch_num*16:])

                if numpy.mod(len(batch_L), 16) == 0:
                    print('Previous And Recent Top Averaged Loss Is:')
                    print(numpy.hstack([previous_opt, current_opt]))

                if previous_opt - current_opt <= numpy.abs(previous_opt * factr):
                    converge = True
                    break

                if len(batch_L) >= max_batch:
                    converge = True
                    break

        per_idx = numpy.random.permutation(n_y)

        q = q[per_idx]

        ln_q = ln_q[per_idx]

        ln_1_q = ln_1_q[per_idx]

        ln_s = ln_s[per_idx]

        mu = mu[per_idx]

        sigma = sigma[per_idx]

        if converge:
            break

    return fin_theta


def kernel_diag(theta, mu, sigma, jitter=tf.constant(1e-2, dtype='float64')):

    length_scale = theta[0]

    std = theta[1]

    omega = tf.reshape(theta[2:5], [3, 1])

    kappa = theta[5:8]

    K_p = RBF_p_diag(sigma, length_scale, std)

    B = coregion(omega, kappa)

    C = tf.concat(tf.unstack(tf.concat(tf.unstack(tf.tensordot(tf.reshape(tf.squeeze(K_p), [-1, 1]), B, axes=0),
                                                  axis=0), axis=1), axis=0), axis=1)

    return C + tf.tile(tf.linalg.eye(3, dtype='float64'), [len(mu), 1]) * jitter


def kernel_test(theta, mu, sigma, mu_train, sigma_train):

    length_scale = theta[0]

    std = theta[1]

    omega = tf.reshape(theta[2:5], [3, 1])

    kappa = theta[5:8]

    K_p = RBF_p_test(mu, sigma, mu_train, sigma_train, length_scale, std)

    B = coregion(omega, kappa)

    C = tf.concat(tf.unstack(tf.concat(tf.unstack(tf.tensordot(K_p, B, axes=0), axis=0), axis=1), axis=0), axis=1)

    return C


def kernel(theta, mu, sigma, jitter=tf.constant(1e-2, dtype='float64')):

    length_scale = theta[0]

    std = theta[1]

    omega = tf.reshape(theta[2:5], [3, 1])

    kappa = theta[5:8]

    K_p = RBF_p(mu, sigma, length_scale, std)

    B = coregion(omega, kappa)

    C = tf.concat(tf.unstack(tf.concat(tf.unstack(tf.tensordot(K_p, B, axes=0), axis=0), axis=1), axis=0), axis=1) + \
        tf.linalg.eye(3*len(mu), dtype='float64')*jitter

    return C


def RBF_p_test(mu, sigma, mu_train, sigma_train, length_scale, std):

    n_train = numpy.shape(mu_train)[0]

    n_test = numpy.shape(mu)[0]

    sigma2 = sigma ** 2

    sigma2_train = sigma_train ** 2

    mu_1 = tf.concat([tf.tensordot(mu_train, tf.ones((1, n_test), dtype='float64'),
                                   axes=[1, 0]), tf.transpose(mu)], axis=0)

    sigma2_1 = tf.concat([tf.tensordot(sigma2_train, tf.ones((1, n_test), dtype='float64'), axes=[1, 0]),
                          tf.transpose(sigma2)], axis=0)

    mu_2 = tf.tensordot(tf.ones((1, n_train + 1), dtype='float64'), mu, axes=[0, 1])

    sigma2_2 = tf.tensordot(tf.ones((1, n_train + 1), dtype='float64'), sigma2, axes=[0, 1])

    S = (sigma2_1 + sigma2_2 + (length_scale ** 2))

    K = length_scale * tf.math.pow(S, -0.5) * tf.math.exp(-0.5 * ((mu_1 - mu_2) ** 2) / S) * (std ** 2)

    return K


def RBF_p_diag(sigma, length_scale, std):

    sigma2 = sigma ** 2

    S = (sigma2 + sigma2 + (length_scale ** 2))

    K = length_scale * tf.math.pow(S, -0.5) * (std ** 2)

    return K


def RBF_p(mu, sigma, length_scale, std):
    # mu and sigma are column vector

    sigma2 = sigma ** 2

    n = numpy.shape(mu)[0]

    mu_1 = tf.tensordot(mu, tf.ones((1, n), dtype='float64'), axes=[1, 0])

    sigma2_1 = tf.tensordot(sigma2, tf.ones((1, n), dtype='float64'), axes=[1, 0])

    mu_2 = tf.tensordot(tf.ones((1, n), dtype='float64'), mu, axes=[0, 1])

    sigma2_2 = tf.tensordot(tf.ones((1, n), dtype='float64'), sigma2, axes=[0, 1])

    S = (sigma2_1 + sigma2_2 + (length_scale ** 2))

    K = length_scale * tf.math.pow(S, -0.5) * tf.math.exp(-0.5 * ((mu_1 - mu_2) ** 2) / S) * (std ** 2)

    return K


def coregion(omega, kappa):

    B = tf.tensordot(omega, tf.transpose(omega), [1, 0])

    B = B + tf.linalg.diag(tf.squeeze(kappa))

    return B


def get_obj_g(theta, q, ln_q, ln_1_q, ln_s, mu, sigma, n_u, n_y, raw_sample_w, jitter):

    with tf.GradientTape() as gt:
        
        gt.watch(theta)

        obj = vi_obj(theta, q, ln_q, ln_1_q, ln_s, mu, sigma, n_u, n_y, raw_sample_w, jitter)

        g = gt.gradient(obj, theta)

    return obj, g


# get_obj_g = tfe.gradients_function(vi_obj, params=[0])

