import autograd.numpy as numpy

import autograd

import scipy.linalg

import scipy.stats

import scipy.optimize

import visualisation

from tqdm import tqdm

numpy.set_printoptions(precision=2)

eps = numpy.finfo(numpy.random.randn(1).dtype).eps


class GP_Beta:

    def __init__(self, length_scale=None, std=None,
                 omega=None, kappa=None):

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

        self.C_inv = None

        self.H = None

        self.H_U = None

        self.H_V = None

        self.G = None

        self.w_map = None

        self.theta = None

        self.mu_w_test = None

        self.cov_w_test = None

        self.check_map = None

        self.gradient = None

        self.NLL = None

        self.A = None

        self.A_u = None

        self.L_u = None

        self.approx = None

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

    def fit(self, y, mu, sigma, n_u, dataset, model_class, fold_idx=0, approx='VI'):

        self.approx = approx

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

        c_theta[0] = (numpy.max(mu) - numpy.min(mu)) * 0.5
        
        # mu_std = numpy.std(mu)
        #
        # if mu_std > 0:
        #     c_theta[0] = mu_std

        c_theta[2:5] = numpy.random.randn(3) * 1e-8

        c_theta[5:7] = 1e-2

        if approx == 'VI':

            self.n_u = n_u

            self.mu_u = (numpy.random.choice(mu.ravel(), self.n_u, replace=False) +
                         numpy.random.randn(self.n_u) * 1e-1).reshape(-1, 1)

            self.sigma_u = numpy.abs(numpy.random.choice(sigma.ravel(), self.n_u, replace=False)).reshape(-1, 1)

            C_u, _ = kernel(c_theta, self.mu_u, self.sigma_u)

            A_u = numpy.zeros((self.n_u, 3))

            L_u = scipy.linalg.cholesky(C_u)

            mu_shift = numpy.array([1, 0, 1, 0, 1, 0])

            theta = numpy.hstack([c_theta.ravel(), A_u.ravel(), L_u.ravel(),
                                  self.mu_u.ravel(), self.sigma_u.ravel(), mu_shift])

            theta = adam_update(theta, self.q, self.ln_q, self.ln_1_q, self.ln_s, self.mu, self.sigma,
                                self.n_u, self.n, dataset, model_class, fold_idx)

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

            self.B = coregion(self.omega, self.kappa)[0]

            # self.C, K_g = kernel(c_theta, self.mu, self.sigma)

            C_u = kernel(c_theta, self.mu_u, self.sigma_u)[0]

            C_inv_u = numpy.linalg.inv(C_u)

            self.C_u = C_u

            self.C_u_inv = C_inv_u

        elif approx == 'LA':

            self.w_map = numpy.zeros([numpy.shape(y)[0], 3]).ravel()

            self.A = numpy.zeros([numpy.shape(y)[0], 1])

            res = scipy.optimize.fmin_l_bfgs_b(func=all_log_lik, x0=c_theta,
                                               args=(self, ),
                                               bounds=((1e-8, None), (1.0, 1.0),
                                                       (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
                                                       (1e-8, None), (1e-8, None), (1e-8, None)),
                                               approx_grad=False,
                                               maxls=64,
                                               iprint=1,
                                               m=512)

            theta = res[0]

            self.length_scale = theta[0]

            self.std = theta[1]

            self.omega = theta[2:5].reshape(3, 1)

            self.kappa = theta[5:8].reshape(3, 1)

            self.theta = theta

            self.B = coregion(self.omega, self.kappa)[0]

            self.C, K_g = kernel(theta, self.mu, self.sigma)

            self.w_map, self.A = newton_update(self.q, self.ln_q, self.ln_1_q, self.ln_s, self.C, self.n)

            link_g = get_link_g(self.w_map, self.q, self.ln_q, self.ln_1_q, self.ln_s).reshape(-1, 1)

            self.check_map = numpy.sum(numpy.abs(numpy.matmul(self.C, link_g) - self.w_map.reshape(-1, 1)))

            self.H, self.H_U, self.H_V = get_link_h(self.w_map, self.q, self.ln_q, self.ln_1_q, self.ln_s)

            self.G = numpy.eye(self.n * 3) + numpy.matmul(numpy.matmul(self.H_V, self.C), self.H_U)

        # print('=============================================================')
        #
        # print('Current parameters are:')
        #
        # print(self.theta)
        #
        # print('L-BFGS-B iteration:')
        #
        # print(res[2]['nit'])
        #
        # print('L-BFGS-B Func calls:')
        #
        # print(res[2]['funcalls'])
        #
        # print('=============================================================')

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

    if mdl.approx == 'LA':

        mu_train = mdl.mu

        sigma_train = mdl.sigma

        C_test = kernel_test(theta, mu_test, sigma_test, mu_train, sigma_train)[0]

        C_upper = C_test[:3 * n_y, :]

        C_lower = C_test[3 * n_y:3 * n_y + 3, :]

        GQ, GR = scipy.linalg.qr(mdl.G)

        H_U = mdl.H_U

        H_V = mdl.H_V

        CH_inv = numpy.matmul(H_U, scipy.linalg.solve_triangular(GR, numpy.matmul(GQ.transpose(), H_V)))

        A = mdl.A

        for j in range(0, n_test):

            mu_w_hat[j, :] = numpy.matmul(C_upper[:, j*3:(j+1)*3].transpose(), A).ravel()

            cov_w_hat[j, :, :] = C_lower[:, j*3:(j+1)*3] - numpy.matmul(
                numpy.matmul(C_upper[:, j*3:(j+1)*3].transpose(), CH_inv), C_upper[:, j*3:(j+1)*3])

    elif mdl.approx == 'VI':

        mu_train = mdl.mu_u

        C_u = mdl.C_u

        C_u_inv = mdl.C_u_inv

        sigma_train = mdl.sigma_u

        C_test = kernel_test(theta, mu_test, sigma_test, mu_train, sigma_train)[0]

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

        w_sample = scipy.stats.multivariate_normal.rvs(size=1024, mean=mu_w[i, :], cov=cov_w[i, :, :])

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


def link_log_lik(w, q, ln_q, ln_1_q, ln_s):

    w = w.reshape(-1, 3)

    a = -numpy.exp(w[:, 0]).reshape(-1, 1)

    b = numpy.exp(w[:, 1]).reshape(-1, 1)

    c = w[:, 2].reshape(-1, 1)

    tmp_sum = a * ln_q + c + b * ln_1_q

    tmp_exp = numpy.exp(tmp_sum)

    tmp_de = numpy.where(tmp_exp.ravel() <= 1e-16,
                         2 * numpy.log(1 + tmp_exp.ravel()),
                         2 * (tmp_sum.ravel() + numpy.log(1 + 1 / tmp_exp.ravel()))).reshape(-1, 1)

    ln_s_hat = ln_s + tmp_sum + numpy.log((a + b) * q - a) - ln_q - ln_1_q - tmp_de

    # L = numpy.sum(ln_s_hat)

    # if numpy.isnan(L):
    #     import pdb; pdb.set_trace()
    #
    # if numpy.isinf(L):
    #     import pdb; pdb.set_trace()

    # print([numpy.mean(ln_s), numpy.mean(ln_s_hat)])

    return ln_s_hat


def prior_error(mu_shift, w, n_u):

    a = -numpy.abs(w[:, numpy.arange(0, n_u)*3] + mu_shift[1])

    b = numpy.abs(w[:, numpy.arange(0, n_u)*3+1] + mu_shift[3])

    c = w[:, numpy.arange(0, n_u)*3+2] + mu_shift[5]

    q = numpy.linspace(1e-8, 1 - 1e-8, 128)

    # q = q.ravel()

    q_hat = numpy.mean(1 / (1 + numpy.exp(numpy.log(q)[None, None, :] * a[:, :, None] +
                                          numpy.log(1 - q)[None, None, :] * b[:, :, None] +
                                          c[:, :, None])), axis=0)

    return numpy.mean((q - q_hat) ** 2)


prior_error_grad = autograd.jacobian(prior_error, 0)


def match_prior(mu_shift_0, C_u, sample_size=128, lr=1e-3, beta_1=0.9, beta_2=0.999, maxiter=int(1024), factr=1e-4):

    n_u = int(numpy.shape(C_u)[0] / 3)

    L = []

    m = numpy.zeros(6)

    v = numpy.zeros(6)

    mu_shift = mu_shift_0

    fin_mu_shift = mu_shift_0

    fin_L = None

    for i in range(0, maxiter):

        w = scipy.stats.multivariate_normal(mean=numpy.zeros(3*n_u), cov=C_u).rvs(sample_size)

        L.append(prior_error(mu_shift, w, n_u))

        g = prior_error_grad(mu_shift, w, n_u)

        g[numpy.isnan(g)] = 0.0

        g[numpy.isinf(g)] = 0.0

        m = beta_1 * m + (1 - beta_1) * g

        v = beta_2 * v + (1 - beta_2) * g * g

        mu_shift = mu_shift - lr * m / (v**0.5 + eps)

        if len(L) >= 2:
            if L[-1] < numpy.min(L[:-1]):

                fin_L = L[-1].copy()

                fin_mu_shift = mu_shift.copy()

        if len(L) > 32:

            previous_opt = numpy.min(L.copy()[:-32])

            current_opt = numpy.min(L.copy()[-32:])

            if previous_opt - current_opt <= numpy.abs(previous_opt * factr):

                break

    print('=============================================================================')

    print('Prior Matched: ')

    print('Total Iterations: ' + str(i), ', Loss: ' + str(fin_L) + ', Mu_Shift:' + str(fin_mu_shift))

    print('=============================================================================')

    return fin_mu_shift


def mc_link_lik(w, mu_shift, q, ln_q, ln_1_q, ln_s):

    n = numpy.shape(q)[0]

    w_a = w[:, numpy.arange(0, n)*3]

    w_b = w[:, numpy.arange(0, n)*3+1]

    a = -numpy.exp(w_a / mu_shift[0] + mu_shift[1])

    b = numpy.exp(w_b / mu_shift[2] + mu_shift[3])

    c = w[:, numpy.arange(0, n)*3+2] / mu_shift[4] + mu_shift[5]

    tmp_sum = a * ln_q.ravel() + b * ln_1_q.ravel() + c

    tmp_de = numpy.where(tmp_sum <= 0,
                         2 * numpy.log(1 + numpy.exp(tmp_sum)),
                         2 * (tmp_sum + numpy.log(1 + 1 / (numpy.exp(tmp_sum)))))

    ln_s_hat = (tmp_sum + numpy.log((a + b) * q.ravel() - a) - ln_q.ravel() - ln_1_q.ravel() - tmp_de) + ln_s.ravel()

    mean_exp = numpy.mean(numpy.exp(ln_s_hat), axis=0)

    ln_mean_s_hat = numpy.where(mean_exp > 0, numpy.log(mean_exp), numpy.log(1e-16))

    link_ll = numpy.sum(ln_mean_s_hat)

    return link_ll


def get_sample_w_step(A_u, Q_w, C_wu, raw_sample_w):

    n_u = int(len(A_u)/3)

    A_u = A_u.reshape(-1, 1)

    Q_w = Q_w.reshape(3, 3)

    C_wu = C_wu.reshape(-1, 3 * n_u)

    mu_w = numpy.matmul(C_wu, A_u).ravel()

    return (numpy.matmul(raw_sample_w, numpy.linalg.cholesky(Q_w)) + mu_w).ravel()


def get_Q_w(L_u, C_u, C_wu, C_diag_w, n_u):

    C_u = C_u.reshape(3*n_u, 3*n_u)

    C_wu = C_wu.reshape(-1, 3*n_u)

    C_diag_w = C_diag_w.reshape(3, 3)

    T_wu = numpy.matmul(C_wu, numpy.linalg.inv(C_u))

    L_u_hat = L_u.reshape(3*n_u, 3*n_u)

    V_u = numpy.matmul(L_u_hat.transpose(), L_u_hat)

    D_u = V_u - C_u

    return (C_diag_w + numpy.matmul(numpy.matmul(T_wu, D_u), T_wu.transpose())).ravel()


def get_sample_w(u, C_u, C_wu, C_diag_w, raw_sample_w, n_u, n_y):

    A_u = u[:3*n_u]

    L_u = u[3*n_u:]

    sample_size_w = numpy.shape(raw_sample_w)[0]

    sample_w = numpy.zeros((sample_size_w, 3*n_y))

    Q_w = []

    A_u_g = numpy.zeros((sample_size_w*3*n_y, len(A_u)))

    L_u_g = numpy.zeros((sample_size_w*3*n_y, len(L_u)))

    C_u_g = numpy.zeros((sample_size_w*3*n_y, len(C_u)))

    C_wu_g = numpy.zeros((sample_size_w*3*n_y, len(C_wu)))

    C_diag_w_g = numpy.zeros((sample_size_w*3*n_y, len(C_diag_w)))

    for i in range(0, n_y):

        Q_w.append(get_Q_w(L_u, C_u, C_wu[i*(9*n_u):(i+1)*(9*n_u)], C_diag_w[i*9:(i+1)*9], n_u))

        sample_w[:, i*3:(i+1)*3] = \
            get_sample_w_step(A_u, Q_w[-1].ravel(), C_wu[i*(9*n_u):(i+1)*(9*n_u)],
                              raw_sample_w[:, i*3:(i+1)*3]).reshape(-1, 3)

        tmp_idx = ((numpy.arange(0, sample_size_w) * (3*n_y)).reshape(-1, 1).repeat(3, axis=1) +
                   (numpy.array([0, 1, 2]) + 3*i).ravel()).ravel()

        A_u_g[tmp_idx, :] = \
            A_u_grad(A_u, Q_w[-1], C_wu[i*(9*n_u):(i+1)*(9*n_u)], raw_sample_w[:, i*3:(i+1)*3])

        L_u_g[tmp_idx, :] = \
            numpy.matmul(Q_w_grad(A_u, Q_w[-1], C_wu[i*(9*n_u):(i+1)*(9*n_u)], raw_sample_w[:, i*3:(i+1)*3]),
                         L_u_grad(L_u, C_u, C_wu[i*(9*n_u):(i+1)*(9*n_u)], C_diag_w[i*9:(i+1)*9], n_u))

        C_u_g[tmp_idx, :] = \
            numpy.matmul(Q_w_grad(A_u, Q_w[-1], C_wu[i*(9*n_u):(i+1)*(9*n_u)], raw_sample_w[:, i*3:(i+1)*3]),
                         C_u_grad(L_u, C_u, C_wu[i*(9*n_u):(i+1)*(9*n_u)], C_diag_w[i*9:(i+1)*9], n_u))

        C_diag_w_g[tmp_idx, i*9:(i+1)*9] = \
            numpy.matmul(Q_w_grad(A_u, Q_w[-1], C_wu[i*(9*n_u):(i+1)*(9*n_u)], raw_sample_w[:, i*3:(i+1)*3]),
                         C_diag_w_grad(L_u, C_u, C_wu[i*(9*n_u):(i+1)*(9*n_u)], C_diag_w[i*9:(i+1)*9], n_u))

        C_wu_g[tmp_idx, i*(9*n_u):(i+1)*(9*n_u)] = \
            C_wu_grad(A_u, Q_w[-1], C_wu[i*(9*n_u):(i+1)*(9*n_u)], raw_sample_w[:, i*3:(i+1)*3]) + \
            numpy.matmul(Q_w_grad(A_u, Q_w[-1], C_wu[i*(9*n_u):(i+1)*(9*n_u)], raw_sample_w[:, i*3:(i+1)*3]),
                         C_wu_grad_Q_w(L_u, C_u, C_wu[i*(9*n_u):(i+1)*(9*n_u)], C_diag_w[i*9:(i+1)*9], n_u))

    return sample_w, A_u_g, L_u_g, C_u_g, C_diag_w_g, C_wu_g


def get_noraml_kl(u, C_u, n_u):

    C_u = C_u.reshape(3*n_u, 3*n_u)

    A_u = u[:3*n_u].reshape(-1, 1)

    mu_u = numpy.matmul(C_u, A_u).ravel()

    L_u = u[3*n_u:].reshape(3*n_u, 3*n_u)

    V_u = numpy.matmul(L_u.transpose(), L_u)

    kl = -0.5 * numpy.linalg.slogdet(V_u)[1] + \
        0.5 * numpy.linalg.slogdet(C_u)[1] + \
        0.5 * numpy.matmul(mu_u.reshape(1, -1), A_u) + 0.5 * numpy.trace(numpy.matmul(numpy.linalg.inv(C_u), V_u))

    return kl.ravel()


def vi_obj(theta, q, ln_q, ln_1_q, ln_s, mu, sigma,
           n_u, n_y, raw_sample_w):

    c_theta = theta[:8]

    u = theta[8:8+3*n_u+9*n_u**2]

    mu_u = theta[8+3*n_u+9*n_u**2:8+3*n_u+9*n_u**2+n_u].reshape(-1, 1)

    sigma_u = theta[8+3*n_u+9*n_u**2+n_u:-6].reshape(-1, 1)

    C_u, C_g_u = kernel(c_theta, mu_u, sigma_u)

    C_wu, C_g_wu = kernel_test(c_theta, mu, sigma, mu_u, sigma_u)

    C_wu = C_wu[:3*n_u, :].transpose()

    for i in range(0, 10):
        C_g_wu[i] = C_g_wu[i][:3*n_u, :].transpose()

    C_diag_w, C_g_diag_w = kernel_diag(c_theta, mu, sigma)

    sample_w, A_u_g, L_u_g, C_u_g, C_diag_w_g, C_wu_g = \
        get_sample_w(u, C_u.ravel(), C_wu.ravel(), C_diag_w.ravel(), raw_sample_w, n_u, n_y)

    # mu_shift = match_prior(mu_shift_0=theta[-6:], C_u=C_u, sample_size=32)

    mu_shift = theta[-6:]

    link_ll = mc_link_lik(sample_w, mu_shift, q, ln_q, ln_1_q, ln_s)

    kl = get_noraml_kl(u, C_u.ravel(), n_u)

    link_g = -get_mc_link_g(sample_w, mu_shift, q, ln_q, ln_1_q, ln_s)

    mu_shift_g = -get_mu_shift_g(sample_w, mu_shift, q, ln_q, ln_1_q, ln_s)

    kl_g = get_kl_g(u, C_u.ravel(), n_u).ravel()

    kl_g_C_u = get_kl_g_C_u(u, C_u.ravel(), n_u).ravel()

    mu_shift_g[numpy.isnan(mu_shift_g)] = 0

    mu_shift_g[numpy.isinf(mu_shift_g)] = 0

    kl_g[numpy.isnan(kl_g)] = 0

    kl_g[numpy.isinf(kl_g)] = 0

    A_u_g[numpy.isnan(A_u_g)] = 0

    A_u_g[numpy.isinf(A_u_g)] = 0

    L_u_g[numpy.isnan(L_u_g)] = 0

    L_u_g[numpy.isinf(L_u_g)] = 0

    C_u_g[numpy.isnan(C_u_g)] = 0

    C_u_g[numpy.isinf(C_u_g)] = 0

    C_diag_w_g[numpy.isnan(C_diag_w_g)] = 0

    C_diag_w_g[numpy.isinf(C_diag_w_g)] = 0

    C_wu_g[numpy.isnan(C_wu_g)] = 0

    C_wu_g[numpy.isinf(C_wu_g)] = 0

    link_g[numpy.isnan(link_g)] = 0

    link_g[numpy.isinf(link_g)] = 0

    kl_g_C_u[numpy.isnan(kl_g_C_u)] = 0

    kl_g_C_u[numpy.isinf(kl_g_C_u)] = 0

    obj = - link_ll + kl

    u_g = numpy.zeros_like(u)

    u_g[:3*n_u] = numpy.matmul(link_g.ravel().reshape(1, -1), A_u_g)

    u_g[3*n_u:] = numpy.matmul(link_g.ravel().reshape(1, -1), L_u_g)

    u_g = u_g + kl_g

    theta_g = numpy.zeros_like(c_theta)

    for i in range(0, len(theta_g)):

        theta_g[i] = numpy.matmul(numpy.matmul(link_g.ravel().reshape(1, -1), C_u_g),
                                  C_g_u[i].ravel().reshape(-1, 1))[0][0] + \
        numpy.matmul(numpy.matmul(link_g.ravel().reshape(1, -1), C_wu_g),
                     C_g_wu[i].ravel().reshape(-1, 1))[0][0] + \
        numpy.matmul(numpy.matmul(link_g.ravel().reshape(1, -1), C_diag_w_g),
                     C_g_diag_w[i].ravel().reshape(-1, 1))[0][0] + \
        numpy.matmul(kl_g_C_u.ravel().reshape(1, -1), C_g_u[i].ravel().reshape(-1, 1))[0][0]

    mu_C_u_g = numpy.matmul(link_g.ravel().reshape(1, -1), C_u_g).reshape(3*n_u, 3*n_u) * C_g_u[8] + \
        kl_g_C_u.reshape(3*n_u, 3*n_u) * C_g_u[8]

    sigma_C_u_g = numpy.matmul(link_g.ravel().reshape(1, -1), C_u_g).reshape(3*n_u, 3*n_u) * C_g_u[9] + \
        kl_g_C_u.reshape(3*n_u, 3*n_u) * C_g_u[9]

    mu_C_wu_g = numpy.matmul(link_g.ravel().reshape(1, -1), C_wu_g).reshape(-1, 3*n_u) * C_g_wu[8]

    sigma_C_wu_g = numpy.matmul(link_g.ravel().reshape(1, -1), C_wu_g).reshape(-1, 3*n_u) * C_g_wu[9]

    mu_g = numpy.sum(mu_C_u_g[numpy.arange(0, n_u)*3, :], axis=1).ravel()*2 + \
        numpy.sum(mu_C_u_g[numpy.arange(0, n_u)*3+1, :], axis=1).ravel()*2 + \
        numpy.sum(mu_C_u_g[numpy.arange(0, n_u)*3+2, :], axis=1).ravel()*2 + \
        numpy.sum(mu_C_wu_g[:, numpy.arange(0, n_u)*3], axis=0).ravel() + \
        numpy.sum(mu_C_wu_g[:, numpy.arange(0, n_u)*3+1], axis=0).ravel() + \
        numpy.sum(mu_C_wu_g[:, numpy.arange(0, n_u)*3+2], axis=0).ravel()

    sigma_g = numpy.sum(sigma_C_u_g[numpy.arange(0, n_u)*3, :], axis=1).ravel()*2 + \
        numpy.sum(sigma_C_u_g[numpy.arange(0, n_u)*3+1, :], axis=1).ravel()*2 + \
        numpy.sum(sigma_C_u_g[numpy.arange(0, n_u)*3+2, :], axis=1).ravel()*2 + \
        numpy.sum(sigma_C_wu_g[:, numpy.arange(0, n_u)*3], axis=0).ravel() + \
        numpy.sum(sigma_C_wu_g[:, numpy.arange(0, n_u)*3+1], axis=0).ravel() + \
        numpy.sum(sigma_C_wu_g[:, numpy.arange(0, n_u)*3+2], axis=0).ravel()

    obj_g = numpy.hstack([theta_g, u_g, mu_g, sigma_g])

    obj_g[numpy.isnan(obj_g)] = 0

    obj_g[numpy.isinf(obj_g)] = 0

    # print(numpy.array([numpy.sum(ln_s)/n_y, link_ll/n_y, -link_ll, kl, obj]))

    return obj, numpy.hstack([theta_g, u_g, mu_g, sigma_g, mu_shift_g]), -link_ll, -kl


def all_log_lik(theta, *args):

    (mdl,) = args

    # print('=============================================================')
    #
    # print('current theta is:')

    print(theta.ravel())

    C, K_g = kernel(theta, mdl.mu, mdl.sigma)

    w_map, A = newton_update(mdl.q, mdl.ln_q, mdl.ln_1_q, mdl.ln_s, C, mdl.n)

    link_g = get_link_g(w_map, mdl.q, mdl.ln_q, mdl.ln_1_q, mdl.ln_s).reshape(-1, 1)

    mdl.check_map = numpy.sum(numpy.abs(numpy.matmul(C, + link_g) - w_map.reshape(-1, 1)))

    mdl.w_map = w_map

    mdl.A = A

    H, H_U, H_V = get_link_h(w_map, mdl.q, mdl.ln_q, mdl.ln_1_q, mdl.ln_s)

    G = numpy.eye(numpy.shape(mdl.y)[0] * 3) + numpy.matmul(numpy.matmul(H_V, C), H_U)

    gradient = get_gradient(theta, w_map, A, C, G, H_U, H_V, link_g,
                            mdl.y, mdl.mu, mdl.sigma,
                            mdl.q, mdl.ln_q, mdl.ln_1_q, mdl.ln_s, K_g)

    mdl.gradient = gradient.copy()

    link_lik = numpy.sum(link_log_lik(w_map, mdl.q, mdl.ln_q, mdl.ln_1_q, mdl.ln_s))

    L = -0.5 * numpy.matmul(A.transpose(), w_map.reshape(-1, 1))[0, 0] + link_lik - 0.5 * numpy.linalg.slogdet(G)[1]

    # print('current gradient is:')
    #
    # print(gradient)
    #
    # print('Check MAP:')
    #
    # print(mdl.check_map)
    #
    # print('w_map is:')
    #
    # print(w_map)
    #
    # print('Log-Lik is:')
    #
    # print(L)
    #
    # print('=============================================================')
    #
    # print([-L, mdl.check_map, numpy.abs(gradient).sum()])

    return - L, - gradient


def get_L(A, w, link_lik, G):

    L = -0.5 * numpy.matmul(A.transpose(), w.reshape(-1, 1))[0, 0] + link_lik - 0.5 * numpy.linalg.slogdet(G)[1]

    return L


def e_link_log_lik(w_0, w_1, w_2, q, ln_q, ln_1_q, ln_s):

    a = -numpy.exp(w_0).reshape(-1, 1)

    b = numpy.exp(w_1).reshape(-1, 1)

    c = w_2.reshape(-1, 1)

    tmp_sum = a * ln_q + c + b * ln_1_q

    tmp_exp = numpy.exp(tmp_sum)

    tmp_de = numpy.where(tmp_exp.ravel() <= 1e-16,
                         2 * numpy.log(1 + tmp_exp.ravel()),
                         2 * (tmp_sum.ravel() + numpy.log(1 + 1 / tmp_exp.ravel()))).reshape(-1, 1)

    ln_s_hat = ln_s + tmp_sum + numpy.log((a + b) * q - a) - ln_q - ln_1_q - tmp_de

    return ln_s_hat


def get_link_g(w, q, ln_q, ln_1_q, ln_s):

    w = w.reshape(-1, 3)

    n = numpy.shape(w)[0]

    g = numpy.zeros((n, 3))

    for i in range(0, 3):
        tmp_grad = autograd.elementwise_grad(e_link_log_lik, i)
        g[:, i] = tmp_grad(w[:, 0].reshape(-1, 1), w[:, 1].reshape(-1, 1), w[:, 2].reshape(-1, 1),
                           q, ln_q, ln_1_q, ln_s).ravel()

    return g.ravel()


def get_link_h(w, q, ln_q, ln_1_q, ln_s):

    w = w.reshape(-1, 3)

    n = numpy.shape(w)[0]

    h = numpy.zeros((n*3, n*3))

    for i in range(0, 3):
        for j in range(0, 3):
            tmp_grad = autograd.elementwise_grad(autograd.elementwise_grad(e_link_log_lik, i), j)
            h[numpy.arange(0, n)*3 + i, numpy.arange(0, n)*3 + j] = \
                tmp_grad(w[:, 0].reshape(-1, 1), w[:, 1].reshape(-1, 1), w[:, 2].reshape(-1, 1),
                         q, ln_q, ln_1_q, ln_s).ravel()

    h_u = numpy.zeros((n*3, n*3))

    h_v = numpy.zeros((n*3, n*3))

    for i in range(0, n):
        tmp_u, tmp_s, tmp_v = scipy.linalg.svd(a=-h[i*3:(i+1)*3, i*3:(i+1)*3])

        tmp_s = tmp_s ** 0.5

        h_u[i*3:(i+1)*3, i*3:(i+1)*3] = numpy.matmul(tmp_u, numpy.diag(tmp_s))

        h_v[i*3:(i+1)*3, i*3:(i+1)*3] = numpy.matmul(numpy.diag(tmp_s), tmp_v)

    return -h, h_u, h_v


def get_link_hessian_g(w, q, ln_q, ln_1_q, ln_s):

    w = w.reshape(-1, 3)

    n = numpy.shape(w)[0]

    # h = numpy.zeros((n * 3, n * 3, n * 3))

    h = numpy.zeros((n, 3, 3, 3))

    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                tmp_grad = autograd.elementwise_grad(
                    autograd.elementwise_grad(autograd.elementwise_grad(e_link_log_lik, i), j), k)
                # h[numpy.arange(0, n) * 3 + i, numpy.arange(0, n) * 3 + j, numpy.arange(0, n) * 3 + k] = \
                h[:, i, j, k] = \
                    tmp_grad(w[:, 0].reshape(-1, 1), w[:, 1].reshape(-1, 1), w[:, 2].reshape(-1, 1),
                             q, ln_q, ln_1_q, ln_s).ravel()

    return h


def get_gradient(theta, w_map, A, C, G, H_U, H_V, link_g, y, mu, sigma, q, ln_q, ln_1_q, ln_s, K_g):

    n = int(numpy.shape(C)[0] / 3)

    GQ, GR = scipy.linalg.qr(G)

    # (C + H^-1)^-1
    CH_inv = numpy.matmul(H_U, scipy.linalg.solve_triangular(GR, numpy.matmul(GQ.transpose(), H_V)))

    H_g = - get_link_hessian_g(w_map, q, ln_q, ln_1_q, ln_s)

    # (C^-1 + H)^-1
    L = C - numpy.matmul(numpy.matmul(C, H_U),
                         scipy.linalg.solve_triangular(GR, numpy.matmul(GQ.transpose(), numpy.matmul(H_V, C))))

    # (I + CH)^-1
    CH = numpy.eye(3*n) - numpy.matmul(C, CH_inv)

    g_direct = numpy.zeros_like(theta)

    g_indirect = numpy.zeros_like(theta)

    for i in range(0, len(g_direct)):

        g_direct[i] = 0.5 * numpy.matmul(numpy.matmul(A.transpose(), K_g[i]), A) \
                      - 0.5 * numpy.trace(numpy.matmul(CH_inv, K_g[i]))

    tmp_trace = numpy.zeros(3 * n)

    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                tmp_trace[i*n:(i+1)*n] = tmp_trace[i*n:(i+1)*n] + \
                                         (L[numpy.arange(0, n)*3+j, numpy.arange(0, n)*3+k] * H_g[:, i, j, k])

    for i in range(0, len(g_indirect)):

        tmp_g = numpy.matmul(numpy.matmul(CH, K_g[i]), + link_g)

        g_indirect[i] = numpy.sum((-0.5) * tmp_trace * tmp_g.ravel())

    return g_direct + g_indirect


def adam_update(theta_0, q, ln_q, ln_1_q, ln_s, mu, sigma, n_u, n_y,
                dataset, model_class, fold_idx, sample_size_w=32, batch_size=128,
                lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-8, max_batch=int(1024), factr=1e-8):

    batch_L = []

    batch_nll = []

    back_kl = []

    n_theta = len(theta_0)

    m = numpy.zeros(n_theta)

    v = numpy.zeros(n_theta)

    theta = theta_0

    fin_theta = theta_0

    if batch_size is None:
        batch_size = n_y

    batch_idx = numpy.arange(0, n_y, batch_size)

    batch_num = len(batch_idx) - 1

    converge = False

    mean_L = []

    for i in range(0, int(1e8)):

        for j in range(0, batch_num):

            raw_sample_w = numpy.random.randn(sample_size_w, 3*(batch_idx[j+1]-batch_idx[j]))

            L_t, g_t, link_nll_t, kl_t = vi_obj(theta,
                                                q[batch_idx[j]:batch_idx[j + 1]],
                                                ln_q[batch_idx[j]:batch_idx[j + 1]],
                                                ln_1_q[batch_idx[j]:batch_idx[j + 1]],
                                                ln_s[batch_idx[j]:batch_idx[j + 1]],
                                                mu[batch_idx[j]:batch_idx[j + 1]],
                                                sigma[batch_idx[j]:batch_idx[j + 1]],
                                                n_u, (batch_idx[j + 1] - batch_idx[j]),
                                                raw_sample_w)

            m = beta_1 * m + (1 - beta_1) * g_t

            v = beta_2 * v + (1 - beta_2) * g_t * g_t

            theta = theta - lr * m / (v**0.5 + eps)

            theta[:2] = numpy.abs(theta[:2])

            theta[:2][theta[:2] <= 1e-8] = 1e-8

            theta[5:8][theta[5:8] <= 1e-8] = 1e-8

            batch_L.append(L_t)

            batch_nll.append(link_nll_t / (batch_idx[j+1] - batch_idx[j]))

            back_kl.append(kl_t)

            if len(batch_L) >= 8:
                mean_L.append(numpy.mean(batch_L[-8:]))
            else:
                for k in range(0, len(mean_L)):
                    mean_L[k] = numpy.mean(batch_L)
                mean_L.append(numpy.mean(batch_L))

            if len(mean_L) >= 2:
                if mean_L[-1] < numpy.min(mean_L[:-1]):
                    fin_theta = theta.copy()

            if len(mean_L) >= 2:
                visualisation.plot_learning_loss(mean_L, batch_nll, back_kl, ln_s,
                                                 dataset, model_class, n_u, lr, fold_idx)

            if len(mean_L) > 64:
                previous_opt = numpy.min(mean_L.copy()[:-64])

                current_opt = numpy.min(mean_L.copy()[-64:])

                print('=============================================================================')

                print('Previous And Recent 32 Top Loss Is:')

                print(numpy.hstack([previous_opt, current_opt]))

                print('=============================================================================')

                if previous_opt - current_opt <= numpy.abs(previous_opt * factr):
                    converge = True
                    break

                if len(mean_L) >= max_batch:
                    converge = True
                    break

        per_idx = numpy.random.permutation(n_y)

        q = q[per_idx]

        ln_q = ln_q[per_idx]

        ln_1_q = ln_1_q[per_idx]

        ln_s = ln_s[per_idx]

        mu = mu[per_idx]

        sigma = sigma[per_idx]

        print('=============================================================================')

        print('Dataset: ' + str(dataset) + ', Base: ' + model_class + ', N_u: ' + str(n_u) +
              ' , Fold: ' + str(fold_idx))

        print('Iteration: ' + str(i), ', Loss: ' + str(numpy.array(numpy.sum(batch_L[-batch_num:]))) +
              ', Link:' + str(numpy.array(numpy.mean(batch_nll[-batch_num:]))))

        print('=============================================================================')

        if converge:
            break

    return fin_theta


def newton_update(q, ln_q, ln_1_q, ln_s, C, n, maxiter=int(1024), ftol=0.0):

    # print('=============================================================')

    w = numpy.zeros((n, 3)).ravel()

    A = numpy.zeros((3*n, 1))

    L_0 = - (numpy.sum(link_log_lik(w, q, ln_q, ln_1_q, ln_s)) - 0.5 * numpy.linalg.slogdet(C)[1])

    L_old = L_0

    step_size_list = numpy.hstack([numpy.logspace(0, -1, 8)])

    link_g = get_link_g(w, q, ln_q, ln_1_q, ln_s).reshape(-1, 1)

    for i in range(0, maxiter):

        H, H_U, H_V = get_link_h(w, q, ln_q, ln_1_q, ln_s)

        G = numpy.eye(n * 3) + numpy.matmul(numpy.matmul(H_V, C), H_U)

        GQ, GR = scipy.linalg.qr(G)

        for j in range(0, len(step_size_list)):

            step = step_size_list[j]

            b = (1 - step) * A + numpy.matmul(H, w.reshape(-1, 1)) + step * link_g

            tmp_A = (b - numpy.matmul(H_U, scipy.linalg.solve_triangular(GR,
                                                                         numpy.matmul(GQ.transpose(),
                                                                                      numpy.matmul(numpy.matmul(H_V, C),
                                                                                                   b)))))

            tmp_w = numpy.matmul(C, tmp_A).ravel()

            L = - (numpy.sum(link_log_lik(tmp_w, q, ln_q, ln_1_q, ln_s))
                   - 0.5 * numpy.matmul(tmp_w.reshape(1, -1), tmp_A)[0, 0] - 0.5 * numpy.linalg.slogdet(C)[1])

            if (L - L_old) < 0:
                break

        if L < L_old:

            w = tmp_w.copy()

            A = tmp_A.copy()

        else:

            # step = 0

            L = L_old

        link_g = get_link_g(w, q, ln_q, ln_1_q, ln_s).reshape(-1, 1)

        # check_map = numpy.sum(numpy.abs(numpy.matmul(C, + link_g) - w.reshape(-1, 1)))

        # print([i, L, check_map, step])

        if L_old - L <= ftol:
            # print('terminate as there is not enough changes on Psi.')
            # if check_map >= 1.0:
            #     import pdb; pdb.set_trace()
            break

        L_old = L

        if numpy.isinf(L) | numpy.isnan(L):
            w = numpy.zeros_like(w)
            break

    # print('=============================================================')

    return w, A


def kernel_diag(theta, mu, sigma, jitter=1e-2):

    length_scale = theta[0]

    std = theta[1]

    omega = numpy.reshape(theta[2:5], (3, 1))

    kappa = theta[5:8]

    K_p, raw_K_p_g_0, raw_K_p_g_1, raw_mu_g, raw_sigma_g = RBF_p_diag(sigma, length_scale, std)

    B, omega_0_g, omega_1_g, omega_2_g, kappa_0_g, kappa_1_g, kappa_2_g = coregion(omega, kappa)

    C = numpy.kron(K_p.ravel().reshape(-1, 1)+jitter, B)

    length_scale_g = numpy.kron(raw_K_p_g_0.ravel().reshape(-1, 1), B)

    std_g = numpy.kron(raw_K_p_g_1.ravel().reshape(-1, 1), B)

    mu_g = numpy.kron(raw_mu_g.ravel().reshape(-1, 1), B)

    sigma_g = numpy.kron(raw_sigma_g.ravel().reshape(-1, 1), B)

    omega_0_g = numpy.kron(K_p.ravel().reshape(-1, 1), omega_0_g)

    omega_1_g = numpy.kron(K_p.ravel().reshape(-1, 1), omega_1_g)

    omega_2_g = numpy.kron(K_p.ravel().reshape(-1, 1), omega_2_g)

    kappa_0_g = numpy.kron(K_p.ravel().reshape(-1, 1), kappa_0_g)

    kappa_1_g = numpy.kron(K_p.ravel().reshape(-1, 1), kappa_1_g)

    kappa_2_g = numpy.kron(K_p.ravel().reshape(-1, 1), kappa_2_g)

    return C, [length_scale_g, std_g,
               omega_0_g, omega_1_g, omega_2_g, kappa_0_g, kappa_1_g, kappa_2_g, mu_g, sigma_g]


def kernel_test(theta, mu, sigma, mu_train, sigma_train):

    length_scale = theta[0]

    std = theta[1]

    omega = numpy.reshape(theta[2:5], (3, 1))

    kappa = theta[5:8]

    K_p, raw_K_p_g_0, raw_K_p_g_1, raw_mu_g, raw_sigma_g = RBF_p_test(mu, sigma, mu_train, sigma_train, length_scale, std)

    B, omega_0_g, omega_1_g, omega_2_g, kappa_0_g, kappa_1_g, kappa_2_g = coregion(omega, kappa)

    C = numpy.kron(K_p, B)

    length_scale_g = numpy.kron(raw_K_p_g_0, B)

    std_g = numpy.kron(raw_K_p_g_1, B)

    mu_g = numpy.kron(raw_mu_g, B)

    sigma_g = numpy.kron(raw_sigma_g, B)

    omega_0_g = numpy.kron(K_p, omega_0_g)

    omega_1_g = numpy.kron(K_p, omega_1_g)

    omega_2_g = numpy.kron(K_p, omega_2_g)

    kappa_0_g = numpy.kron(K_p, kappa_0_g)

    kappa_1_g = numpy.kron(K_p, kappa_1_g)

    kappa_2_g = numpy.kron(K_p, kappa_2_g)

    return C, [length_scale_g, std_g,
               omega_0_g, omega_1_g, omega_2_g, kappa_0_g, kappa_1_g, kappa_2_g, mu_g, sigma_g]


def kernel(theta, mu, sigma, jitter=1e-2):

    length_scale = theta[0]

    std = theta[1]

    omega = numpy.reshape(theta[2:5], (3, 1))

    kappa = theta[5:8]

    K_p, raw_K_p_g_0, raw_K_p_g_1, raw_mu_g, raw_sigma_g = RBF_p(mu, sigma, length_scale, std)

    B, omega_0_g, omega_1_g, omega_2_g, kappa_0_g, kappa_1_g, kappa_2_g = coregion(omega, kappa)

    C = numpy.kron(K_p, B) + numpy.eye(3*len(mu))*jitter

    length_scale_g = numpy.kron(raw_K_p_g_0, B)

    std_g = numpy.kron(raw_K_p_g_1, B)

    mu_g = numpy.kron(raw_mu_g, B)

    sigma_g = numpy.kron(raw_sigma_g, B)

    omega_0_g = numpy.kron(K_p, omega_0_g)

    omega_1_g = numpy.kron(K_p, omega_1_g)

    omega_2_g = numpy.kron(K_p, omega_2_g)

    kappa_0_g = numpy.kron(K_p, kappa_0_g)

    kappa_1_g = numpy.kron(K_p, kappa_1_g)

    kappa_2_g = numpy.kron(K_p, kappa_2_g)

    return C, [length_scale_g, std_g,
               omega_0_g, omega_1_g, omega_2_g, kappa_0_g, kappa_1_g, kappa_2_g, mu_g, sigma_g]


def RBF_p_test(mu, sigma, mu_train, sigma_train, length_scale, std):

    if (numpy.max(sigma_train) - numpy.min(sigma_train)) == 0:
        sigma_train = sigma_train * 0
        sigma = sigma * 0

    n_train = numpy.shape(mu_train)[0]

    n_test = numpy.shape(mu)[0]

    sigma2 = sigma ** 2

    sigma2_train = sigma_train ** 2

    mu_1 = mu_train.repeat(n_test, axis=1)

    mu_1 = numpy.vstack([mu_1, mu.transpose()])

    sigma_1 = sigma_train.repeat(n_test, axis=1)

    sigma_1 = numpy.vstack([sigma_1, sigma.transpose()])

    sigma2_1 = sigma2_train.repeat(n_test, axis=1)

    sigma2_1 = numpy.vstack([sigma2_1, sigma.transpose()])

    mu_2 = mu.transpose().repeat(n_train + 1, axis=0)

    sigma_2 = sigma.transpose().repeat(n_train + 1, axis=0)

    sigma2_2 = sigma2.transpose().repeat(n_train + 1, axis=0)

    S = (sigma2_1 + sigma2_2 + (length_scale ** 2))

    K = length_scale * (S ** -0.5) * numpy.exp(-0.5 * ((mu_1 - mu_2) ** 2) / S) * (std ** 2)

    length_scale_g = (std ** 2 * numpy.exp(-(mu_1 - mu_2) ** 2 / (2 * (length_scale ** 2 + sigma2_1 + sigma2_2))) *
                      (length_scale ** 2 * mu_1 ** 2 - 2 * length_scale ** 2 * mu_1 * mu_2 +
                       length_scale ** 2 * mu_2 ** 2 + length_scale ** 2 * sigma2_1 + length_scale ** 2 * sigma2_2 +
                       sigma2_1 ** 2 + 2 * sigma2_1 * sigma2_2 + sigma2_2 ** 2)) / (length_scale ** 2
                                                                                    + sigma2_1 + sigma2_2) ** (5 / 2)

    std_g = 2 * K / std

    mu_g = -(length_scale * std ** 2 * numpy.exp(
        -(mu_1 - mu_2) ** 2 / (2 * (length_scale ** 2 + sigma2_1 + sigma2_2))) *
             (2 * mu_1 - 2 * mu_2)) / (2 * (length_scale ** 2 + sigma2_1 + sigma2_2) ** (3 / 2))

    sigma_g = -(length_scale * sigma_1 * std ** 2 * numpy.exp(
        -(mu_1 - mu_2) ** 2 / (2 * (length_scale ** 2 + sigma_1 ** 2 + sigma_2 ** 2)))
                * (length_scale ** 2 - mu_1 ** 2 + 2 * mu_1 * mu_2 - mu_2 ** 2 + sigma_1 ** 2 + sigma_2 ** 2)) / (
                          length_scale ** 2 +
                          sigma_1 ** 2 +
                          sigma_2 ** 2) ** (5 / 2)

    if (numpy.max(sigma) - numpy.min(sigma)) == 0:
        sigma_g = sigma_g * 0

    return K, length_scale_g, std_g, mu_g, sigma_g


def RBF_p_diag(sigma, length_scale, std):

    if (numpy.max(sigma) - numpy.min(sigma)) == 0:
        sigma = sigma * 0

    sigma2 = sigma ** 2

    S = (sigma2 + sigma2 + (length_scale ** 2))

    K = length_scale * (S ** -0.5) * (std ** 2)

    length_scale_g = (std ** 2 * (sigma2 + sigma2)) / (length_scale ** 2 + sigma2 + sigma2) ** (3 / 2)

    std_g = 2 * K / std

    sigma_g = -(length_scale*sigma*std**2)/(length_scale**2 + sigma**2 + sigma**2)**(3/2)

    mu_g = numpy.zeros_like(sigma_g)

    if (numpy.max(sigma) - numpy.min(sigma)) == 0:
        sigma_g = sigma_g * 0

    return K, length_scale_g, std_g, mu_g, sigma_g


def RBF_p(mu, sigma, length_scale, std):
    # mu and sigma are column vector

    if (numpy.max(sigma) - numpy.min(sigma)) == 0:
        sigma = sigma * 0

    sigma2 = sigma ** 2

    n = numpy.shape(mu)[0]

    mu_1 = mu.repeat(n, axis=1)

    sigma_1 = sigma.repeat(n, axis=1)

    sigma2_1 = sigma2.repeat(n, axis=1)

    mu_2 = mu.transpose().repeat(n, axis=0)

    sigma2_2 = sigma2.transpose().repeat(n, axis=0)

    sigma_2 = sigma.transpose().repeat(n, axis=0)

    S = (sigma2_1 + sigma2_2 + (length_scale ** 2))

    K = length_scale * (S ** -0.5) * numpy.exp(-0.5 * ((mu_1 - mu_2) ** 2) / S) * (std ** 2)

    length_scale_g = (std ** 2 * numpy.exp(-(mu_1 - mu_2) ** 2 / (2 * (length_scale ** 2 + sigma2_1 + sigma2_2))) *
                      (length_scale ** 2 * mu_1 ** 2 - 2 * length_scale ** 2 * mu_1 * mu_2 +
                       length_scale ** 2 * mu_2 ** 2 + length_scale ** 2 * sigma2_1 + length_scale ** 2 * sigma2_2 +
                       sigma2_1 ** 2 + 2 * sigma2_1 * sigma2_2 + sigma2_2 ** 2)) / (length_scale ** 2
                                                                                    + sigma2_1 + sigma2_2) ** (5 / 2)

    std_g = 2 * K / std

    mu_g = -(length_scale*std**2*numpy.exp(-(mu_1 - mu_2)**2/(2*(length_scale**2 + sigma2_1 + sigma2_2))) * \
             (2*mu_1 - 2*mu_2))/(2*(length_scale**2 + sigma2_1 + sigma2_2)**(3/2))

    sigma_g = -(length_scale*sigma_1*std**2*numpy.exp(-(mu_1 - mu_2)**2/(2*(length_scale**2 + sigma_1**2 + sigma_2**2)))
                * (length_scale**2 - mu_1**2 + 2*mu_1*mu_2 - mu_2**2 + sigma_1**2 + sigma_2**2))/(length_scale**2 +
                                                                                                  sigma_1**2 +
                                                                                                  sigma_2**2)**(5/2)

    if (numpy.max(sigma) - numpy.min(sigma)) == 0:
        sigma_g = sigma_g * 0

    return K, length_scale_g, std_g, mu_g, sigma_g


def coregion(omega, kappa):

    B = numpy.dot(omega, omega.transpose())

    B = B + numpy.diag(kappa.ravel())

    B_g = numpy.zeros([3, 3, 3])

    for i in range(0, 3):
        B_g[i, :, i] = B_g[i, :, i] + omega.ravel()

        B_g[i, i, :] = B_g[i, i, :] + omega.ravel()

    omega_0_g = B_g[0, :, :]

    omega_1_g = B_g[1, :, :]

    omega_2_g = B_g[2, :, :]

    B_g = numpy.zeros([3, 3, 3])

    for i in range(0, 3):
        B_g[i, i, i] = 1

    kappa_0_g = B_g[0, :, :]

    kappa_1_g = B_g[1, :, :]

    kappa_2_g = B_g[2, :, :]

    return B, omega_0_g, omega_1_g, omega_2_g, kappa_0_g, kappa_1_g, kappa_2_g


A_u_grad = autograd.jacobian(get_sample_w_step, 0)

Q_w_grad = autograd.jacobian(get_sample_w_step, 1)

C_wu_grad = autograd.jacobian(get_sample_w_step, 2)

L_u_grad = autograd.jacobian(get_Q_w, 0)

C_u_grad = autograd.jacobian(get_Q_w, 1)

C_wu_grad_Q_w = autograd.jacobian(get_Q_w, 2)

C_diag_w_grad = autograd.jacobian(get_Q_w, 3)

get_mc_link_g = autograd.jacobian(mc_link_lik, 0)

get_prior_link_g = autograd.jacobian(mc_link_lik, 1)

get_kl_g = autograd.jacobian(get_noraml_kl, 0)

get_kl_g_C_u = autograd.jacobian(get_noraml_kl, 1)

get_mu_shift_g = autograd.jacobian(mc_link_lik, 1)

