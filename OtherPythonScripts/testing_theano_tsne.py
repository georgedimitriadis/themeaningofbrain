
import os
import numpy as np
from scipy.optimize import *
import matplotlib.pyplot as plt
import theano.tensor as tt
import theano
import time
from sklearn.manifold import TSNE as tsne
os.environ['BREZE_PARAMETERSET_DEVICE'] = 'cpu'
from breze.learn.tsne import Tsne as tsne_breze

np.random.seed(0)

D_in = 100
D_out = 2
perplexity_target = 50
pts = np.vstack(
    [np.random.multivariate_normal(np.zeros((D_in,)), 0.01*np.eye(D_in), size=100)] +
    [np.random.multivariate_normal(np.eye(D_in)[i], 0.01*np.eye(D_in), size=100) for i in range(D_in)]
    ).astype(np.float32)








perplexity = perplexity_target
early_exaggeration_float = 1.0
learning_rate = 100
model = tsne(n_components=D_out, perplexity=perplexity, early_exaggeration=early_exaggeration_float,
             learning_rate=learning_rate, metric='euclidean',
             random_state=None, init='random', verbose=10)
t0 = time.time()
yf = model.fit_transform(pts)
t1 = time.time()
print("Sklearn t-sne took {} seconds".format(t1 - t0))




early_exaggeration_int = 50
t0 = time.time()
model = tsne_breze(n_inpt=pts.shape[1], n_lowdim=2, perplexity=perplexity_target,
                   early_exaggeration=early_exaggeration_int, max_iter=300, verbose=True)
yf = model.fit_transform(pts)
t1 = time.time()
print("Breze t-sne took {} seconds".format(t1 - t0))




N = pts.shape[0]
X = theano.shared(pts, name="X")
Y = tt.matrix("Y")
sigma = tt.vector("sigma")

pdist2 = lambda A : ((A[:, np.newaxis, :] - A[np.newaxis, :, :])**2).sum(2)

pdist_X = pdist2(X)
pdist_Y = pdist2(Y)

P_c = tt.exp(-pdist_X / (2 * sigma**2)) / tt.exp(-pdist_X / (2 * sigma**2)).sum(1)[:, np.newaxis]
P = (P_c + P_c.T) / (2*N)
Q = (1/(1+pdist_Y)) / (1/(1+pdist_Y)).sum()
KL = tt.where(abs(P) > 1e-8, P * tt.log(P / Q), 0).sum(1)
C = KL.sum()
LogPerplexity = -tt.where(abs(P_c) > 1e-8, P_c*tt.log2(P_c), 0).sum(1)
PerplexityCost = 0.5*((LogPerplexity - np.log2(perplexity_target))**2).sum()

#### Sigma
s0 = np.ones(N, np.float32)
I = tt.iscalar("I")
prp = theano.function([sigma, I], LogPerplexity[I], allow_input_downcast=True)
t0 = time.time()
print("Init PC:", PerplexityCost.eval({sigma:s0}))
t1 = time.time()
print("Init PC calculation took %f seconds" % (t1 - t0))
t0 = time.time()
for i in range(N):
    f = lambda s : (prp(s*np.ones(N), i) - np.log2(perplexity_target))
    s0[i] = brentq(f, 1e-6, 10, rtol=1e-8)
t1 = time.time()
print("Perplexity calculation took %f seconds" % (t1 - t0))
t0 = time.time()
print("Final PC:", PerplexityCost.eval({sigma:s0}))
t1 = time.time()
print("Final PC calculation took %f seconds" % (t1 - t0))

#### Y
f_g = theano.function([Y, sigma], [C, tt.grad(C, Y)])
def f_and_g(y):
    f, g = f_g(y.astype(np.float32).reshape((N, D_out)), s0)
    return f, g.astype(np.float64).reshape(-1)

from sklearn.decomposition import TruncatedSVD
y0 = TruncatedSVD(n_components=D_out).fit_transform(pts)
#y0 = np.random.rand(N, D_out)
t0 = time.time()
yf = fmin_l_bfgs_b(f_and_g, y0, disp=1, maxfun=200)[0]
t1 = time.time()
print("Minimizing calculation took %f seconds" % (t1 - t0))
yf = yf.reshape(N, D_out)





for i, pp in enumerate(np.split(yf, D_in+1, axis=0)):
    plt.scatter(pp[:,0], pp[:,1], color=plt.cm.Paired(i/(D_in+1)), label=str(i))
plt.show()