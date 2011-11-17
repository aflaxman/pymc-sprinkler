import pylab as pl
import pymc as mc

G_obs = [1.]
N = len(G_obs)

R = mc.Bernoulli('R', .2, value=pl.ones(N))

p_S = mc.Lambda('p_S', lambda R=R: pl.where(R, .01, .4),
                doc='Pr[S|R]')
S = mc.Bernoulli('S', p_S, value=pl.ones(N))

p_G = mc.Lambda('p_G', lambda S=S, R=R: pl.where(S, pl.where(R, .99, .9), pl.where(R, .8, 0.)),
                doc='Pr[G|S,R]')
G = mc.Bernoulli('G', p_G, value=G_obs, observed=True)
