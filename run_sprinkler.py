""" script to run the sprinkler model and display the results"""

import pymc as mc
import pylab as pl
import pandas

import sprinkler
reload(sprinkler)

## fit the model with MCMC
m = mc.MCMC(sprinkler)
m.sample(100000, 500)


## report the results
print 'Pr[R|G] =', m.R.stats()['mean'].round(3)
print 'MCMC Error:', m.R.stats()['mc error'].round(4)
print
print


## confirm that the conditional probability of each node of the model
# is correct
p_rain = pandas.DataFrame(index=[0])
p_sprinkler = pandas.DataFrame()
p_grass_wet = pandas.DataFrame()
for R_val in [True, False]:
    m.R.value = R_val
    p_rain[R_val] = pl.exp(m.R.logp)

    p_sprinkler_i = pandas.DataFrame(dict(rain=[R_val]))
    for S_val in [True, False]:
        m.S.value = S_val
        p_sprinkler_i[S_val] = pl.exp(m.S.logp)

        p_grass_wet_i = pandas.DataFrame(index=['i'])
        p_grass_wet_i['sprinkler'] = [S_val]
        p_grass_wet_i['rain'] = [R_val]
        p_grass_wet_i[True] = [m.p_G.value]
        p_grass_wet_i[False] = [1 - m.p_G.value]
        p_grass_wet = p_grass_wet.append(p_grass_wet_i, ignore_index=True)

    p_sprinkler = p_sprinkler_i.append(p_sprinkler, ignore_index=True)

p_grass_wet = p_grass_wet.sort('rain').sort('sprinkler')

print 'Pr[R] = '
print p_rain

print 'Pr[S|R] = '
print p_sprinkler

print 'Pr[G|S,R] = '
print p_grass_wet

m.R.value = True  # make sure that the model is not left in an impossible state


## explore the error in the estimation as a function of iter and thin
results = pandas.DataFrame()
for iter in [1e2, 1e3, 1e4, 1e5]:
    for thin in [1, 10, 100]:
        for rep in range(10):
            if iter/thin >= 100:
                m = mc.MCMC(sprinkler)
                m.sample(iter=iter, thin=thin)

                stats = m.R.stats()
                results = results.append(pandas.DataFrame(
                        dict(iter=[iter], thin=[thin], rep=[rep],
                             p_R=[stats['mean']], mc_error=[stats['mc error']])),
                                         ignore_index=True)
