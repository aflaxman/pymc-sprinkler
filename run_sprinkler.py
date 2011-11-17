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
for iter in [1e2, 1e3, 1e4, 1e5, 1e6]:
    for rep in range(10):
        m = mc.MCMC(sprinkler)
        m.sample(iter=iter)

        stats = m.R.stats()
        results = results.append(pandas.DataFrame(
                dict(iter=[iter], rep=[rep],
                     p_R=[stats['mean']],
                     p_R_lb=[stats['95% HPD interval'][0]],
                     p_R_ub=[stats['95% HPD interval'][1]],
                     mc_error=[stats['mc error']])),
                                 ignore_index=True)

results['true'] = (.00198 + .1584) / (.00198 + .288 + .1584 + 0.)
results['err'] = results['true'] - results['p_R']
results['abs_err'] = pl.absolute(results['err'])
results['covered'] = (results['p_R_lb'] <= results['true']) & (results['p_R_ub'] >= results['true'])
results['covered'] = 1.*results['covered']
results['thin'] = 1

results_summary = results.groupby(['iter', 'thin']).describe().unstack()
print 'median bias:'
print results_summary['err', '50%']
print

print 'median absolute error (and mc error):'
print results_summary.ix[:, [('abs_err', '50%'), ('mc_error', '50%')]]
print

print 'percent covered:'
print results_summary['covered', 'mean']
print
