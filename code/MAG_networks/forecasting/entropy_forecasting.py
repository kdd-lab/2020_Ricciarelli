import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

s = {'100017046': {
     '2002': {'entropy': -0.61, 'class': 'different',
              'affiliation': 'Italy'},
     '2003': {'entropy': -0.67, 'class': 'different',
              'affiliation': 'Italy'},
     '2004': {'entropy': -0.68, 'class': 'different',
              'affiliation': 'Italy'},
     '2005': {'entropy': -0.76, 'class': 'different',
              'affiliation': 'Italy'},
     '2006': {'entropy': -0.63, 'class': 'different',
              'affiliation': 'Italy'},
     '2007': {'entropy': -0.74, 'class': 'different',
              'affiliation': 'Italy'},
     '2008': {'entropy': -0.6, 'class': 'different',
              'affiliation': 'Italy'},
     '2009': {'entropy': -0.58, 'class': 'different',
              'affiliation': 'Italy'},
     '2010': {'entropy': -0.57, 'class': 'different',
              'affiliation': 'Italy'},
     '2011': {'entropy': -0.58, 'class': 'different',
              'affiliation': 'Italy'},
     '2012': {'entropy': -0.5, 'class': 'different',
              'affiliation': 'Italy'},
     '2013': {'entropy': -0.49, 'class': 'different',
              'affiliation': 'Italy'},
     '2014': {'entropy': -0.57, 'class': 'different',
              'affiliation': 'Italy'},
     '2015': {'entropy': -0.59, 'class': 'different',
              'affiliation': 'Italy'},
     '2016': {'entropy': -0.56, 'class': 'different',
              'affiliation': 'Italy'},
     '2017': {'entropy': -0.53, 'class': 'different',
              'affiliation': 'Italy'},
     '2018': {'entropy': -0.44, 'class': 'different',
              'affiliation': 'Italy'}}}

plt.rcParams.update({'font.size': 8})

entropies = None

for s_id in s:
        entropies = [[s[s_id][year]['entropy']for year in sorted(s[s_id])]][0]

entropies = np.array(entropies)

fig, axes = plt.subplots(3, 2, sharex=True, constrained_layout=True)

for x in np.arange(0, len(axes)):
    axes[x][0].plot(np.diff(entropies, n=x))
    axes[x][0].set_title('Original Series' if x == 0 else
                         '{}st Order Differencing'.format(x))
    plot_acf(entropies if x == 0 else np.diff(entropies, n=x), ax=axes[x][1],
             alpha=0.05)

fig.savefig('../images/forecasting/autocorrelation.pdf',
            format='pdf', bbox_inches='tight')

plt.close(fig)

fig, axes = plt.subplots(3, 2, sharex=True, constrained_layout=True)

for x in np.arange(0, len(axes)):
    axes[x][0].plot(np.diff(entropies, n=x))
    axes[x][0].set_title('Original Series' if x == 0 else
                         '{}st Order Differencing'.format(x))
    plot_pacf(entropies if x == 0 else np.diff(entropies, n=x), ax=axes[x][1],
              alpha=0.05)

fig.savefig('../images/forecasting/partial_autocorrelation.pdf',
            format='pdf', bbox_inches='tight')

train, test = entropies[:13], entropies[13:]

model = pm.auto_arima(train, start_p=1, d=None, start_q=1, max_p=3,
                      max_d=3, max_q=3, information_criterion='aic',
                      test='adf', error_action='ignore',
                      suppress_warnings=True)

predictions = model.predict(n_periods=8)

fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(np.arange(2002, 2015), train, linewidth=2, color='#4682b4',
        label='Train Set')
ax.plot(np.arange(2015, 2019), test, linewidth=2, color='#b44682',
        label='Test Set')
ax.plot(np.arange(2015, 2023), predictions, linewidth=2, color='#82b446',
        label='Prediction')
ax.set_xlim(2000, 2023)
ax.set_ylim(-1.0, -0.4)
ax.set_xticks(np.arange(2000, 2023, 5))
ax.set_xticks(np.arange(2000, 2023), minor=True)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_title('Example of Prediction', fontsize=10)
ax.set_xlabel('Year', fontsize=8)
ax.set_ylabel('Entropy', fontsize=8)
ax.legend()

fig.savefig('../images/forecasting/prediction_example.pdf', format='pdf')
