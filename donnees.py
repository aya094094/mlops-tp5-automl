import numpy as np
import pandas as pd
rng = np.random.default_rng(7) # graine fixe : meme jeu pour toute la promo
n = 5000 # nombre de transactions
montant = np.round(rng.lognormal(6.2, 1.1, n), 2) # beaucoup de petits montants
anciennete = rng.integers(0, 120, n) # anciennete du client en mois
freq = rng.poisson(3, n) + 1 # transactions sur 24 h
nuit = rng.binomial(1, 0.18, n) # paiement entre 0 h et 5 h
score = 2.2*(np.log10(montant)-3.0) - anciennete/60 + (freq-3)/3 + nuit*1.2
fraude = ((score + rng.normal(0, 0.35, n)) > 0.6).astype(int) # etiquette 0 / 1
df = pd.DataFrame({'montant': montant, 'anciennete': anciennete,
 'freq_24h': freq, 'nuit': nuit, 'fraude': fraude})
df.to_csv('data/transactions.csv', index=False)
print('data/transactions.csv ecrit :', n, 'lignes,', fraude.sum(), 'fraudes')
