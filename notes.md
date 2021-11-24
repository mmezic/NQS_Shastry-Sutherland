## Schůzka 18.11.2021
- https://www.netket.org/docs/_generated/samplers/netket.sampler.MetropolisExchange.html
    - n_chains
- ve .init_parameters by neměl být seed
- změna v SS_MSR na součet
- zkusit udělat `MetropolisLocal` namísto `MetropolisExchange` pro malé mřížky
    - při `TOTAL_SZ = None` v dimerové fázi
    - podaří se to zkonvergovat i v dimerové fázi?
- zkusit Jastrow ansatz na 6*6
- zkusit projet celý interval (pro N = 20) ještě jednou pomocí ED s `TOTAL_ST=0` a ověřit, že to dává stejné výsledky
    - možná to nebude třeba, protože Qspin to spočítal stejně
- rozdělit na amplitudu a fázi a 2*RBM (viz. tutorial netket3 9.)

### Úkoly:
- [x] znovu přegenerovat grafy se správným výpočtěm parametrů uspořádání pro MSR
- [ ] vypnout `TOTAL_SZ` a zkusit `MetropolisLocal`
- [ ] Jastrow
- [ ] rozštěpení na amplitudu a fázi

## poznámky
- výsledky mřížky `N=36`:
    - pro `TOTAL_SZ=0` a `MetropolisExchange` to hlouběji v DS a AF fázích konverguje relativně OK, ale v místě fázového přechodu to NEKONVERGUJE VŮBEC i když dám hodně moc samplů (1000) a iterací (1800) (viz. chaotický grav - v něm je ještě m_s a m_p pro MSR počítáno špatně)
    - v oblasti toho fázového přechodu jsem to projel podrobněji, ale moc to nepomohlo
- vypnuté `TOTAL_SZ` spolu s `MetropolisLocal` dává docela podobné výsledky jako `MetropolisExchange`
    - pro `N=16` je dobře vidět, že MSR báze konverguje v Neélově fázi a normální báze konverguje v dimerové fázi
