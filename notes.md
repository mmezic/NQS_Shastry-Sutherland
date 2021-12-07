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
- [x] vypnout `TOTAL_SZ` a zkusit `MetropolisLocal`
- [ ] Jastrow
- [x] rozštěpení na amplitudu a fázi

## poznámky
- výsledky mřížky `N=36`:
    - pro `TOTAL_SZ=0` a `MetropolisExchange` to hlouběji v DS a AF fázích konverguje relativně OK, ale v místě fázového přechodu to NEKONVERGUJE VŮBEC i když dám hodně moc samplů (1000) a iterací (1800) (viz. chaotický grav - v něm je ještě m_s a m_p pro MSR počítáno špatně)
    - v oblasti toho fázového přechodu jsem to projel podrobněji, ale moc to nepomohlo
- vypnuté `TOTAL_SZ` spolu s `MetropolisLocal` dává docela podobné výsledky jako `MetropolisExchange`
    - pro `N=16` je dobře vidět, že MSR báze konverguje v Neélově fázi a normální báze konverguje v dimerové fázi
- rozštěpení na amplitudu a fázi je předimplementované v modulu `RBMModPhase`, ale zatím mi to konverguje jenom někde
    - `DTYPE` obou RBMek je `np.float64`
    - η amplitudy 0 -> 0.01
    - η fáze      0.05 -> 0.01 ... stejně jako v tutoriálu netket3
    - asi si budu muset ještě pohrát s η a nebo přidat symmetrie

## Schůzka 26.11.2021
 - [x] 1) napočítat počty automorfismů pro 1D mřížku J1-J2 a zkontrolovat, jestli to sedí se skutečnými symetriemi
    - zjistit, kde je problém v RBMsymm
        - buď je problém v automorfismech (tj. symetrická báze není vlastní podprostor H)
        - nebo přímo v definici RBMsymm, jak to aplikuje
 - [ ] 2) udělat grid-search na metacentru pro RBMsymm (variovat ALPHA a ETA)
 - [ ] udělat grid-search na metacentru pro RBMmodPhase (variovat všechny možné hyperparametry)
    - udělat amplitudu symetrickou
 - [x] zdvojnásobit počet samplů (např 1500 samples) a nechat běžet dlouho na metacentru (několik dní - podívat se, kolik běželo předtím těch málo samplů, abych věděl, kolik tomu dát času)
 - [x] znovu proběhnout všechny mřížky (i 36 + zvýšit počet MC kroků) s RBM (ne-sym!) zatím bez rozdělením na modPhase
 

 ## poznámky
 1) funkce `automorphisms()` funguje správně - když přebarvím J_2
    - až na to, že nedokáže rozlišit mezi tím, když je stejná hrana vícenásobná
    - používají na to funkci: https://igraph.org/python/api/latest/igraph._igraph.GraphBase.html#get_isomorphisms_vf2

2) mám to ve složce SS_model/energy_plots/GrodSearch-symmRBM, ale ještě jsem to nezvizualizoval a nezanalyzoval

 - RBM (ne-symm) to počítá opravdu hodně rychleji a lépe
 - udělal jsem grid-search (v `eta` a `alpha`) pro RBMsymm v dimerové fázi (J2=0.3)
    - pro N=4 to NIKDY nenašlo gs (viz data ve složce GridSearch-symmRBM)
    - pro osatní to jsem submitnul na metacentrum
 - v nové verzi `3.2` netketu byla do `Heisenberg` přidána podpora různých kaplovacích konstant pro různé barvy hran grafu
    - a taky implementovalý nějaký Parallel Tempering sampler (nevím, jestli může být užitečný)

- Jaký mám použít Jastrow?
    - zkoušel jsem $ \log\psi(\sigma) = \sum_i a_i \sigma_i + \sum_{i,j} \sigma_i J_{i,j} \sigma_j $ - na malých mřížkách to fungovalo, ale pro N=36 to nenašlo dimerovou fázi, tak jsem to vzdal...

## další poznámky

- <a href="https://www.physicsforums.com/threads/when-does-a-wavefunction-inherit-the-symmetries-of-the-hamiltonian.643672/">link</a> In general the whole eigensubspace will be invariant under any transformation that leaves the hamiltonian invariant. For a one dimensional ground state that means you have invariance up to a phase factor.
- Jaktože Carleo nepoužíval MSR a přesto mu to našlo GS, který nesplňuje symetrie

## schůzka 3.12

- promyslet si jak naimplementovat symmetrie
- [ ] udělat dimer pomocí G-CNN - dokáží ho vyřešit?
- [ ] podívat se, jaké symmetrie v RBM jdou vypnout a zkusit si zadefinovat vlastní symetrie
    - zkusit udělat pouze **translace o 2**
- článek od sandvika (kniha ) - rozepsat si to pro dimer a promyslet si jestli/proč $K \neq 0$
- [ ] zkusit udělat dimerovou fázi v 1D a 2D
    - najít základní stav a podívat se, jaké je $K$ ?
    <!-- - symmetrie pouze translace -->

## poznámky
- vypnul jsem všechny ostatní symetrie a nechal jsem jenom translace o 2 a pořád mi to nekonverguje
- [ ] udělat si testy na dvou-dimerovém toy modelu

### one dimer

A dimer is described by a hamiltonian
$$ H_{\text{one}} = \bm{S}_1\cdot\bm{S}_2 = \begin{pmatrix} 
1 & 0 & 0 & 0\\ 
0 &-1 & 2 & 0\\
0 & 2 &-1 & 0\\
0 & 0 & 0 & 1
\end{pmatrix}$$