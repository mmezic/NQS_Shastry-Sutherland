# My ISDONE list
### abstract
### úvod
- [x] je potřeba silnější motivaci
    - [x] proč ML? ... podle článku od gcarlea popisujícího generalitu ML
    - [x] proč SSM? ... má 2 úplně rozdílné fáze -> dobrý testbed pro generalizační schopnosti ML
    - [x] nikdo nezkoušel ML na SSM
    - [x] všichni používají $H_z = 0$ !!
### 1.kapitola
- [x] obrázky z arxivu můžu asi volně zkopírovat fo kap 2.2 `getry-data` program
    - [x] buď naskenovat obrázek z nějakého článku nebo nakreslit vlastní sketch
        - **¿¿** experimentální hosnoty nebo ze simulace **??**
        - zkopíroval jsem obrázek ze SciPostu
- [x] nemá být AF order parameter přenásobený čtyřmi?
    - asi je OK, že nevychází 1 (viz. fig5 v Yang,Sandvik)
        - určitě je to OK, protože 1 by to byla jen v čisté AF fází; my ale máme J=0.9
### 2.kapitola
- [x] sign problem
    - [x] :warning: pochopil jsem to dobře?! :warning:
- [x] volba trial funkce vlastně určije volbu podprostoru funkcí v Hilbertova prostoru
### 3.kapitola
### 4.kapitola
- [x] udělat obrázek popisující průměrování energií z posledních 100 iterací a záporné hodnoty zpsůobené asi numerickou přesnotsí
- [x] pod fig 4.1 (¿je to v MSR?) přidat další graf, kde bude 10 000 samplů
    + finite size scaling ... jaké používat errorbary? ... jejich velikost totiž nesouvisí s η, ale s #samples
    - Podle [knížky](G:/My%20Drive/!MFF/!!Diplomka/literatura/Federico%20Becca%20-%20Quantum%20Monte%20Carlo%20Approaches%20for%20Correlated%20Systems-Cambridge%20University%20Press%20(2017).pdf) kapitola 3.11 by mohlo být ok použít místo errorbaru varianci za posledních ~50 kroků.
    - snad to řeší přídání grafu s porovnáním samplů - je to ok pro max 3000?
- [x] dopsat do práce, že jsme narazili na problém s konvergencí bez viditelných biasů, ale bylo to způsobeno pouze moc malou inicializací - z toho můžeme usuzovat, že visible biasy mohou pomáhat při konvergenci v těžkých chvílích
- [x] přidat poznámku, že errorbal z MC (probable error) není úplně směrodatný
### 5.kapitola
#### transfer learning
- [x] přidat dogenerovaný graf na širším rozsahu (běží na metacentru 10959130)
- [x] znovu vygenerovat výsledky pro RBM16, aby tam byly i errorbary avg50
    - už to běží na Barborce
- [x] připsat obkec
### 6. kapitola
- [x] **opravit šilené errorbary spuštěním znovu s malým learníng ratem** z konečného stavu, kde mi to (ne) zkonvergovalo
- [x] přidat obkec
- [x] říct, že pro mag pole nemusí být nejlepší ten model, který vyšel nejlépe na 4x4 RBM v tabulce
### závěr
### Appendix A
- [x] přidat podrobnou benchmarkingovou tabulku včetně běhu se zvýšeným počtem samplů
### Appendix B
### other stuff
- [x] **spustit graf (ve kterém je gap) s Metropolisem, a podívat se, jestli tam bude gap** abychom zjistili, jestli není problém v Exact samplingu od Netketu
    - metropolis má tak velký rozptyl oproti exact sampleru, že z toho nejde vůbec vykoukat informace o gapu
    - gap tam vskutku není
- [x] doplnit bod $N=20$ pomocí ED

TODO:
- [x] dogenerovat tabulku...
- [ ] ... a vybrat model, který pustím na případu s mag polem (na $H_Z$)
- [x] zkusit dva modely z tabulky pustit s SAMPLES=5000 (do appendixu) a podívat se, jak hodně to zlepší accuracy
- [x] do tabulky už nepřidávat zmenšování learning ratu
- [x] SAMPLES mění finální rozptyl nad exaktní energií


# My TODO list


### abstract
- [ ] ne až tak technický
- [x] napsat jej jako příběh
### úvod
- [x] připsat cíle práce (asi až nakonci)
### 1. kapitola
- [x] doplnit popis order parametrů
    - [x] odkaz na článek Yang Sandvik: *The inset of Fig. 5 shows how PS order is stabilized only for the larger system sizes inside the PS phase, reflecting large fluctuations in small systems (as shown explicitly in Supplemental Material [28]).*
    - [x] viz komentare od Zondy
- [ ] radši ještě jednou zkontrolovat faktor v definici $\hat{H}$
### 2. kapitola
- [x] přidat popis Jastrowa a popsat na něm VMC (přesnost závisí na kvalitě trial funkce)
    - stačí takto stručný popis?

- [ ] zkusit otestovat, jestli RBM = Jastrow po Hubbard-Stratonowith transformaci -> pak by se to dalo použít jako motivace pro RBM
### 3. kapitola
- [ ] připsat popis fyzikální interpretace hidden vrstvy
- [x] zdůraznit v textu věci, které jsem udělal sám
- [ ] sjednotit popis MSR s kapitolou 1
### 4. kapitola
- [ ] alphas comparison
### 5. kapitola
- [ ] opravit vystřelený bod v grafu
- [x] přidat graf total_sz = 0
    - [ ] vygenerovat do něj další body v okolí J/J' = 0.8
- [ ] do finite-size scaling přidat (extrapolating) lines to guide the eye 
### 6. kapitola

### závěr
- [ ] future outlook
    - [ ] použití symetrií na SSM h=0 se správnými charaktery, abychom zjistili, jestli dokážou efektivněji popsat tyto tři fáze
    - [ ] vytvoření lepší architektury na zkoumání SSM h>0 (zjistit, proč to nedokáže dobře odchytit stavy s vyšším h) -- případně by tam mohli hrát důležitou roli biasy, které mají význam mag pole
- [ ] zmínit, že bychom mohli použít všechny irreps a pak vybrat tu s nejnižsí energií
- [ ] reformulate the two consecutive opposing paragraphs about transfer learning

### Appendix A
- [ ] přidat obkec
### Appendix B
- [ ] jak přidat do appendixu kód?
## Další věci
- [ ] rozebrat ve finite-size scaling, podrobnější rozbor škálování ($O(n^2)$) oproxi exponenciálním metodám (vše, co je polynomiální, je dobré)
- [x] zkontrolovat m_s^2 v AF
- [ ] zkusit TOTAL_SZ na najití PS fáze (asi někde mimo práci) - spustit na KFES na CPU
    - nezapomenout změnit sampler na exchange
    - už to běží na KFESu
    - vypadá to na **náznak PS fáze**
        - běží to ještě kolem podezřelého bodu -> zpřesnění



## zavrhnuto alias opt-out
- pustit 64 na KFESu s HODN2 samples, abchoym viděli, jak se zmenšuje errorbar
- **udělat errorbar zprůměrováním posledních 50 hodnot a zjistit, jestli to dává smysl**


## další poznámky
- [ ] for the sake of clarity we have not included the errorbars
- počkat, až doběhne KFES, a pak vybrat bod blízko transice a...
    - spustit jedno MC aby vyhodnotilo parametry uspořádání s **větším počtem samplů**
- jsem moc defenzivní
- [x] why not exact? vždy je třeba vybrat samply
    - dopsat podrobněji, že implementace neumožňuje použít všechny samply
- [ ] gap souvisí jenom s metodou exaktního samplingu
- unlike ... works, with accordance with ... works
- [x] chap4 - znovu definovat alpha (člověk to už zapoměl)
- 4.3.1
    - začít: Takto jsem nastavil parametry a takové to dává výsledky. There is also a possibility that more thorough parameters would give a better performance. Of course there is a possibility that a thorough scan of hyperparameters would lead to better results.
- **!!!** PS order parametr je asi definován stejně, jak to má sandvik (nepřenásoben 4)!!


## Agenda schůzky 21.4.
- pokud to skutečně najde PS fázi, tak bychom mohli použít transfer learning a dostat se do správného poměru J/J'
- co vše se hodí do outlooku?