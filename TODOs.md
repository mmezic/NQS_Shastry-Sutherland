# My TODO list

## moje TODO
- [ ] udělat obrázek popisující průměrování energií z posledních 100 iterací a záporné hodnoty zpsůobené asi numerickou přesnotsí


TODO:
- [ ] dogenerovat tabulku a vybrat model, který použiu na $H_Z$
- [ ] zkusit dva modely z tabulky pustit s SAMPLES=5000 (do appendixu) a podívat se, jak hodně to zlepší accuracy
- [ ] do tabulky už nepřidávat zmenšování learning ratu
- [ ] SAMPLES mění finální rozptyl nad exaktní energií

TODO:
- [ ] dokončit tabulku
- [ ] vybrat si jeden nejlepší model, který potom pustím na případu s mag polem

## Thesis text
- [x] Dají se lépe formátovat citace? Jako [69] -> Ano
- [ ] Zmínit článek `Neural tensor contractions and the expressive power of deep neural quantum states` v úvodu
- [ ] dopsat do práce, že jsme narazili na problém s konvergencí bez viditelných biasů, ale bylo to způsobeno pouze moc malou inicializací - z toho můžeme usuzovat, že visible biasy mohou pomáhat při konvergenci v těžkých chvílích
### anstract
- [ ] ne až tak technický
### úvod
- [ ] je potřeba silnější motivaci
    - [ ] proč ML? ... podle článku od gcarlea popisujícího generalitu ML
    - [ ] proč SSM? ... má 2 úplně rozdílné fáze -> dobrý testbed pro generalizační schopnosti ML
    - [ ] nikdo nezkoušel ML na SSM
    - [x] všichni používají $H_z = 0$ !!
### 1. kapitola
- [ ] přidat popis Jastrowa a popsat na něm VMC (přesnost závisí na kvalitě trial funkce)
### 2. kapitola
- [ ] obrázky z arxivu můžu asi volně zkopírovat fo kap 2.2 `getry-data` program
    - [ ] buď naskenovat obrázek z nějakého článku nebo nakreslit vlastní sketch
        - **¿¿** experimentální hosnoty nebo ze simulace **??**
### 3. kapitola
- [ ] připsat popis fyzikální interpretace hidden vrstvy
- [ ] zdůraznit v textu věci, které jsem udělal sám
### 4. kapitola
- [ ] pod fig 4.1 (¿je to v MSR?) přidat další graf, kde bude 10 000 samplů
    + finite size scaling ... jaké používat errorbary? ... jejich velikost totiž nesouvisí s η, ale s #samples
### Appendixové

## Další věci
- pustit 64 na KFESu s HODN2 samples, abchoym viděli, jak se zmenšuje errorbar
- **udělat errorbar zprůměrováním posledních 50 hodnot a zjistit, jestli to dává smysl**
