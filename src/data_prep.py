#Uvoz potrebnih biblioteka
import argparse #citanje argumenata iz komandne linije
import re #trazenje/ciscenje teksta
import pandas as pd #rad sa tablicama i CSV-ovima

def procisti_tekst(t: str) -> str: #funkcija koja prima tekst i vraca string
    if pd.isna(t): #provjerava ako je prazna vrijednost, vraca prazan string
        return ""
    t = str(t) #pretvara t u string
    t = t.replace("\r", " ").replace("\n", " ").strip() #pretvara vise redni email u tekst u jednom 
    #redu bez razmaka na pocetku i kraju
    t = re.sub(r"\s+", " ", t) #ako ima 10 razmaka skrati ih u 1
    return t

def label_u_klasu(lbl) -> str: #prima oznaku klase i vraca standaridiziranu klasu
    if pd.isna(lbl): #ako label nedostaje, vrati prazno
        return ""
    s = str(lbl).strip().lower() #pretvori u stirng, bez razmaka na krajevima i u mala slova
    if s in {"1", "phishing", "phish", "spam"}: #ako je label nesto od ovoga
        return "phishing" #mapira u phishing
    if s in {"0", "legit", "legitimate", "ham", "non-phishing", "not phishing"}:
        return "legit" #ista stvar samo vraca u drugu mapu
    
    return s

def ucitaj_spoji(putanja: str, text_cols: list[str], label_col: str) -> pd.DataFrame: #ucitava csv, tekst prevara u listu stringova, i napravi label,
    # a na kraju vraca gotov dataframe
    df = pd.read_csv(putanja) #procitaj podatke iz csva

    
    for c in text_cols + [label_col]: #napravi se lista stupaca koji trebaju postojati
        if c not in df.columns: #i onda se u csvu traze ti stucpi
            raise ValueError(f"U datoteci {putanja} ne postoji stupac: {c}. Postoje: {list(df.columns)}")
           #ako ne postoje vraca se error i poruka
    tekst = [] #stvaranje tekst liste
    for _, row in df.iterrows(): #petlja koja prolazi kroz sve redove csva, _ indeks reda, red - series
        dijelovi = [] #prazna lista
        for c in text_cols: 
            dijelovi.append(procisti_tekst(row[c])) #poziva se funckija koja cisti red i onda tako za svaki redak se radi
            # te se popunjava dijelovi lista
        tekst.append(" ".join([x for x in dijelovi if x])) #na kraju se svi ti dijelovi spoje, tako da se redovi
        #odvoje razmakom

    out = pd.DataFrame({ #stvara se dataframe s 2 stupca, text i label
        "text": tekst,  #tekst
        "label": [label_u_klasu(x) for x in df[label_col]] #mapiranje, prolazi kroz sve labele te ih mapira u
        #phishing il legit
    })

    out = out[(out["text"].str.len() > 0) & (out["label"].str.len() > 0)] #ciscenje i filtriranje, zadrzi redove gdje
    # tekst i label nisu prazni
    out = out.drop_duplicates(subset=["text", "label"]).reset_index(drop=True) #makni duplikate

    out = out[out["label"].isin({"phishing", "legit"})].reset_index(drop=True) #zadrzi redove samo tamo gdje je label
    #legit ili phishing, ostalo izbaci
    return out #vraca ocisceni dataframe

def main(): #main funckija
    parser = argparse.ArgumentParser() #parser koji cita argumente iz komandne linije
    parser.add_argument("--out", required=True, help="Izlazni processed CSV") #zapocinje sa out
    args = parser.parse_args()#parsira dokumente i sprema ih u args
   #primjer python skripta.py --out data/processed.csv
   #args.out == "data/processed.csv"
    
    datasets = [
        ("data/raw/phishing_email.csv", ["text_combined"], "label"),
        ("data/raw/Enron.csv", ["subject", "body"], "label"),
        ("data/raw/Nazario.csv", ["subject", "body"], "label"),
    ] #3 dataseta, kao tuple, 1.dataset ima text combined, druga dva spajaju subject i body

    svi = [] #prazna lista 
    for put, tcols, lcol in datasets: #stvaras te varijable da poprime oblike onog iz dataseta
        df = ucitaj_spoji(put, tcols, lcol) #iz csva se onda uzimaju stupci fitrira se poruka, da bude jasna i kratka
        # te da samo ostanu stvari koje su pozitivne na phishing ili legit, ostani stupci text i label
        df["source"] = put.split("/")[-1] #dodaje novi stupac koji pokazuje putanju nekog teksta, tj izvor odkud je
        svi.append(df) #sve se sprema u listu svi, tj taj obradeni dio

    spojeno = pd.concat(svi, ignore_index=True) #spoji dataframeove jedan ispod drugog

    spojeno = spojeno.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    # makni duplikate
    print("Ukupno zapisa:", len(spojeno)) #ispisuje koliko redova ima ukupno
    print("Distribucija labela:") 
    print(spojeno["label"].value_counts()) #koliko puta se pojavljuje koja labela
    print("\nDistribucija po izvoru:") 
    print(spojeno.groupby(["source", "label"]).size()) #broj redoba po labeli i izvoru

    spojeno.to_csv(args.out, index=False) #sprema dataframe u csv na putanju koju smo dali preko --out
    print(f"\nSpremio: {args.out}") #ispis gdje je spremljen csv

if __name__ == "__main__":
    main() #provjera da smo stvarno uvezli tu datoteku