import argparse #pokretanje komandne linije i hvatanje redaka iz iste
import os #putanje i rad sa direktorijima
import joblib #spreamnje i ucitavanje modela scikit learn
import numpy as np #rad sa podacima
import pandas as pd #rad sa tablicama i csv

from sklearn.model_selection import train_test_split, GridSearchCV #train i test grupe, trazi najbolje kombinacije parametara
from sklearn.pipeline import Pipeline #omogucuje da spojis korake u jedan lanac 
from sklearn.metrics import classification_report, confusion_matrix #matrica zabuna (koliko je fulo ili pogodio), precision recall po klasama

from sklearn.naive_bayes import MultinomialNB #NaiveBayes za tekst radi s frekvencijama
from sklearn.linear_model import LogisticRegression #logisticka regresija (klasifikator)
from sklearn.svm import LinearSVC #linerani SVM
from sklearn.calibration import CalibratedClassifierCV # moze davati vjerojatnosti
from pomocne_klase_modela import KombiniraniVektorizator #uvozimo klasu Kom... iz pomocne klase
 

def ucitaj_processed(putanja: str) -> pd.DataFrame: #funkcija prima putanju do csv i vraca dataframe
    df = pd.read_csv(putanja) #ucitava csv tablicu
    df = df[["text", "label"]].dropna() #izbaci redove gdje je nest prazno
    df["text"] = df["text"].astype(str) #forsira da sve u stupcu text bude u formatu string
    df["label"] = df["label"].astype(str) #isto za label
    df = df.drop_duplicates(subset=["text", "label"]) #brise duplikate
    return df #vraca dataframe


def napravi_model(naziv: str): #radi sklearn model
    naziv = naziv.lower() #naziv vraca u malim slovima

    if naziv == "nb": #ako je naziv nb koristi se ta metoda
        return MultinomialNB()

    if naziv == "logreg": #isto
        return LogisticRegression(max_iter=3000, class_weight="balanced")

    if naziv == "svm": #isto
        baza = LinearSVC(max_iter=5000)
        return CalibratedClassifierCV(baza, method="sigmoid", cv=3)

    raise ValueError("Model mora biti: nb, logreg ili svm") #vraca error ako naziv modela nije ispravan


def main(): #main funkcija
    parser = argparse.ArgumentParser() #parser koji cita argumente iz komandne linije
    parser.add_argument("--data", required=True, help="data/processed/processed.csv") #dodaje opcije koje mozemo koristi, poput podataka, navesti csv
    parser.add_argument("--model", default="svm", choices=["nb", "logreg", "svm"]) #metode
    parser.add_argument("--out", default="models/phishing_model.joblib") #putanja gdje ce se spremit istrenirani pipeline
    parser.add_argument("--report_dir", default="reports") #folder za izvjestaj
    parser.add_argument("--tuning", action="store_true", help="Pokreni GridSearchCV (sporije, ali bolje)") #flag, ako dodas je true, ak ne ostane false
    args = parser.parse_args() #cita argumente iz terminala

    os.makedirs(os.path.dirname(args.out), exist_ok=True) #kreira folder ako ne postoji
    os.makedirs(args.report_dir, exist_ok=True) #isto napravi za folder report

    df = ucitaj_processed(args.data) #ucita csv
    X = df["text"].values #ulazi/tekstovi
    y = df["label"].values #labele

    X_train, X_test, y_train, y_test = train_test_split( #80% trening i 20%test, uvijek ista podjela
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vekt = KombiniraniVektorizator() #stvori objekt koji ce tekst pretvariti u feature vektore
    clf = napravi_model(args.model) #napravi model NB ili logreg ili SVM

    pipeline = Pipeline([ #pipeline je lanac koraka
        ("feat", vekt), #feat dobije tekst i napravi feature matrice
        ("clf", clf) #onda clf dobije taj feature i uci klasificirati
    ])

    if args.tuning: #ako je user dao tuning, ulazi ovdje
        param_grid = { #definira koje parametre zelis isprobat
            "feat__min_df": [1, 2, 3], #minimalno koliko puta se token mora pojaviti da se uvrsti.
            "feat__max_df": [0.90, 0.95, 0.98], #izbacuje prečeste tokene (npr. “the”, “and”) ako su prečesti
            "feat__word_ngram": [(1, 1), (1, 2)],#word n-gram raspon (npr. (1,2) znači unigram + bigram)
            "feat__char_ngram": [(3, 5), (4, 6)],#char n-gram raspon (pomaže za URL-ove, čudne riječi, maskiranje)
        }

        if args.model == "nb": #za NB, alpha je smoothing operator, parametar od korakak clf?
            param_grid["clf__alpha"] = [0.1, 0.5, 1.0]
        elif args.model == "logreg": #za logreg, c je jacina regularizacije
            param_grid["clf__C"] = [0.5, 1.0, 2.0, 4.0]
        elif args.model == "svm": #za SVM je slucaj specifican, jer je clf zapravo CalibratedClassifierCSV
            param_grid["clf__estimator__C"] = [0.5, 1.0, 2.0, 4.0]

        gs = GridSearchCV(
            pipeline, #pipeline je objekt koji se tunira
            param_grid=param_grid, #govori koje kombinacije isprobati
            scoring="f1_macro", #optimizira makro F1
            cv=4, #4fold cross validation
            n_jobs=-1, #Koristi sve CPU jezgre
            verbose=2 #ispisuje detalje rada
        )
        gs.fit(X_train, y_train) #gridSearch trenira puno modela s razlicitim parametrima
        pipeline = gs.best_estimator_ #Uzme najbolju kombinaciju (najbolji pipeline) i postavi ga kao glavni.
        print("Najbolji parametri:", gs.best_params_) #spiše koji su parametri ispali najbolji.
    else:
        pipeline.fit(X_train, y_train) #Samo trenira pipeline s default postavkama.

    y_pred = pipeline.predict(X_test) #Model predvidi labelu za svaki tekst iz test skupa.

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y)) #labels=np.unique(y) daje listu klasa (npr. ["legit", "phishing"]) i osigurava redoslijed
    rep = classification_report(y_test, y_pred, digits=4) #Generira tekstualni report s precision/recall/f1 i support. 
    #digits=4 → 4 decimale.

    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", rep) #Ispis u konzolu

    with open(os.path.join(args.report_dir, "classification_report.txt"), "w", encoding="utf-8") as f: #os.path.join(...) složi putanju (sigurno za OS). Otvori datoteku u write modu.
        f.write(rep) # Zapiše report unutra.

    joblib.dump(pipeline, args.out) #Spremi cijeli pipeline (vektorizator + model) u .joblib datoteku.
    print(f"\nModel spremljen u: {args.out}") #
    print(f"Izvjestaj u: {args.report_dir}/classification_report.txt") #Potvrda gdje je spremljeno.


if __name__ == "__main__": #Ako se datoteka pokreće direktno → pokreni main().
    main() #

