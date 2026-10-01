# Phishing Message Detection (Email / SMS) with Classical ML

A text classifier that labels an email or SMS as **PHISHING** or **LEGIT**, plus a small desktop app for checking a message in real time.

> Course project: **Introduction to Artificial Intelligence** (Uvod u umjetnu inteligenciju), FOI, University of Zagreb, January 2026  
> Authors: **Lovro Alfirević & Josip Ančić** · Mentor: Assoc. Prof. Dijana Oreški, PhD

## How it works

| Step | File | What it does |
|---|---|---|
| 1. Data prep | `data_prep.py` | merges several message datasets, normalises text, maps labels to `phishing` / `legit`, removes empty and duplicate rows |
| 2. Features | `pomocne_klase_modela.py` | word n-gram TF-IDF, char n-gram TF-IDF and hand-crafted features (length, number of URLs, digits, `!`, `?`, uppercase) |
| 3. Training | `train.py` | 80/20 split, trains **Multinomial Naive Bayes**, **Logistic Regression** or **SVM** (calibrated for probabilities), uses `GridSearchCV`, saves the report and the `.joblib` pipeline |
| 4. App | `gui_tkinter.py` | paste a message, click *Provjeri*, and get a red PHISHING or green LEGIT result with class probabilities |

## Tech stack

Python · scikit-learn · pandas · TF-IDF · joblib · Tkinter · LaTeX (documentation)

## Run

```bash
pip install scikit-learn pandas joblib
python data_prep.py
python train.py          # choose model: nb / logreg / svm
python gui_tkinter.py
```

> The source code is being uploaded separately. The full documentation (in Croatian) is in [`docs/`](docs/).

## Takeaways

Classical models are fast, interpretable and accurate enough to act as a lightweight first line of defence. TF-IDF misses deeper semantic context, so a hybrid approach with language models would be the next step for more sophisticated social-engineering attacks.

---

## Hrvatski

# Detekcija phishing poruka (e-mail/SMS) klasičnim strojnim učenjem

Klasifikator koji poruku označava kao **PHISHING** ili **LEGIT**, uz desktop aplikaciju za provjeru poruke u stvarnom vremenu.

> Projekt iz kolegija **Uvod u umjetnu inteligenciju**, FOI, siječanj 2026.  
> Autori: **Lovro Alfirević i Josip Ančić** · Mentorica: izv. prof. dr. sc. Dijana Oreški

### Postupak

1. `data_prep.py`: spajanje izvora, čišćenje teksta, labele phishing/legit, uklanjanje duplikata
2. `pomocne_klase_modela.py`: TF-IDF nad riječima i znakovima te dodatne značajke (duljina, URL-ovi, znamenke, uskličnici, velika slova)
3. `train.py`: podjela 80/20, Naive Bayes / logistička regresija / SVM uz GridSearchCV, spremanje modela (.joblib)
4. `gui_tkinter.py`: aplikacija u kojoj se zalijepi poruka i dobije rezultat s vjerojatnostima

Kod se dodaje naknadno, a cjelovita dokumentacija nalazi se u mapi `docs/`.
