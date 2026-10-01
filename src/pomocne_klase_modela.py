import re
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class DodatneZnacajke(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._url_re = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        url_broj = []
        duljina = []
        broj_brojeva = []
        broj_uzvicnika = []
        broj_upitnika = []
        broj_velikih = []

        for t in X:
            t = "" if t is None else str(t)
            duljina.append(len(t))
            url_broj.append(len(self._url_re.findall(t)))
            broj_brojeva.append(sum(ch.isdigit() for ch in t))
            broj_uzvicnika.append(t.count("!"))
            broj_upitnika.append(t.count("?"))
            broj_velikih.append(sum(ch.isupper() for ch in t))

        mat = np.array(
            [duljina, url_broj, broj_brojeva, broj_uzvicnika, broj_upitnika, broj_velikih],
            dtype=np.float32
        ).T

        return csr_matrix(mat)


class KombiniraniVektorizator(BaseEstimator, TransformerMixin):
    def __init__(self, word_ngram=(1, 2), char_ngram=(3, 5), min_df=2, max_df=0.95):
        self.word_ngram = word_ngram
        self.char_ngram = char_ngram
        self.min_df = min_df
        self.max_df = max_df

        self.word = TfidfVectorizer(
            lowercase=True,
            ngram_range=self.word_ngram,
            min_df=self.min_df,
            max_df=self.max_df
        )
        self.char = TfidfVectorizer(
            lowercase=True,
            analyzer="char",
            ngram_range=self.char_ngram,
            min_df=self.min_df,
            max_df=self.max_df
        )
        self.extra = DodatneZnacajke()

    def fit(self, X, y=None):
        self.word.fit(X, y)
        self.char.fit(X, y)
        self.extra.fit(X, y)
        return self

    def transform(self, X):
        Xw = self.word.transform(X)
        Xc = self.char.transform(X)
        Xe = self.extra.transform(X)
        return hstack([Xw, Xc, Xe]).tocsr()
