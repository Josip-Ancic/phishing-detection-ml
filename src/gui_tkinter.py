from pomocne_klase_modela import KombiniraniVektorizator, DodatneZnacajke
import tkinter as tk
from tkinter import messagebox, scrolledtext
import os
import joblib


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "phishing_model.joblib")


class PhishingApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Detekcija phishing poruka (UUI projekt)")
        self.geometry("800x520")

        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Greška",
                f"Model nije pronađen: {MODEL_PATH}\n\nPrvo pokreni train.py da se model spremi."
            )
            self.destroy()
            return
        

        try:
            self.model = joblib.load(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Greška", f"Ne mogu učitati model.\n\nDetalji:\n{e}")
            self.destroy()
            return

        self._napravi_ui()

    def _napravi_ui(self):
        naslov = tk.Label(self, text="Detekcija phishing poruka (email/SMS) - klasični ML", font=("Arial", 14, "bold"))
        naslov.pack(pady=10)

        opis = tk.Label(
            self,
            text="Unesi tekst poruke i klikni 'Provjeri'.",
            font=("Arial", 11)
        )
        opis.pack(pady=5)

        okvir = tk.Frame(self)
        okvir.pack(fill="both", expand=True, padx=12, pady=10)

        self.ulaz = scrolledtext.ScrolledText(okvir, wrap=tk.WORD, font=("Arial", 11))
        self.ulaz.pack(fill="both", expand=True)

        gumbi = tk.Frame(self)
        gumbi.pack(pady=10)

        btn_provjeri = tk.Button(gumbi, text="Provjeri", width=15, command=self.provjeri)
        btn_provjeri.grid(row=0, column=0, padx=8)

        btn_ocisti = tk.Button(gumbi, text="Očisti", width=15, command=self.ocisti)
        btn_ocisti.grid(row=0, column=1, padx=8)

        self.rezultat = tk.Label(self, text="Rezultat: -", font=("Arial", 12, "bold"))
        self.rezultat.pack(pady=8)

        self.detalji = tk.Label(self, text="", font=("Arial", 11))
        self.detalji.pack(pady=2)

    def ocisti(self):
        self.ulaz.delete("1.0", tk.END)
        self.rezultat.config(text="Rezultat: -")
        self.detalji.config(text="")

    def provjeri(self):
        tekst = self.ulaz.get("1.0", tk.END).strip()
        if not tekst:
            messagebox.showwarning("Upozorenje", "Unesi tekst poruke.")
            return

        try:
            pred = self.model.predict([tekst])[0]
        except Exception as e:
            messagebox.showerror("Greška", f"Ne mogu napraviti predikciju.\n\nDetalji:\n{e}")
            return

        if pred == "phishing":
            self.rezultat.config(text="Rezultat: PHISHING", fg="red")
        else:
            self.rezultat.config(text="Rezultat: LEGIT", fg="green")

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba([tekst])[0]
            klase = list(self.model.classes_)
            score = {k: float(p) for k, p in zip(klase, proba)}

            p_phish = score.get("phishing", 0.0)
            p_legit = score.get("legit", 0.0)

            self.detalji.config(text=f"Vjerojatnost phishing: {p_phish:.4f} | legit: {p_legit:.4f}")
        else:
            self.detalji.config(text="(Model ne vraća vjerojatnosti.)")


if __name__ == "__main__":
    app = PhishingApp()
    app.mainloop()
