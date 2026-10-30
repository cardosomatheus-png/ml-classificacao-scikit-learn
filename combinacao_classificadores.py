"""
Atividade: combinação de classificadores (scikit-learn)

Compara três classificadores individuais (k-NN, árvore de decisão e naïve Bayes)
com técnicas de combinação:
  - Bagging (árvores de decisão)
  - Random Forest
  - AdaBoost (boosting)
  - Voting hard e soft (k-NN + árvore + naïve Bayes)
  - Stacking (os três classificadores + regressão logística como meta-modelo)

Todos são avaliados com a mesma validação cruzada estratificada repetida, para a
comparação ser justa. Os melhores modelos também são avaliados num conjunto de
teste separado (holdout).

Uso:
    python combinacao_classificadores.py
    python combinacao_classificadores.py --dados base.csv --alvo classe
    python combinacao_classificadores.py --dados base.csv --ref 0.90

  --ref  acurácia reportada no livro (p. 211), em fração (0.90) ou percentual (90);
         o script diz se cada modelo ficou melhor ou pior que esse valor.
Se --alvo não for informado, a última coluna do arquivo é usada como classe.
Requisitos: pandas, scikit-learn e matplotlib (opcional).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEMENTE = 42


def carregar_dados(caminho, alvo, sep):
    if caminho is None:
        iris = load_iris(as_frame=True)
        df = iris.frame.drop(columns="target")
        df["classe"] = iris.target_names[iris.target]
        return df, "classe"
    df = pd.read_csv(caminho, sep=sep)
    return df, (alvo or df.columns[-1])


def preparar(df, alvo):
    df = df.dropna()
    return pd.get_dummies(df.drop(columns=alvo)), df[alvo]


def montar_modelos():
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    arvore = DecisionTreeClassifier(random_state=SEMENTE)
    nb = GaussianNB()
    base = [("knn", knn), ("arvore", arvore), ("nb", nb)]

    individuais = {
        "k-NN (k=5)": knn,
        "Árvore de decisão": arvore,
        "Naïve Bayes": nb,
    }
    combinados = {
        "Bagging (árvores)": BaggingClassifier(
            DecisionTreeClassifier(), n_estimators=100, random_state=SEMENTE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=SEMENTE),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=SEMENTE),
        "Voting (hard)": VotingClassifier(base, voting="hard"),
        "Voting (soft)": VotingClassifier(base, voting="soft"),
        "Stacking": StackingClassifier(
            base, final_estimator=LogisticRegression(max_iter=1000), cv=5),
    }
    return individuais, combinados


def normalizar_ref(valor):
    if valor is None:
        return None
    return valor / 100 if valor > 1 else valor


def comparar_com_ref(nome, media, ref):
    if ref is None:
        return ""
    dif = media - ref
    if abs(dif) < 0.005:
        return f"  ≈ igual à referência ({ref:.3f})"
    return f"  {'MELHOR' if dif > 0 else 'PIOR'} que a referência ({ref:.3f}) em {abs(dif):.3f}"


def salvar_grafico(resultados):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib não instalado: gráfico não foi gerado)")
        return
    nomes = list(resultados.index)
    medias = resultados["media"].values
    cores = ["tab:gray" if t == "individual" else "tab:blue" for t in resultados["tipo"]]
    plt.figure(figsize=(8, 5))
    plt.barh(nomes, medias, xerr=resultados["desvio"].values, color=cores, capsize=3)
    plt.gca().invert_yaxis()
    plt.xlim(max(0, medias.min() - 0.1), 1.0)
    plt.xlabel("Acurácia média (validação cruzada)")
    plt.title("Classificadores individuais (cinza) x combinados (azul)")
    plt.tight_layout()
    plt.savefig("comparacao_classificadores.png", dpi=150)
    plt.close()
    print("Gráfico salvo: comparacao_classificadores.png")


def main():
    ap = argparse.ArgumentParser(description="Combinação de classificadores")
    ap.add_argument("--dados", help="caminho ou URL de um CSV (padrão: Iris)")
    ap.add_argument("--alvo", help="coluna de classe (padrão: última coluna)")
    ap.add_argument("--sep", default=",", help="separador do CSV (padrão: ',')")
    ap.add_argument("--ref", type=float,
                    help="acurácia reportada no livro (p. 211) para comparação")
    args = ap.parse_args()
    ref = normalizar_ref(args.ref)

    df, alvo = carregar_dados(args.dados, args.alvo, args.sep)
    X, y = preparar(df, alvo)
    print(f"Base: {len(X)} instâncias, {X.shape[1]} atributos, "
          f"{y.nunique()} classes ({alvo})\n{y.value_counts().to_string()}\n")

    individuais, combinados = montar_modelos()
    todos = {**{n: (m, "individual") for n, m in individuais.items()},
             **{n: (m, "combinado") for n, m in combinados.items()}}

    # 1) Comparação por validação cruzada repetida (5 folds x 3 repetições)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEMENTE)
    print("=" * 66)
    print("1) VALIDAÇÃO CRUZADA (5 folds x 3 repetições) - acurácia")
    print("=" * 66)
    linhas = {}
    for nome, (modelo, tipo) in todos.items():
        notas = cross_val_score(modelo, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        linhas[nome] = {"tipo": tipo, "media": notas.mean(), "desvio": notas.std()}
        print(f"{nome:<20} {notas.mean():.3f} (± {notas.std():.3f})"
              f"{comparar_com_ref(nome, notas.mean(), ref)}")
    res = pd.DataFrame(linhas).T
    res[["media", "desvio"]] = res[["media", "desvio"]].astype(float)

    melhor_ind = res[res.tipo == "individual"]["media"].idxmax()
    melhor_comb = res[res.tipo == "combinado"]["media"].idxmax()
    dif = res.loc[melhor_comb, "media"] - res.loc[melhor_ind, "media"]
    print(f"\nMelhor individual: {melhor_ind} ({res.loc[melhor_ind, 'media']:.3f})")
    print(f"Melhor combinado:  {melhor_comb} ({res.loc[melhor_comb, 'media']:.3f})")
    print("A combinação "
          + ("superou" if dif > 0.005 else "empatou com" if dif > -0.005 else "ficou abaixo d")
          + f"{'o' if dif < -0.005 else ''} melhor classificador individual "
          f"(diferença de {dif:+.3f}).\n")

    # 2) Holdout: treino 70% / teste 30%
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEMENTE)
    print("=" * 66)
    print("2) CONJUNTO DE TESTE (holdout 70/30) - acurácia")
    print("=" * 66)
    for nome, (modelo, _) in todos.items():
        modelo.fit(X_tr, y_tr)
        print(f"{nome:<20} {accuracy_score(y_te, modelo.predict(X_te)):.3f}")

    print(f"\nRelatório detalhado - {melhor_comb}:")
    print(classification_report(y_te, todos[melhor_comb][0].predict(X_te), digits=3))

    salvar_grafico(res)


if __name__ == "__main__":
    main()
