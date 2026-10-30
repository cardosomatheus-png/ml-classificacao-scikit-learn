"""
Atividade: classificação supervisionada com k-NN (scikit-learn)

Etapas:
  1. Carregar a base de dados (CSV local, URL ou, por padrão, a base Iris).
  2. Explorar os dados (tamanho, tipos, valores ausentes, distribuição das classes).
  3. Separar treino e teste (holdout estratificado).
  4. Escolher o melhor k por validação cruzada, usando apenas o treino.
  5. Avaliar no teste: acurácia, matriz de confusão, precisão, revocação e F1-score.
  6. Classificar um novo exemplo.

Uso:
    python knn_classificacao.py
    python knn_classificacao.py --dados base.csv --alvo classe
    python knn_classificacao.py --dados https://exemplo.com/base.csv --sep ";" --alvo classe

Se --alvo não for informado, a última coluna do arquivo é usada como classe.
Requisitos: pandas, scikit-learn e matplotlib (opcional, para os gráficos).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEMENTE = 42


def carregar_dados(caminho, alvo, sep):
    """Lê o CSV (arquivo ou URL). Sem caminho, usa a base Iris do scikit-learn."""
    if caminho is None:
        iris = load_iris(as_frame=True)
        df = iris.frame.drop(columns="target")
        df["classe"] = iris.target_names[iris.target]
        return df, "classe"
    df = pd.read_csv(caminho, sep=sep)
    return df, (alvo or df.columns[-1])


def explorar(df, alvo):
    print("=" * 60)
    print("1) EXPLORAÇÃO DOS DADOS")
    print("=" * 60)
    print(f"Instâncias: {len(df)} | Atributos descritivos: {df.shape[1] - 1}")
    print(f"Atributo de classe: '{alvo}'\n")
    print(df.head(), "\n")
    print("Valores ausentes por coluna:")
    print(df.isna().sum().to_string(), "\n")
    print("Distribuição das classes:")
    print(df[alvo].value_counts().to_string(), "\n")


def preparar(df, alvo):
    """Remove linhas incompletas e converte atributos categóricos em numéricos."""
    df = df.dropna()
    y = df[alvo]
    X = pd.get_dummies(df.drop(columns=alvo))  # texto -> colunas 0/1
    return X, y


def escolher_k(X_treino, y_treino, ks):
    """Avalia cada k por validação cruzada estratificada (somente no treino)."""
    folds = min(10, int(y_treino.value_counts().min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEMENTE)
    medias, desvios = [], []
    for k in ks:
        modelo = Pipeline([("padroniza", StandardScaler()),
                           ("knn", KNeighborsClassifier(n_neighbors=k))])
        notas = cross_val_score(modelo, X_treino, y_treino, cv=cv,
                                scoring="accuracy")
        medias.append(notas.mean())
        desvios.append(notas.std())
    return np.array(medias), np.array(desvios), folds


def salvar_graficos(ks, medias, desvios, melhor_k, cm, classes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError:
        print("(matplotlib não instalado: gráficos não foram gerados)")
        return
    plt.figure(figsize=(7, 4))
    plt.errorbar(ks, medias, yerr=desvios, marker="o", capsize=3)
    plt.axvline(melhor_k, color="tab:red", linestyle="--", label=f"melhor k = {melhor_k}")
    plt.xlabel("k (número de vizinhos)")
    plt.ylabel("Acurácia média na validação cruzada")
    plt.title("Escolha do k")
    plt.legend()
    plt.tight_layout()
    plt.savefig("k_vs_acuracia.png", dpi=150)
    plt.close()

    ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues")
    plt.title("Matriz de confusão (teste)")
    plt.tight_layout()
    plt.savefig("matriz_confusao.png", dpi=150)
    plt.close()
    print("Gráficos salvos: k_vs_acuracia.png e matriz_confusao.png")


def main():
    ap = argparse.ArgumentParser(description="Classificação com k-NN")
    ap.add_argument("--dados", help="caminho ou URL de um CSV (padrão: Iris)")
    ap.add_argument("--alvo", help="nome da coluna de classe (padrão: última coluna)")
    ap.add_argument("--sep", default=",", help="separador do CSV (padrão: ',')")
    args = ap.parse_args()

    df, alvo = carregar_dados(args.dados, args.alvo, args.sep)
    explorar(df, alvo)
    X, y = preparar(df, alvo)

    # 3) Holdout estratificado: 70% treino, 30% teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEMENTE)
    print(f"Treino: {len(X_treino)} instâncias | Teste: {len(X_teste)} instâncias\n")

    # 4) Escolha do k
    print("=" * 60)
    print("2) ESCOLHA DO k (validação cruzada no treino)")
    print("=" * 60)
    ks = list(range(1, 22, 2))  # valores ímpares evitam empates
    medias, desvios, folds = escolher_k(X_treino, y_treino, ks)
    for k, m, d in zip(ks, medias, desvios):
        print(f"k = {k:2d} | acurácia = {m:.3f} (± {d:.3f})")
    melhor_k = ks[int(np.argmax(medias))]
    print(f"\nValidação cruzada com {folds} folds. Melhor k = {melhor_k}\n")

    # 5) Modelo final e avaliação no teste
    modelo = Pipeline([("padroniza", StandardScaler()),
                       ("knn", KNeighborsClassifier(n_neighbors=melhor_k))])
    modelo.fit(X_treino, y_treino)
    previsto = modelo.predict(X_teste)

    print("=" * 60)
    print("3) AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("=" * 60)
    print(f"Acurácia: {accuracy_score(y_teste, previsto):.3f}\n")
    classes = modelo.classes_
    cm = confusion_matrix(y_teste, previsto, labels=classes)
    print("Matriz de confusão (linhas = classe real, colunas = classe prevista):")
    print(pd.DataFrame(cm, index=classes, columns=classes), "\n")
    print("Precisão, revocação e F1-score por classe:")
    print(classification_report(y_teste, previsto, digits=3))

    salvar_graficos(ks, medias, desvios, melhor_k, cm, classes)

    # 6) Classificar um novo exemplo (a primeira linha do teste, como demonstração)
    exemplo = X_teste.iloc[[0]]
    print("\nNovo exemplo:")
    print(exemplo.to_string(index=False))
    print(f"Classe prevista: {modelo.predict(exemplo)[0]} | classe real: {y_teste.iloc[0]}")


if __name__ == "__main__":
    main()
