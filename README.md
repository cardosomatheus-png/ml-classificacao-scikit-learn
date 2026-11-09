# Classificação supervisionada e combinação de classificadores com scikit-learn

Atividades da graduação em Ciência de Dados (UFMS) implementadas em Python. Os dois scripts seguem o mesmo fluxo: carregar a base, explorar os dados, treinar, avaliar e comparar modelos.

## Conteúdo

| Arquivo | O que faz |
| --- | --- |
| `knn_classificacao.py` | Classificação com k-NN: exploração dos dados, padronização, escolha do k por validação cruzada e avaliação com acurácia, matriz de confusão, precisão, revocação e F1-score. |
| `combinacao_classificadores.py` | Compara k-NN, árvore de decisão e naïve Bayes com Bagging, Random Forest, AdaBoost, Voting (hard e soft) e Stacking, usando a mesma validação cruzada repetida para todos. |

## Como executar

```bash
pip install -r requirements.txt

python knn_classificacao.py
python combinacao_classificadores.py
```

Por padrão os scripts usam a base Iris do scikit-learn. Para usar outra base em CSV:

```bash
python knn_classificacao.py --dados base.csv --alvo classe
python combinacao_classificadores.py --dados base.csv --alvo classe --sep ";"
```

Se `--alvo` não for informado, a última coluna do arquivo é usada como classe.

## Resultados com a base Iris

- **k-NN:** o melhor k na validação cruzada foi 3, com acurácia de 0,911 no conjunto de teste (70/30).
- **Combinação de classificadores:** na validação cruzada, o melhor classificador individual (k-NN, 0,962) empatou com o melhor combinado (Random Forest, 0,960). Em uma base pequena e de classes bem separadas, combinar modelos traz pouco ganho.

Os scripts também salvam gráficos (`k_vs_acuracia.png`, `matriz_confusao.png` e `comparacao_classificadores.png`).

## Tecnologias

Python, pandas, scikit-learn e matplotlib.

## Autor

Matheus Cardoso da Silva — [LinkedIn](https://www.linkedin.com/in/matheus-cardoso-da-silva-a21797162/)
