# Titanic — Previsão de Sobrevivência com Random Forest

Análise exploratória e modelo de classificação para prever a sobrevivência de passageiros do Titanic com base em características como gênero, classe social, idade e tarifa paga, utilizando Random Forest com Pipeline e OneHotEncoder.

---

## Objetivo

Identificar quais fatores mais influenciaram a sobrevivência dos passageiros do Titanic e construir um modelo capaz de prever se um passageiro sobreviveu ou não com base em suas características.

---

## Estrutura do Projeto

```
 projeto
 ┣ Titanic.ipynb              # Notebook principal
 ┣ Titanic-Dataset.csv        # Dataset
 ┗ README.md
```

---

## Dataset

- **Fonte:** [Kaggle — Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- **891 registros originais** | **12 colunas** | **876 registros após limpeza**
- **Variável alvo:** `Survived` 0 = Não Sobreviveu | 1 = Sobreviveu
- **Distribuição:** 535 não sobreviventes (61.1%) | 341 sobreviventes (38.9%)

### Colunas originais

| Coluna | Descrição |
|--------|-----------|
| `PassengerId` | Identificador único do passageiro — removida |
| `Survived` | **Variável alvo** — 0 = não sobreviveu | 1 = sobreviveu |
| `Pclass` | Classe do bilhete — 1ª, 2ª ou 3ª classe |
| `Name` | Nome do passageiro — removida |
| `Sex` | Gênero do passageiro |
| `Age` | Idade do passageiro |
| `SibSp` | Número de irmãos/cônjuges a bordo — removida |
| `Parch` | Número de pais/filhos a bordo |
| `Ticket` | Número do bilhete — removida |
| `Fare` | Tarifa paga pela passagem (£) |
| `Cabin` | Número da cabine — removida (77% nulos) |
| `Embarked` | Porto de embarque: S=Southampton, C=Cherbourg, Q=Queenstown |

---

## Limpeza dos Dados

| Problema | Solução |
|----------|---------|
| `Cabin` — 687 nulos (77%) | Coluna removida |
| `PassengerId`, `Ticket`, `Name` | Removidas — sem valor preditivo |
| `SibSp` | Removida |
| `Age` — 177 nulos | Preenchidos com a mediana (28 anos) e arredondados para inteiros |
| `Embarked` — 2 nulos | Preenchidos com a moda (Southampton) |
| 15 passageiros com `Fare = 0` | Removidos — registros inconsistentes |
| Idades decimais (ex: 0.83, 34.5) | Arredondadas para inteiros — bebês registrados em fração de ano |

---

## Análise Exploratória (EDA)

### Principais insights

**Gênero foi o maior determinante de sobrevivência**
Mulheres tiveram taxa de sobrevivência muito superior aos homens reflexo direto da política "mulheres e crianças primeiro" adotada durante a evacuação. A diferença é expressiva e visível nos gráficos.

**Classe social influenciou diretamente as chances de sobreviver**
Passageiros da 1ª classe tiveram acesso mais fácil aos botes salva-vidas e ficavam nos conveses superiores do navio. A 3ª classe, alojada nas partes mais baixas, teve a menor taxa de sobrevivência.

**Distribuição de idades parecida entre sobreviventes e não sobreviventes**
Não há uma faixa etária claramente mais favorecida embora crianças pequenas tenham tido prioridade na evacuação, o efeito não é suficientemente forte para distorcer a distribuição geral.

**Porto de embarque e tarifa estão correlacionados com a classe**
Passageiros embarcados em Cherbourg (C) pagaram tarifas mais altas, concentrando passageiros de 1ª classe. Southampton (S) embarcou a maior parte dos passageiros, incluindo a maioria da 3ª classe.

---

## Modelo — Random Forest

### Features utilizadas
`Pclass`, `Sex`, `Age`, `Parch`, `Fare`, `Embarked`

### Por que OneHotEncoder?
As colunas `Sex` e `Embarked` são categóricas sem ordem natural. O **OneHotEncoder** cria colunas binárias para cada categoria, evitando que o modelo interprete uma hierarquia falsa entre os valores.

### Pipeline
```python
model = Pipeline(steps=[
    ('preprocessor', ColumnTransformer([
        ('num', SimpleImputer(strategy='mean'), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])
```

### Divisão dos dados
- **80% treino** — 700 amostras
- **20% teste** — 176 amostras

---

## Resultados

| Métrica | Não Sobreviveu (0) | Sobreviveu (1) |
|---------|-------------------|----------------|
| Precision | **0.82** | 0.61 |
| Recall | 0.74 | **0.71** |
| F1-score | 0.78 | 0.66 |
| **Acurácia geral** | | **73%** |

### Interpretação

O modelo atingiu **73% de acurácia** resultado razoável para uma Random Forest. É mais preciso ao classificar quem **não sobreviveu** (Precision 0.82) do que quem sobreviveu (0.61), o que é esperado dado o leve desbalanceamento do dataset.

O **Recall de 0.71 para sobreviventes** indica que o modelo identificou corretamente 70% dos que realmente sobreviveram os 30% restantes foram classificados incorretamente como não sobreviventes.


<img width="497" height="385" alt="image" src="https://github.com/user-attachments/assets/6ac12034-b3ee-4f56-9dfa-20b8ee3fad9b" />

- Como próximos passos, seria interessante realizar a otimização de hiperparâmetros (como n_estimators e max_depth), testar algoritmos de Gradient Boosting (como XGBoost ou LightGBM) ou reavaliar o impacto de manter a coluna SibSp ou criar novas features para melhorar o desempenho na classe dos sobreviventes.
---

## Tecnologias Utilizadas

- Python 3
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Como Executar

1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
```

2. Instale as dependências
```bash
pip install pandas matplotlib seaborn scikit-learn
```

3. Abra o notebook
```bash
jupyter notebook Titanic.ipynb
```

> Certifique-se de que o arquivo `Titanic-Dataset.csv` está na mesma pasta do notebook antes de rodar.
