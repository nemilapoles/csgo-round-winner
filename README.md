# Contexto

O dataset contém o resultado de 122 mil momentos que aconteceram durante mais de 700 partidas de campeonatos de alto nível do jogo Counter-Strike: Global Offensive. 

[Database aqui!](https://docs.google.com/spreadsheets/d/1bM7H4ppjKmAv6W6vfrMpyOHiFgI6vYuj/edit?usp=drive_link&ouid=104240799900660421773&rtpof=true&sd=true)

# Objetivo

É um problema de classificação binária, pois o que queremos é prever qual das equipes do jogo vencerá uma rodada de CS:GO com base nas informações disponíveis. Como o *target* (resposta a ser alcançada) **possui duas opções de classe (Terroristas e Contra-Terroristas), temos um problema de aprendizado supervisionado, pois já conhecemos as respostas dos dados utilizados para treinar o algoritmo, permitindo que ele aprenda a identificar padrões e faça previsões com base neles. Assim, trata-se de uma classificação binária supervisionada. 

O objetivo desse projeto foi aplicar técnicas de preparação, seleção de atributos e treinamento com métodos de Machine Learning, usando dados como o arsenal de ambas as equipes (T  e CT), o dinheiro disponível para as duas equipes, o mapa, entre outros. A ideia foi prever o vencedor da rodada (*round_winner*).

---

### ❗Importante:

Mensagens concentradas na formatação de citação:

> Citação exemplo
> 

Serão os retornos do terminal.

---

Em um notebook Python, importei a base de dados para utilização:

# **Preparando os dados**

1. Importo as bibliotecas necessárias para todos os processos que farei:

```python
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
```

2. Peço para que o algoritmo leia o database:

```python
df = pd.read_excel('csgo_round_snapshots.xlsx')
```

3. Utilizo o atributo `shape` para saber quantas colunas (*features*) tem no database:

```python
df.shape
```

> (122410, 94)
> 

4. Utilizo o atributo `info` para saber o datatype de cada coluna e verificar se alguma delas pode criar ruído nas análises:

```python
df.info()
```

> <class 'pandas.core.frame.DataFrame'>
> RangeIndex: 122410 entries, 0 to 122409
> Data columns (total 94 columns):
> .
> .
> .
> 3   map                           122410 non-null  object
> .
> .
> .
> 93  round_winner                  122410 non-null  object
dtypes: float64(1), int64(91), object(2)
memory usage: 87.8+ MB
> 

Concluo que a coluna `‘map’`, em vermelho, pode me atrapalhar na seleção das melhores features para o treinamento do algoritmo.

Em verde, temos nossa variável alvo (target), a que vai servir de base para conferir o quão próxima a previsão vai estar da realidade.

5. Separo as features do target e excluo a coluna ‘map’:

```python
X = df.drop(columns = ['round_winner', 'map'])
y = df['round_winner']
```

<aside>

*X ser maiúsculo e y ser minúsculo em exemplos e projetos tem uma razão!
É uma convenção matemática/computacional, já que X é usado para representar **características** (matriz), e y o **alvo** (vetor).*

</aside>

6. O código a seguir nos mostra colunas em que os valores são constantes, ou seja, não são úteis para mostrar a diferença entre as equipes. As identificamos com o método `nunique()` do Pandas, e as excluímos:

```python
constant_cols = X.columns[X.nunique() <= 1]
X[constant_cols].head()
```

|  | **ct_weapon_bizon** | **ct_weapon_g3sg1** | **t_weapon_m249** | **ct_weapon_negev** | **ct_weapon_r8revolver** | **ct_weapon_sawedoff** |
| --- | --- | --- | --- | --- | --- | --- |
| **0** | 0 | 0 | 0 | 0 | 0 | 0 |
| **1** | 0 | 0 | 0 | 0 | 0 | 0 |
| **2** | 0 | 0 | 0 | 0 | 0 | 0 |
| **3** | 0 | 0 | 0 | 0 | 0 | 0 |
| **4** | 0 | 0 | 0 | 0 | 0 | 0 |

```python
X = X.drop(columns = constant_cols)
```

# **Divisão do dataset**

1. Utilizo a função `train_test_split( )` para dividir os dados em conjuntos de treinamento e teste. Destino 25% a teste e o restante para o treinamento. É importante para que eu consiga avaliar se o modelo funciona com dados que ele não usou no treinamento:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25, random_state = 2000)
```

2. Uso `head()` para rapidamente conferir se a separação está correta visualmente:

```python
X_train.head()
```

|  | **time_left** | **ct_score** | **t_score** | **ct_health** | **t_health** | **ct_money** | **t_money** | **ct_helmets** | **t_helmets** | **ct_defuse_kits** | **...** | **ct_grenade_flashbang** | **t_grenade_flashbang** | **ct_grenade_smokegrenade** | **t_grenade_smokegrenade** | **ct_grenade_incendiarygrenade** | **t_grenade_incendiarygrenade** | **ct_grenade_molotovgrenade** | **t_grenade_molotovgrenade** | **ct_grenade_decoygrenade** | **t_grenade_decoygrenade** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **66505** | 54.89 | 1 | 8 | 492 | 500 | 250 | 58000 | 0 | 5 | 0 | ... | 4 | 5 | 3 | 3 | 1 | 0 | 0 | 3 | 0 | 0 |
| **108375** | 175.00 | 8 | 0 | 500 | 500 | 59700 | 18200 | 3 | 0 | 3 | ... | 4 | 0 | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| **27174** | 14.90 | 11 | 10 | 371 | 170 | 37300 | 600 | 3 | 2 | 4 | ... | 3 | 1 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| **57525** | 169.91 | 5 | 4 | 500 | 500 | 22650 | 21100 | 0 | 2 | 0 | ... | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **73383** | 106.91 | 5 | 13 | 500 | 500 | 10500 | 500 | 4 | 2 | 3 | ... | 5 | 1 | 5 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |

> 5 rows × 86 columns
> 

```python
X_test.head()
```

|  | **time_left** | **ct_score** | **t_score** | **ct_health** | **t_health** | **ct_money** | **t_money** | **ct_helmets** | **t_helmets** | **ct_defuse_kits** | **...** | **ct_grenade_flashbang** | **t_grenade_flashbang** | **ct_grenade_smokegrenade** | **t_grenade_smokegrenade** | **ct_grenade_incendiarygrenade** | **t_grenade_incendiarygrenade** | **ct_grenade_molotovgrenade** | **t_grenade_molotovgrenade** | **ct_grenade_decoygrenade** | **t_grenade_decoygrenade** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **77680** | 69.94 | 1 | 3 | 467 | 497 | 12850 | 2700 | 3 | 5 | 3 | ... | 5 | 4 | 3 | 3 | 3 | 0 | 0 | 3 | 0 | 0 |
| **30145** | 174.94 | 6 | 11 | 500 | 500 | 9500 | 18700 | 0 | 1 | 0 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **94364** | 159.19 | 9 | 15 | 500 | 500 | 1350 | 4200 | 2 | 5 | 2 | ... | 2 | 5 | 4 | 5 | 3 | 0 | 0 | 5 | 0 | 0 |
| **7026** | 74.96 | 10 | 13 | 399 | 390 | 5050 | 8950 | 4 | 4 | 2 | ... | 4 | 4 | 1 | 2 | 3 | 0 | 0 | 4 | 0 | 0 |
| **101522** | 54.91 | 12 | 5 | 109 | 353 | 6100 | 12600 | 0 | 4 | 0 | ... | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |

> 5 rows × 86 columns
> 

## **Seleção de atributos**

1. Inicializo o SelectKBest, definindo que serão selecionados os 10 atributos mais relevantes para prever a variável-alvo, utilizando a função `f_classif` para avaliar cada atributo.

```python
seletor = SelectKBest(score_func = f_classif, k = 10)
```

2. Em seguida, utilizo o método `fit_transform()` para identificar e selecionar os 10 melhores atributos do conjunto de treinamento. Depois, aplico o método `transform()` ao conjunto de teste, mantendo apenas os mesmos atributos selecionados anteriormente.

```python
X_train_sel = seletor.fit_transform(X_train, y_train)
X_test_sel = seletor.transform(X_test)
```

3. Por fim, utilizo o método `get_support()` para visualizar as colunas selecionadas.

```python
cols = seletor.get_support(indices = True)
X_train.iloc[:, cols]
```

|  | **ct_health** | **ct_helmets** | **t_helmets** | **ct_defuse_kits** | **ct_players_alive** | **t_weapon_ak47** | **ct_weapon_awp** | **ct_weapon_m4a4** | **ct_grenade_flashbang** | **ct_grenade_smokegrenade** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **66505** | 492 | 0 | 5 | 0 | 5 | 3 | 1 | 2 | 4 | 3 |
| **108375** | 500 | 3 | 0 | 3 | 5 | 0 | 2 | 0 | 4 | 4 |
| **27174** | 371 | 3 | 2 | 4 | 4 | 0 | 2 | 0 | 3 | 2 |
| **57525** | 500 | 0 | 2 | 0 | 5 | 1 | 0 | 0 | 0 | 0 |
| **73383** | 500 | 4 | 2 | 3 | 5 | 0 | 0 | 0 | 5 | 5 |
| **...** | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| **34358** | 500 | 0 | 1 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| **69581** | 100 | 1 | 3 | 1 | 1 | 2 | 0 | 0 | 1 | 1 |
| **62670** | 500 | 0 | 5 | 0 | 5 | 2 | 1 | 2 | 3 | 3 |
| **37704** | 496 | 4 | 5 | 3 | 5 | 3 | 1 | 4 | 4 | 2 |
| **93110** | 400 | 1 | 2 | 1 | 4 | 2 | 1 | 0 | 0 | 0 |

> 91807 rows × 10 columns
> 

Aqui finaliza a seleção de atributos.

# **Treinamento**

1. Crio o modelo de árvore de decisão. Em seguida, utilizo o método `fit()` para treinar o modelo com os atributos selecionados e seus respectivos resultados reais:

```python
modelo_dtc = DecisionTreeClassifier(random_state = 2000)
modelo_dtc.fit(X_train_sel, y_train)
```

> 
> 
> 
> DecisionTreeClassifier
> 
> - Parameters
>     
>     
>     |  | random_state | 2000 |
>     | --- | --- | --- |
>     |  | criterion | 'gini' |
>     |  | splitter | 'best' |
>     |  | max_depth | None |
>     |  | min_samples_split | 2 |
>     |  | min_samples_leaf | 1 |
>     |  | min_weight_fraction_leaf | 0.0 |
>     |  | max_features | None |
>     |  | max_leaf_nodes | None |
>     |  | min_impurity_decrease | 0.0 |
>     |  | class_weight | None |
>     |  | ccp_alpha | 0.0 |
>     |  | monotonic_cst | None |
> - Fitted attributes
>     
>     
>     | **Name** | **Type** | **Value** |
>     | --- | --- | --- |
>     | classes_ | ndarrayobject | ['CT','T'] |
>     | feature_importances_ | ndarrayfloat64 | [0.15,0.14,0.18,...,0.07,0.15,0.07] |
>     | max_features_ | int | 10 |
>     | n_classes_ | int64 | np.int64(2) |
>     | n_features_in_ | int | 10 |
>     | n_outputs_ | int | 1 |
>     | tree_ | Tree | <sklearn.tree...00152C37F39F0> |
2. Realizo as previsões com o modelo treinado e exibo as 10 primeiras:

```python
y_predict = modelo_dtc.predict(X_test_sel)
print(y_predict[:10])
```

> ['T' 'T' 'T' 'CT' 'T' 'CT' 'T' 'CT' 'T' 'T']
> 

Comparação de resultados:

```python
print(pd.DataFrame({
    'Real': y_test,
    '| DTC': y_predict
}).head(10))
```

> Real | DTC
77680     T     T
30145     T     T
94364     T     T
7026     CT    CT
101522    T     T
50029    CT    CT
41732    CT     T
43972    CT    CT
17367     T     T
66447     T     T
> 
3. Utilizo a função `accuracy_score()` para comparar as previsões do modelo com os resultados reais. A acurácia obtida foi de aproximadamente **73,41%**, ou seja, o modelo acertou cerca de 73 de cada 100 previsões realizadas no conjunto de teste.

```python
acy = accuracy_score(y_test, y_predict)
print('Acurácia do modelo:', acy)
```

> Acurácia do modelo: 0.7341110348658628
