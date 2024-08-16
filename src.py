import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Carregando os dados (substitui 'caminho_do_arquivo' pelo caminho real do arquivo CSV)
dados = pd.read_csv('caminho_do_arquivo/dados_educacionais.csv')

# Visualizando as primeiras linhas dos dados
print(dados.head())

# Analisando a distribuição das notas dos alunos
plt.figure(figsize=(10, 6))
sns.histplot(dados['nota_final'], kde=True, color='blue')
plt.title('Distribuição das Notas Finais')
plt.xlabel('Nota Final')
plt.ylabel('Frequência')
plt.show()

# Analisando a relação entre a frequência e a nota final
plt.figure(figsize=(10, 6))
sns.scatterplot(x='frequencia', y='nota_final', data=dados, hue='evasao', palette='coolwarm')
plt.title('Relação entre Frequência e Nota Final')
plt.xlabel('Frequência (%)')
plt.ylabel('Nota Final')
plt.show()

# Prevendo risco de evasão escolar com RandomForest
# Separando os dados em variáveis independentes (X) e dependente (y)
X = dados[['nota_final', 'frequencia']]
y = dados['evasao']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criando o modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Fazendo previsões
y_pred = modelo.predict(X_test)

# Avaliando o modelo
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Importância das variáveis
importancias = modelo.feature_importances_
indices = pd.Series(importancias, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=indices, y=indices.index)
plt.title('Importância das Variáveis no Modelo de Previsão de Evasão')
plt.xlabel('Importância')
plt.ylabel('Variáveis')
plt.show()
