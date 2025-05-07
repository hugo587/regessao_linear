
"""
Algoritmo de Regressão Linear Simples usando dados de Excel
Autor: Hugo Vieira
Descrição: Este script realiza uma regressão linear simples com base em dados lidos de um arquivo Excel.
"""

# ======================
# Importação de Bibliotecas
# ======================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ======================
#  Leitura do Arquivo Excel
# ======================

# Importação dos arquivos
df = pd.read_excel(
    'C:\\Users\\hugol\\Desktop\\dadosExemplo.xlsx', engine='openpyxl')

# Visualização inicial dos dados
print(" Visualizando os dados:")
print(df.head())

# ======================
# 🔍 Pré-processamento
# ======================
#

X = df[['X']]  # O scikit-learn espera um array 2D para X
y = df['Y']

# ======================
#  Divisão em Conjunto de Treinamento e Teste
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ======================
# ⚙️ Criação e Treinamento do Modelo
# ======================
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# ======================
# Avaliação do Modelo
# ======================
y_pred = modelo.predict(X_test)

print("\n Avaliação do Modelo:")
print(f"Coeficiente Angular (Slope): {modelo.coef_[0]}")
print(f"Coeficiente Linear (Intercept): {modelo.intercept_}")
print(f"Erro Quadrático Médio (MSE): {mean_squared_error(y_test, y_pred):.2f}")
print(f"Coeficiente de Determinação (R²): {r2_score(y_test, y_pred):.2f}")

# ======================
#  Visualização Gráfica
# ======================
plt.scatter(X_test, y_test, color='blue', label='Dados Reais')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linha de Regressão')
plt.title('Regressão Linear Simples')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
