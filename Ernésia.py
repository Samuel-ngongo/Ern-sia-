import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Previsão Inteligente - Ernésia", layout="centered")

st.title("Previsão Inteligente - Queda do Aviator")
st.markdown("Insira os valores (ex: 1.95, 2.00, etc). O sistema aprenderá e preverá o próximo valor provável.")

# Session state para armazenar os dados
if 'dados' not in st.session_state:
    st.session_state.dados = []

# Entrada de dados
novo_valor = st.number_input("Novo valor", min_value=0.0, step=0.01, format="%.2f")
if st.button("Adicionar"):
    st.session_state.dados.append(novo_valor)

# Botão para apagar tudo
if st.button("Apagar Dados"):
    st.session_state.dados.clear()
    st.success("Todos os dados foram apagados.")

dados = st.session_state.dados

# Exibição dos dados
if dados:
    st.subheader("Últimos Valores")
    st.write(dados)

    # Transformar para DataFrame
    df = pd.DataFrame(dados, columns=["Valor"])
    df["Índice"] = range(len(df))

    # Gráfico com regressão
    fig, ax = plt.subplots()
    ax.plot(df["Índice"], df["Valor"], marker="o", label="Valores Inseridos")

    # Cores por classificação
    for i, v in enumerate(dados):
        cor = "red" if v < 2 else "green" if v > 3 else "orange"
        ax.scatter(i, v, color=cor)

    # Regressão Linear
    if len(dados) >= 5:
        ultimos_dados = dados[-15:] if len(dados) > 15 else dados
        X = np.arange(len(ultimos_dados)).reshape(-1, 1)
        y = np.array(ultimos_dados).reshape(-1, 1)
        modelo = LinearRegression()
        modelo.fit(X, y)
        proximo_indice = len(ultimos_dados)
        previsao = modelo.predict([[proximo_indice]])[0][0]
        minimo = previsao * 0.85  # margem de segurança -15%

        # Mostrar a linha de tendência
        y_pred = modelo.predict(X)
        ax.plot(X, y_pred, color="blue", linestyle="--", label="Tendência")

        st.subheader("Previsão Inteligente")
        st.success(f"Valor mínimo provável: {minimo:.2f}")
        st.info(f"Valor médio provável: {previsao:.2f}")

    ax.set_xlabel("Entrada")
    ax.set_ylabel("Valor")
    ax.set_title("Tendência dos Últimos Valores")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Insira alguns valores para começar.")
