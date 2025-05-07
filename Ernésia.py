import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Ernésia - Previsão Aviator", layout="wide")

st.title("Previsão Aviator com Inteligência Simples")

# Sessão de dados
if "dados" not in st.session_state:
    st.session_state["dados"] = []

# Entrada de novo valor
col1, col2 = st.columns([3, 1])
with col1:
    novo_valor = st.number_input("Digite o valor do Aviator:", min_value=0.0, step=0.01)
with col2:
    if st.button("Adicionar"):
        st.session_state["dados"].append(novo_valor)

# Botões extras
col3, col4 = st.columns([1, 1])
with col3:
    if st.button("Apagar dados"):
        st.session_state["dados"] = []
with col4:
    if st.button("Baixar CSV") and st.session_state["dados"]:
        df = pd.DataFrame(st.session_state["dados"], columns=["Valor"])
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Clique para baixar", data=csv, file_name="dados_aviator.csv", mime="text/csv")

# Mostrar dados
dados = st.session_state["dados"]
if dados:
    df = pd.DataFrame(dados, columns=["Valor"])
    st.subheader("Histórico de Dados")
    st.dataframe(df)

    # Previsão
    ultimos = dados[-10:]
    media = np.mean(ultimos)
    minimo = np.min(ultimos)
    st.subheader("Previsão")
    st.markdown(f"**Valor médio (últimos 10):** `{media:.2f}`")
    st.markdown(f"**Valor mínimo (últimos 10):** `{minimo:.2f}`")

    # Gráfico
    st.subheader("Gráfico de Tendência")
    x = np.arange(len(ultimos)).reshape(-1, 1)
    y = np.array(ultimos).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    tendencia = model.predict(x)

    cores = ["green" if val >= 2.0 else "red" for val in ultimos]

    fig, ax = plt.subplots()
    ax.bar(range(len(ultimos)), ultimos, color=cores)
    ax.plot(range(len(ultimos)), tendencia, color="blue", linestyle="--", label="Tendência")
    ax.set_title("Últimos 10 valores")
    ax.set_ylabel("Valor")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Nenhum valor inserido ainda.")
