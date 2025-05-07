import streamlit as st import matplotlib.pyplot as plt import numpy as np import pandas as pd from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Previsão Aviator", layout="wide")

Título elegante

st.markdown(""" <h1 style='text-align: center; color: #0e4c92;'>Previsor Inteligente Aviator</h1> <p style='text-align: center;'>Insira os resultados e veja as previsões com base em análise de dados.</p> """, unsafe_allow_html=True)

Inicializa o estado da sessão

if "valores" not in st.session_state: st.session_state.valores = []

Entrada do usuário em coluna

col1, col2 = st.columns([2, 1]) with col1: novo_valor = st.number_input("Digite o valor do Aviator (ex: 2.3):", min_value=0.0, format="%.2f") with col2: if st.button("Adicionar Valor"): st.session_state.valores.append(novo_valor)

Botão para limpar dados

data_col1, data_col2 = st.columns([1, 1]) with data_col1: if st.button("Apagar Todos os Dados"): st.session_state.valores.clear() st.success("Dados apagados com sucesso!")

Função para preparar dados

def preparar_dados(dados, janela=10): X, y = [], [] for i in range(len(dados) - janela): X.append(dados[i:i + janela]) y.append(dados[i + janela]) return np.array(X), np.array(y)

Função para prever

@st.cache_data

def prever_proximo_valor_intervalo(dados, janela=10): if len(dados) <= janela: return None, None X, y = preparar_dados(dados, janela) modelo = LinearRegression() modelo.fit(X, y) entrada = np.array(dados[-janela:]).reshape(1, -1) previsao = modelo.predict(entrada)[0] minimo = min(dados[-janela:]) return minimo, previsao

Exibir dados e previsão

if st.session_state.valores: st.subheader("Dados Inseridos") df = pd.DataFrame(st.session_state.valores, columns=["Valores"]) st.dataframe(df.tail(15), use_container_width=True)

min_valor, media_valor = prever_proximo_valor_intervalo(st.session_state.valores)

if media_valor is not None:
    cor = "green" if media_valor >= 2.0 else "red"
    st.markdown(f"""
        <div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px;'>
        <b>Previsão Próxima Rodada:</b><br>
        <span style='color:{cor}; font-size: 22px;'>Média: {media_valor:.2f} | Mínimo: {min_valor:.2f}</span>
        </div>
    """, unsafe_allow_html=True)

    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(st.session_state.valores, marker='o', label='Valores Reais')
    ax.axhline(y=media_valor, color='blue', linestyle='--', label=f'Média Prevista')
    ax.set_title("Tendência de Valores")
    ax.set_xlabel("Rodadas")
    ax.set_ylabel("Valor")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Insira pelo menos 11 valores para ativação da inteligência.")

else: st.warning("Nenhum dado inserido ainda. Digite acima para começar.")

