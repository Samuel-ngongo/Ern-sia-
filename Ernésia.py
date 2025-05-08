# PREVISÃO INTELIGENTE COM REGRESSÃO
st.header("Previsão Inteligente")

if len(data_list) >= 15:
    last_15 = data_list[-15:]
    x = np.arange(len(last_15)).reshape(-1, 1)
    y = np.array(last_15).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    next_index = np.array([[len(last_15)]])
    prediction = model.predict(next_index)[0][0]

    # Ajuste para estimar intervalo
    min_pred = prediction * 0.85
    mean_pred = prediction

    st.success(f"Valor mínimo provável: **{min_pred:.2f}**")
    st.info(f"Valor médio provável: **{mean_pred:.2f}**")

    # Mostrar gráfico com linha de tendência
    st.subheader("Gráfico com linha de tendência")
    fig, ax = plt.subplots()
    ax.plot(range(len(last_15)), last_15, 'o-', label='Dados')
    ax.plot(range(len(last_15)) + [len(last_15)], 
            np.append(model.predict(x), prediction), '--', label='Tendência')
    ax.set_xlabel("Posição")
    ax.set_ylabel("Valor")
    ax.legend()
    st.pyplot(fig)

else:
    st.warning("Forneça pelo menos 15 valores para ativar a previsão inteligente.")
