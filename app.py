from turtle import st
from ml import modelos


st.title('🍎 Previsão de Maçãs 🍎')

with st.form('form-1'):
    size = st.number_input('Size', min_value=-5, max_value=5, step=0.01)

    enviar = st.form_submit_button('Previsão')

    if enviar:
        for modelo in modelos.values():
            previsao = modelo.predict([])