import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io
import pandas as pd

# --- Constantes da Aplicação ---

# Caminho para o seu modelo Keras treinado (conforme a estrutura de pastas)
CAMINHO_MODELO = 'models/model_1_cnn.keras'

# O tamanho da imagem que o MobileNetV2 espera (do seu notebook)
TAMANHO_IMG = (224, 224)

# O mapeamento de classes (do output da célula 3 do seu notebook)
# Ordem extraída do notebook: train_images.class_indices
CLASSES_PEIXES = [
    'Black Sea Sprat', 
    'Gilt-Head Bream', 
    'Hourse Mackerel', 
    'Red Mullet', 
    'Red Sea Bream', 
    'Sea Bass', 
    'Shrimp', 
    'Striped Red Mullet', 
    'Trout'
]

# Limiar de confiança para classificar como "Outros"
# (Ajuste conforme necessário. 70% é um bom ponto de partida)
LIMIAR_CONFIANCA = 0.70

# --- Funções ---

@st.cache_resource
def carregar_modelo():
    """
    Carrega o modelo Keras treinado. 
    Usamos @st.cache_resource para que o modelo seja carregado apenas uma vez.
    """
    try:
        model = load_model(CAMINHO_MODELO)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.error("Verifique se o arquivo 'model_1_cnn.keras' está na pasta 'models/'.")
        return None

def preprocessar_imagem(image_pil):
    """
    Pré-processa a imagem PIL para o formato que o MobileNetV2 espera.
    """
    # 1. Converte para RGB (caso seja PNG com canal alfa ou P&B)
    image_rgb = image_pil.convert('RGB')
    
    # 2. Redimensiona para 224x224
    image_resized = image_rgb.resize(TAMANHO_IMG)
    
    # 3. Converte para array numpy
    image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    
    # 4. Adiciona uma dimensão de "batch" (lote)
    # O modelo espera (1, 224, 224, 3) e não (224, 224, 3)
    image_batch = np.expand_dims(image_array, axis=0)
    
    # 5. Aplica a função de pré-processamento específica do MobileNetV2
    image_preprocessed = preprocess_input(image_batch)
    
    return image_preprocessed

# --- Interface Principal do Streamlit ---

def main():
    st.set_page_config(page_title="Classificador de Peixes", layout="centered")
    st.title("🐟 Classificador de Espécies de Peixes")
    
    # Carrega o modelo
    model = carregar_modelo()
    
    if model is None:
        st.stop()

    st.write("Faça o upload de uma imagem de peixe para classificação.")

    # Componente de upload de arquivo
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # 1. Ler e exibir a imagem
            image = Image.open(uploaded_file)
            st.image(image, caption='Imagem Carregada', use_column_width=True)
            
            # 2. Pré-processar a imagem
            processed_image = preprocessar_imagem(image)
            
            # 3. Fazer a predição
            with st.spinner('Classificando...'):
                prediction = model.predict(processed_image)
            
            # 4. Obter a confiança e a classe predita
            confidence = np.max(prediction)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = CLASSES_PEIXES[predicted_class_index]
            
            st.markdown("---")
            
            # 5. Lógica para "Outros"
            if confidence >= LIMIAR_CONFIANCA:
                st.success(f"**Predição:** {predicted_class_name}")
                st.metric(label="Nível de Confiança", value=f"{confidence:.2%}")
            else:
                st.warning(f"**Predição:** Outros")
                st.metric(label="Nível de Confiança (Baixo)", value=f"{confidence:.2%}")
                st.info("O modelo não está confiante de que esta imagem pertença a uma das 9 classes conhecidas.")

            # Opcional: Mostrar detalhes das probabilidades
            with st.expander("Ver todas as probabilidades"):
                prob_df = pd.DataFrame(prediction.T, index=CLASSES_PEIXES, columns=["Probabilidade"])
                st.dataframe(prob_df.style.format("{:.2%}"))

        except Exception as e:
            st.error(f"Erro ao processar a imagem: {e}")

if __name__ == "__main__":
    main()
