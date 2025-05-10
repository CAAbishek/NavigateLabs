import streamlit as st
from PIL import Image
import torch
import pandas as pd
import plotly.express as px
from Text import load_text_model, predict_with_metrics
from Image import load_image_model, generate_caption_with_latency


st.title("Medical Diagnosis Assistant")
mode = st.selectbox("Choose Input Type", ["Text", "Image"])

@st.cache_resource
def get_text_model():
    return load_text_model()

@st.cache_resource
def get_image_model():
    return load_image_model()

if mode == "Text":
    tokenizer, base_model, quant_model = get_text_model()
    symptoms = st.text_area("Describe your symptoms:")

    if st.button("Compare Inference (Original vs Quantized)"):

        if not symptoms.strip():
            st.warning("Please enter your symptoms.")
        else:
            
            base_result = predict_with_metrics(symptoms, tokenizer, base_model)
            quant_result = predict_with_metrics(symptoms, tokenizer, quant_model)

            
            st.subheader("Original Model")
            st.write(f"Prediction: **{base_result['prediction']}**")

            st.subheader("Quantized Model")
            st.write(f"Prediction: **{quant_result['prediction']}**")

            
            base_memory = max(base_result.get('memory', 0), 0)  
            quant_memory = max(quant_result.get('memory', 0), 0)  

            
            df = pd.DataFrame({
                "Metric": ["Inference Time (s)", "Memory Usage (MB)"] * 2,
                "Model": ["Original"] * 2 + ["Quantized"] * 2,
                "Value": [
                    base_result['latency'], base_memory,
                    quant_result['latency'], quant_memory
                ]
            })

            
            fig = px.bar(
                df, x="Metric", y="Value", color="Model", barmode="group",
                title="Original vs Quantized Performance (Text Model)",
                text="Value"
            )

            fig.update_layout(
                xaxis_title="Metric",
                yaxis_title="Value",
                legend_title="Model",
                template="plotly_dark",
                font=dict(size=14)
            )

            st.plotly_chart(fig, use_container_width=True)





elif mode == "Image":
    image_model, image_tokenizer, image_processor, gen_args = get_image_model()
    uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Compare Report (Original vs Quantized)"):
            
            original_caption, original_latency, original_throughput, original_memory = generate_caption_with_latency(
                image, image_model, image_tokenizer, image_processor, gen_args)

            
            quantized_model = torch.quantization.quantize_dynamic(
                image_model, {torch.nn.Linear}, dtype=torch.qint8
            ).eval()
            quantized_caption, quantized_latency, quantized_throughput, quantized_memory = generate_caption_with_latency(
                image, quantized_model, image_tokenizer, image_processor, gen_args)

            
            st.subheader("Original Model")
            st.write(f"Generated Caption: **{original_caption}**")
            st.write(f"Inference Time: **{original_latency:.4f} seconds**")
            st.write(f"Throughput: **{original_throughput:.2f} images/sec**")
            st.write(f"Memory Usage: **{original_memory:.2f} MB**")

            st.subheader("Quantized Model")
            st.write(f"Generated Caption: **{quantized_caption}**")
            st.write(f"Inference Time: **{quantized_latency:.4f} seconds**")
            st.write(f"Throughput: **{quantized_throughput:.2f} images/sec**")
            st.write(f"Memory Usage: **{quantized_memory:.2f} MB**")

            
            improvement = 0
            if original_latency > quantized_latency:
                improvement = ((original_latency - quantized_latency) / original_latency) * 100

            st.info(f"⚙️ Speed Improvement: **{improvement:.2f}%**")

            
            df_image = pd.DataFrame({
                'Metric': ['Inference Time (s)', 'Throughput (img/sec)', 'Memory Usage (MB)'],
                'Original': [original_latency, original_throughput, original_memory],
                'Quantized': [quantized_latency, quantized_throughput, quantized_memory]
            }).melt(id_vars='Metric', var_name='Model', value_name='Value')

            st.subheader("Model Performance Comparison (Image)")
            fig_image = px.bar(df_image, x='Metric', y='Value', color='Model', barmode='group',
                               color_discrete_map={'Original': 'blue', 'Quantized': 'green'})
            st.plotly_chart(fig_image)
