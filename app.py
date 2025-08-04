# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import numpy as np
import joblib
import torch
import gradio as gr
import os
import re

# === Cargar modelos entrenados ===
modelo_riesgo = joblib.load("modelo_riesgo.pkl")
modelo_violencia = joblib.load("modelo_tipo_violencia.pkl")
modelo_medida = joblib.load("modelo_tipo_medida.pkl")
codificadores = joblib.load("codificadores_label_encoder.pkl")
modelo_vector = SentenceTransformer("Snowflake/snowflake-arctic-embed-xs")

# === Cargar modelo de lenguaje Mistral 7B Instruct ===
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=os.environ.get("HF_TOKEN")
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=os.environ.get("HF_TOKEN"),
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# === CSS personalizado Externado ===
css_externado = """
@import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');
body {
    background-color: #004225;
    font-family: 'Roboto', sans-serif;
}
.gr-button {
    background-color: #004225 !important;
    color: white !important;
}
.gr-button:hover {
    background-color: white !important;
}
.gr-input, .gr-output {
    background-color: white !important;
}
"""

# === Frases prototipo para verificador semántico ===
frases_fisica = [
    "Me golpeó con el puño cerrado",
    "Me pateó",
    "Me lanzó contra la pared",
    "Me estranguló",
    "Me fracturó una costilla",
    "Me tiró al piso violentamente"
]
frases_sexual = [
    "Me obligó a tener relaciones sexuales",
    "Me tocó sin consentimiento",
    "Me violó",
    "Me forzó a tener sexo",
    "Me agredió sexualmente"
]
frases_economica = [
    "No me da dinero",
    "Me quita mi salario",
    "Me retiene los recursos económicos",
    "No me deja trabajar",
    "Me obliga a depender económicamente"
]
frases_psicologica = [
    "Me insulta todo el tiempo",
    "Me amenaza constantemente",
    "Me humilla frente a los niños",
    "Me hace sentir inútil",
    "Me controla emocionalmente"
]
frases_negligencia = [
    "No me brinda atención médica",
    "No cuida a los hijos",
    "Me deja sin alimentos",
    "No se preocupa por mi salud",
    "Ignora mis necesidades básicas"
]

# === Embeddings de referencia ===
embeds_fisica = modelo_vector.encode(frases_fisica)
embeds_sexual = modelo_vector.encode(frases_sexual)
embeds_economica = modelo_vector.encode(frases_economica)
embeds_psicologica = modelo_vector.encode(frases_psicologica)
embeds_negligencia = modelo_vector.encode(frases_negligencia)

# === Verificador semántico completo ===
def verificar_semantico(descripcion):
    emb_desc = modelo_vector.encode(descripcion)
    tipos_detectados = []
    if max(util.cos_sim(emb_desc, embeds_fisica)[0]) > 0.8:
        tipos_detectados.append("física")
    if max(util.cos_sim(emb_desc, embeds_sexual)[0]) > 0.9:
        tipos_detectados.append("sexual")
    if max(util.cos_sim(emb_desc, embeds_economica)[0]) > 0.85:
        tipos_detectados.append("económica")
    if max(util.cos_sim(emb_desc, embeds_psicologica)[0]) > 0.85:
        tipos_detectados.append("psicológica")
    if max(util.cos_sim(emb_desc, embeds_negligencia)[0]) > 0.85:
        tipos_detectados.append("negligencia")
    return tipos_detectados

# === FUNCIÓN PRINCIPAL ===
def predecir_con_recomendacion(edad, genero, hijos, convivencia_agresor, consumo_sustancias, apoyo_familiar, descripcion):
    # Codificar variables tabulares
    vector_tabular = np.array([
        int(edad),
        int(hijos),
        codificadores["genero"].transform([genero])[0],
        0, 0, 0,
        codificadores["convivencia_agresor"].transform([convivencia_agresor])[0],
        codificadores["consumo_sustancias"].transform([consumo_sustancias])[0],
        codificadores["apoyo_familiar"].transform([apoyo_familiar])[0]
    ])

    # Vectorizar descripción
    vector_desc = modelo_vector.encode([descripcion])[0]
    entrada = np.concatenate([vector_tabular, vector_desc])

    # Modelos clásicos
    riesgo_cod = modelo_riesgo.predict([entrada])[0]
    tipo_violencia_cod = modelo_violencia.predict([entrada])[0]
    tipo_medida_cod = modelo_medida.predict([entrada])[0]

    # Decodificación
    riesgo = codificadores["riesgo"].inverse_transform([riesgo_cod])[0]
    tipo_violencia_pred = codificadores["tipo_violencia"].inverse_transform([tipo_violencia_cod])[0]
    tipo_medida = codificadores["tipo_medida"].inverse_transform([tipo_medida_cod])[0]

    # Verificador semántico
    tipos_semantico = verificar_semantico(descripcion)

    # Unir modelo + semántico sin duplicados
    tipos_combinados = list(set([tipo_violencia_pred] + tipos_semantico))
    tipos_str = ", ".join(tipos_combinados)

    messages = [
    {
        "role": "system",
        "content": "Eres un jurista colombiano experto en derecho de familia y violencia intrafamiliar. Redacta Autos judiciales reales conforme a la Ley colombiana."
    },
    {
        "role": "user",
        "content": f"""
Con base en la siguiente información del caso, redacta un Auto de Medida Provisional formal, jurídico y completo. Usa el estilo y estructura real de una Comisaría de Familia en Colombia.

Ciudad: Bogotá D.C.
Tipo de violencia: {tipos_str}
Nivel de riesgo: {riesgo}
Medida de protección sugerida: {tipo_medida}
Descripción de los hechos: {descripcion}

El Auto debe tener:

1. Encabezado institucional: “LA COMISARÍA DE FAMILIA DE BOGOTÁ D.C., en uso de sus facultades legales...”
2. Sección CONSIDERACIONES con redacción continua (no listas), que incluya:
   - Hechos relevantes del caso.
   - Valoración del riesgo.
   - Fundamento jurídico: Artículo 5 de la Ley 575 de 2000 y artículos 16 y 17 de la Ley 1257 de 2008.
   - Justificación jurídica específica de cada medida de protección, con literal aplicable (ej: “conforme al literal b)”).
3. Sección RESUELVE, en orden formal:
   - PRIMERO: Admitir la solicitud.
   - SEGUNDO, TERCERO, CUARTO, etc.: Una medida de protección por párrafo, con redacción jurídica y numeración en mayúsculas.
   - ÚLTIMO: Cúmplase y notifíquese.
4. Cierre con la palabra CÚMPLASE. en mayúscula y sin firma.

No repitas fundamentos ni uses términos como “nominar” o “nombrar medidas”.
No menciones violencia económica ni sexual si no está explícitamente descrita.
Redacta todo en máximo 300 palabras.
"""
    }
]

    # Aplicar chat template
    encoded = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    # Generar salida
    generated_ids = model.generate(
        encoded,
        max_new_tokens=1200,
        do_sample=False,
        temperature=0.2,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decodificar
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    auto_completo = decoded[0].strip()

    # Recortar desde la segunda aparición de "CONSIDERACIONES"
    ocurrencias = [i for i in range(len(auto_completo)) if auto_completo.startswith("CONSIDERACIONES", i)]
    if len(ocurrencias) >= 2:
        auto_redactado = auto_completo[ocurrencias[1]:]
    elif len(ocurrencias) == 1:
        auto_redactado = auto_completo[ocurrencias[0]:]
    else:
        auto_redactado = auto_completo

    # Eliminar líneas duplicadas
    lineas = auto_redactado.splitlines()
    lineas_sin_duplicados = []
    vistos = set()
    for linea in lineas:
        if linea.strip() and linea.strip() not in vistos:
            lineas_sin_duplicados.append(linea)
            vistos.add(linea.strip())
    auto_redactado = "\n".join(lineas_sin_duplicados)

    # Renumerar medidas en números romanos
    romanos = ["I","II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    lineas = auto_redactado.splitlines()
    contador = 0
    for i, linea in enumerate(lineas):
        if any(linea.strip().startswith(pref) for pref in ["PRIMERO","SEGUNDO", "TERCERO", "CUARTO", "QUINTO", "SEXTO", "SÉPTIMO", "OCTAVO"]):
            numeral = romanos[contador]
            contenido = linea.split(":", 1)[1].strip()
            lineas[i] = f"{numeral}: {contenido}"
            contador += 1
    auto_redactado = "\n".join(lineas)

    return tipos_str, riesgo, tipo_medida, auto_redactado

# === Interfaz Gradio ===
with gr.Blocks(css=css_externado) as interfaz:
    # Logo alineado a la izquierda
    gr.Markdown("""
    <div style='text-align:left;'>
        <img src='https://huggingface.co/spaces/km1lo/LEGALFAMI/raw/main/Logo%20externado.jpg' style='height: 60px;'/>
    </div>
    """)

    # Título centrado en blanco
    gr.Markdown("""
    <h1 style='color:#FFFFFF; text-align:center; font-family: Roboto, sans-serif;'>
    LEGALFAMI – Asistente Legal con Razonamiento Jurídico
    </h1>
    """)
    
    # Subtítulo
    gr.Markdown("""
    <p style='text-align:center; color:#FFFFFF;'>
    Predice tipo de violencia, nivel de riesgo, medida de protección y redacta un Auto conforme a la Ley 575 Art.5.
    </p>
    """)

    with gr.Row():
        with gr.Column():
            edad = gr.Slider(18, 65, value=30, label="Edad de la Víctima")
            genero = gr.Radio(["F", "M"], label="Género")
            hijos = gr.Slider(0, 5, step=1, value=1, label="Número de Hijos")
            convivencia_agresor = gr.Radio(["sí", "no"], label="¿Convive con el Agresor?")
            consumo_sustancias = gr.Radio(["sí", "no"], label="¿Hay Consumo de Sustancias?")
            apoyo_familiar = gr.Radio(["sí", "no"], label="¿Tiene Apoyo Familiar?")
            descripcion = gr.Textbox(lines=5, placeholder="Describa detalladamente el caso de violencia...", label="Descripción del Caso")
            boton = gr.Button("🔍 Analizar Caso")

        with gr.Column():
            tipo_violencia_out = gr.Textbox(label="🛑 Tipo de Violencia Detectada", interactive=False)
            riesgo_out = gr.Textbox(label="⚠️ Nivel de Riesgo Estimado", interactive=False)
            medida_out = gr.Textbox(label="🧾 Tipo de Medida de Protección Sugerida", interactive=False)
            recomendacion_out = gr.Textbox(label="📋 Auto de Medida Provisional", lines=12, interactive=False)

    boton.click(
        fn=predecir_con_recomendacion,
        inputs=[edad, genero, hijos, convivencia_agresor, consumo_sustancias, apoyo_familiar, descripcion],
        outputs=[tipo_violencia_out, riesgo_out, medida_out, recomendacion_out]
    )

interfaz.launch()