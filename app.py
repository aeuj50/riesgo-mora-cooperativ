# -*- coding: utf-8 -*-
"""
Evaluador de riesgo de mora - Cooperativa de ahorro y credito
Caso integral de referencia · ADSE-II

La aplicacion NO usa TensorFlow. El modelo entrenado se exporto a modelo.npz
como un conjunto de matrices, y la inferencia se hace con NumPy: cada capa densa
es una multiplicacion de matrices, una suma y una activacion.

Dependencias: streamlit y numpy. Nada mas.
"""

from pathlib import Path

import numpy as np
import streamlit as st

AQUI = Path(__file__).resolve().parent

st.set_page_config(page_title="Riesgo de mora", page_icon="📉", layout="centered")

# ------------------------------------------------------------------ el modelo
@st.cache_resource
def cargar_modelo(ruta=None):
    ruta = ruta or (AQUI / "modelo.npz")
    d = np.load(ruta, allow_pickle=True)
    return {
        "columnas": [str(c) for c in d["columnas"]],
        "n_capas": int(d["n_capas"][0]),
        "W": [d[f"W{j}"] for j in range(int(d["n_capas"][0]))],
        "b": [d[f"b{j}"] for j in range(int(d["n_capas"][0]))],
        "media": d["media"],
        "escala": d["escala"],
        "medianas": d["medianas"],
    }


def predecir(m, x):
    """Inferencia completa en NumPy. Replica exactamente al modelo de Keras.

    Reproduce los tres pasos del cuaderno en el mismo orden: imputar los faltantes
    con las medianas del conjunto de entrenamiento, estandarizar, y aplicar las
    capas densas.
    """
    x = np.where(np.isnan(x), m["medianas"], x)
    h = (x - m["media"]) / m["escala"]
    for j in range(m["n_capas"]):
        h = h @ m["W"][j] + m["b"][j]
        h = np.maximum(h, 0.0) if j < m["n_capas"] - 1 else 1.0 / (1.0 + np.exp(-h))
    return float(h.ravel()[0])


modelo = cargar_modelo()

# ------------------------------------------------------------------ interfaz
st.title("Riesgo de mora a 90 días")
st.caption(
    "Cooperativa de ahorro y crédito · caso de referencia del módulo "
    "Análisis de Datos Socioeconómicos y Empresariales II"
)

st.info(
    "**Modelo entrenado con datos sintéticos.** No corresponde a ninguna "
    "institución real y no debe usarse para decidir sobre personas reales. "
    "Es material docente.",
    icon="ℹ️",
)

with st.form("socio"):
    st.subheader("Datos del socio")
    c1, c2 = st.columns(2)
    with c1:
        edad = st.number_input("Edad", 21, 78, 26,
                               help="El modelo se entrenó con socios de 21 a 78 años.")
        antiguedad = st.number_input("Antigüedad como socio (meses)", 6, 300, 14)
        ingreso = st.number_input("Ingreso mensual (USD)", 460.0, 6500.0, 700.0, step=10.0)
        ahorros = st.number_input("Saldo de ahorros (USD)", 0.0, 45000.0, 300.0, step=50.0)
        creditos_previos = st.number_input("Créditos previos", 0, 10, 0)
    with c2:
        monto = st.number_input("Monto del crédito (USD)", 500.0, 45000.0, 5000.0, step=100.0)
        plazo = st.selectbox("Plazo (meses)", [12, 18, 24, 36, 48, 60], index=2)
        tasa = st.number_input("Tasa de interés anual (%)", 9.5, 24.9, 17.5, step=0.1)
        mora_12m = st.number_input("Días de mora máximos en 12 meses", 0, 120, 12)
        refinanciamientos = st.number_input("Refinanciamientos", 0, 5, 0)

    st.subheader("Características del crédito")
    c3, c4 = st.columns(2)
    with c3:
        destino = st.selectbox("Destino", ["consumo", "microempresa", "vivienda", "educacion"])
        sector = st.selectbox("Sector laboral",
                              ["dependiente", "independiente", "comercio", "agricola"])
    with c4:
        zona = st.selectbox("Zona", ["urbana", "rural"])
        garantia = st.selectbox("Garantía", ["quirografaria", "prendaria", "hipotecaria"])

    enviar = st.form_submit_button("Evaluar", use_container_width=True)

# ------------------------------------------------------------------ resultado
if enviar:
    # la cuota se calcula igual que en el cuaderno: sistema francés
    i = tasa / 100 / 12
    cuota = monto * i / (1 - (1 + i) ** (-plazo))
    # mismo recorte que aplicó el generador: [0,02 · 0,95]
    carga = round(min(max(cuota / ingreso, 0.02), 0.95), 4)

    valores = {
        "edad": edad,
        "antiguedad_socio_meses": antiguedad,
        "ingreso_mensual": ingreso,
        "saldo_ahorros": ahorros,
        "num_creditos_previos": creditos_previos,
        "monto_credito": monto,
        "plazo_meses": plazo,
        "tasa_interes_anual": tasa,
        "cuota_mensual": cuota,
        "carga_financiera": carga,
        "dias_mora_max_12m": mora_12m,
        "num_refinanciamientos": refinanciamientos,
        "destino_credito_educacion": int(destino == "educacion"),
        "destino_credito_microempresa": int(destino == "microempresa"),
        "destino_credito_vivienda": int(destino == "vivienda"),
        "sector_laboral_comercio": int(sector == "comercio"),
        "sector_laboral_dependiente": int(sector == "dependiente"),
        "sector_laboral_independiente": int(sector == "independiente"),
        "zona_urbana": int(zona == "urbana"),
        "garantia_prendaria": int(garantia == "prendaria"),
        "garantia_quirografaria": int(garantia == "quirografaria"),
    }
    x = np.array([[valores[c] for c in modelo["columnas"]]], dtype="float64")
    p = predecir(modelo, x)

    st.divider()
    a, b, c = st.columns(3)
    a.metric("Probabilidad de mora", f"{p:.1%}")
    b.metric("Cuota mensual", f"$ {cuota:,.2f}")
    c.metric("Carga financiera", f"{carga:.1%}")

    # Los dos cortes salen de la capacidad operativa, no de números redondos:
    # 0,65 deja aproximadamente el 10 % más riesgoso de la cartera, que es a
    # quienes el equipo alcanza a llamar; 0,30 deja el 20 %.
    if p >= 0.65:
        st.error("**Prioridad alta.** Está en el 10 % más riesgoso de la cartera. "
                 "Incluir en la lista de contacto de este mes.", icon="🔴")
    elif p >= 0.30:
        st.warning("**Prioridad media.** Está en el 20 % más riesgoso. "
                   "Revisar si queda capacidad de contacto.", icon="🟠")
    else:
        st.success("**Prioridad baja.** Por debajo del 20 % más riesgoso de la "
                   "cartera. Seguimiento habitual.", icon="🟢")

    st.progress(min(p, 1.0))

    with st.expander("Cómo leer este resultado"):
        st.markdown(
            f"""
El modelo **ordena** socios por riesgo; no decide nada por sí solo. La cifra de
{p:.1%} sirve para comparar a este socio con el resto de la cartera, no como un
pronóstico individual cerrado.

Dos referencias del cuaderno que ayudan a interpretarla:

- La tasa de mora de la cartera completa es del **18 %**. Todo lo que esté muy
  por encima merece atención.
- El riesgo sube con fuerza cuando la cuota supera el **30 % del ingreso**.
  Aquí la carga financiera es del **{carga:.1%}**.

**Lo que el modelo no sabe:** una enfermedad, la pérdida del empleo o una mala
cosecha. Nada de eso está en los datos, y todo eso causa mora.
"""
        )

    with st.expander("Para probar los efectos no lineales"):
        st.markdown(
            """
El caso que viene cargado es un socio de 26 años, con 14 meses de antigüedad y una
cuota que se lleva el 35 % de su ingreso. Da un riesgo cercano a la media de la
cartera, que es del 18 %.

Tres experimentos, con los valores ya comprobados. Cambia **una sola cosa por vez** y
vuelve a pulsar Evaluar.

**1. La edad tiene forma de U, no de rampa.**

| Edad | Riesgo |
|---|---|
| 26 (el que viene cargado) | 17 % |
| 44 | 1 % |
| 78 | 92 % |

El riesgo mínimo está a mitad del recorrido, no en un extremo. Un modelo lineal solo
puede trazar una recta sobre esta variable: acierte donde acierte, se equivoca en el
otro extremo.

**2. El plazo también.**

| Plazo | Riesgo |
|---|---|
| 24 meses (el que viene cargado) | 17 % |
| 36 meses | 6 % |
| 60 meses | 89 % |

Los plazos cortos aprietan la cuota, los largos acumulan exposición, y el tramo
intermedio es el más seguro.

**3. Un atraso pesa según a quién le pase.**

Sube los días de mora a 60 y deja todo lo demás como está: el riesgo pasa de 17 % a
**76 %**. Ahora, con esos mismos 60 días de mora, sube la antigüedad a 200 meses: baja
a **4 %**.

El mismo atraso, en un socio de dieciséis años en la cooperativa, casi no significa
nada. Eso es una interacción, y ningún modelo lineal la representa sin que alguien la
escriba a mano.
"""
        )

    with st.expander("Qué hay detrás de esta aplicación"):
        capas = " → ".join(str(W.shape[1]) for W in modelo["W"])
        st.markdown(
            f"""
No hay ningún servidor de modelos ni TensorFlow corriendo. Todo el modelo es un
archivo de **{len(modelo['W'])} matrices de pesos** con arquitectura
{modelo['W'][0].shape[0]} → {capas}, más dos vectores con la media y la escala
del estandarizador. Pesa **57 KB**.

La predicción es esta operación, repetida cuatro veces:

```python
h = h @ W + b
h = np.maximum(h, 0)     # ReLU en las capas ocultas
h = 1 / (1 + np.exp(-h)) # sigmoide en la salida
```

Es la misma cuenta que se escribe a mano en la primera semana del módulo para un
perceptrón. Una red profunda es eso, repetido.

**Un límite que conviene tener presente.** El modelo solo vio socios dentro de ciertos
rangos. Fuera de ellos extrapola, y devuelve una cifra con toda la apariencia de estar
calculada aunque no se apoye en ningún dato. Por eso los campos están acotados al
rango con el que se entrenó.
"""
        )
