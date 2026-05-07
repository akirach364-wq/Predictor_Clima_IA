# Predictor Climático — Chilpancingo, Guerrero

**Materia:** Inteligencia Artificial  
**Alumno:** Josué Emanuel Díaz Rodríguez  
**Institución:** Instituto Tecnológico de Chilpancingo  
**Fecha:** Mayo 2025

---

## ¿De qué trata este proyecto?

El objetivo es construir un sistema de predicción meteorológica para Chilpancingo usando machine learning. La idea surgió de que la mayoría de los modelos del tiempo que existen son de escala nacional y no reflejan bien el comportamiento local del clima en la región montañosa de Guerrero.

Lo que hace el proyecto concretamente es:
- Usar datos históricos diarios de variables climáticas (temperatura, lluvia, humedad, viento, presión)
- Transformar esa serie temporal en un problema de regresión supervisada
- Entrenar y comparar tres modelos de ML
- Generar un pronóstico a 30 días con alerta de precipitación

---

## Estructura del proyecto

```
ProyectoClima/
│
├── generar_dataset.py       # Genera el dataset con parámetros reales de Chilpancingo
├── predictor_clima.py       # Entrenamiento, evaluación y pronóstico
├── requirements.txt         # Dependencias necesarias
│
├── datos/
│   ├── chilpancingo_clima_2018_2024.csv   # Dataset generado
│   └── pronostico_30dias.csv             # Resultado del pronóstico
│
└── graficas/
    ├── 01_serie_historica.png
    ├── 02_patron_mensual.png
    ├── 03_comparacion_modelos.png
    ├── 04_real_vs_predicho.png
    ├── 05_importancia_features.png
    ├── 06_correlacion.png
    └── 07_pronostico_30dias.png
```

---

## Dataset

Los datos se generaron sintéticamente pero calibrados con los parámetros climáticos reales de Chilpancingo, Guerrero, tomando como referencia la información pública del Servicio Meteorológico Nacional (SMN).

**Variables incluidas:**

| Variable | Descripción | Unidad |
|---|---|---|
| temp_media | Temperatura promedio del día | °C |
| temp_maxima | Temperatura máxima registrada | °C |
| temp_minima | Temperatura mínima registrada | °C |
| precipitacion_mm | Lluvia acumulada en el día | mm |
| humedad_relativa | Porcentaje de humedad del aire | % |
| velocidad_viento | Velocidad del viento | km/h |
| presion_hpa | Presión atmosférica | hPa |
| nubosidad_oktas | Cobertura de nubes (escala 0-8) | oktas |

**Características climáticas de Chilpancingo usadas como referencia:**
- Altitud: ~1,350 msnm
- Tipo de clima: Semicalido subhumedo (Aw según Köppen)
- Temperatura media anual: ~21°C
- Temporada de lluvias: mayo a octubre (concentra más del 85% de la precipitación)
- Precipitación anual promedio: 900-1100 mm

---

## Metodología

### 1. Pipeline ETL (generar_dataset.py)

El script genera un registro diario para el periodo 2018-2024 (2,557 días). Cada variable se modela con distribuciones estadísticas apropiadas:

- **Temperatura:** normal con media mensual basada en datos históricos + tendencia de calentamiento gradual
- **Precipitación:** probabilidad mensual de lluvia + distribución exponencial para la intensidad (con eventos extremos ocasionales)
- **Humedad:** correlacionada positivamente con precipitación y negativamente con temperatura
- **Viento:** distribución exponencial con ajuste por tormenta y mes

### 2. Ingeniería de Variables (Feature Engineering)

La serie temporal se convierte en un problema de regresión estándar mediante:

**Lags (rezagos):** valores exactos de días anteriores (1, 2, 3, 7 y 14 días atrás)

**Ventanas móviles:** estadísticas calculadas sobre los últimos 7 o 14 días:
- Media móvil (tendencia)
- Desviación estándar (volatilidad)
- Máximo (detección de picos)
- Mínimo

**Variables temporales cíclicas:** el mes y el día del año se codifican con seno y coseno para que el modelo entienda que diciembre está cerca de enero (no al contrario como ocurriría con números enteros).

**Variables complementarias:** presión, humedad y precipitación del día anterior también se usan como predictores.

### 3. División del dataset

Se usa una división temporal estricta (no aleatoria) para evitar data leakage:
- **80% entrenamiento:** datos de 2018 a finales de 2023
- **20% validación:** datos de 2024

### 4. Modelos comparados

| Modelo | Mecanismo | Fortaleza |
|---|---|---|
| **Random Forest** (principal) | Bagging con árboles de decisión en paralelo | Robusto ante ruido, no asume distribución |
| **Gradient Boosting** | Árboles secuenciales que corrigen errores | Alta precisión en patrones complejos |
| **Ridge Regression** | Regresión lineal con regularización L2 | Línea base interpretable |

### 5. Métricas de evaluación

- **MAE (Error Absoluto Medio):** error promedio en las mismas unidades que la variable. Para temperatura, un MAE de 1.8°C significa que en promedio nos equivocamos por ese margen.
- **RMSE (Raíz del Error Cuadrático Medio):** penaliza más los errores grandes. Útil para detectar predicciones muy malas.
- **R² (Coeficiente de determinación):** qué porcentaje de la varianza explica el modelo. 1.0 = perfecto, 0 = igual que predecir siempre el promedio.

### 6. Pronóstico a 30 días (Forecast Rolling)

El pronóstico funciona de manera iterativa:
1. Se toman los últimos datos reales disponibles
2. Se predice el día siguiente
3. Esa predicción se agrega al historial
4. Se repite hasta completar 30 días

Cada día predicho incluye una alerta de precipitación:
- 🟢 **Baja:** menos de 5 mm esperados
- 🟡 **Media:** entre 5 y 20 mm
- 🔴 **Alta:** más de 20 mm (riesgo de tormenta)

---

## Resultados obtenidos

### Temperatura media

| Modelo | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|
| Random Forest | ~1.79 | ~2.22 | ~0.32 |
| Gradient Boosting | ~1.85 | ~2.31 | ~0.27 |
| Ridge Regression | ~1.84 | ~2.29 | ~0.28 |

### Precipitación

| Modelo | MAE (mm) | RMSE (mm) | R² |
|---|---|---|---|
| Random Forest | ~6.39 | ~12.42 | ~0.18 |
| Gradient Boosting | ~6.52 | ~12.85 | ~0.12 |

El Random Forest fue el mejor modelo en ambos casos. Los valores de R² para precipitación son bajos, lo cual es esperado ya que la lluvia es un fenómeno con alta variabilidad estocástica difícil de capturar con datos de series de tiempo simples.

---

## Cómo ejecutar

**1. Instalar dependencias:**
```bash
pip install -r requirements.txt
```

**2. Ejecutar el predictor (genera datos + entrena modelos + hace pronóstico):**
```bash
python predictor_clima.py
```

**3. Si solo se quiere generar/explorar el dataset:**
```bash
python generar_dataset.py
```

Al finalizar se habrán creado:
- `datos/chilpancingo_clima_2018_2024.csv` — dataset completo
- `datos/pronostico_30dias.csv` — pronóstico con alertas
- `graficas/` — carpeta con 7 gráficas de análisis y resultados

---

## Limitaciones y trabajo futuro

El modelo actual tiene algunas limitaciones importantes que vale la pena mencionar honestamente:

- Los datos son sintéticos. Idealmente se usarían los CSVs descargados directamente del portal de datos abiertos del SMN para la estación de Chilpancingo.
- La precipitación es inherentemente difícil de predecir con solo datos históricos de series de tiempo. Modelos más avanzados usan también datos de presión de sistemas atmosféricos, imágenes satelitales o reanálisis climáticos (ERA5, MERRA-2).
- El forecast rolling acumula error con cada día predicho, por lo que la confiabilidad cae significativamente después de 7-10 días.

Posibles mejoras:
- Integrar datos reales del SMN o del servicio CONAGUA
- Agregar variables como índice ENSO (El Niño/La Niña) que afecta fuertemente las lluvias en Guerrero
- Explorar modelos de redes neuronales como LSTM para series temporales
- Implementar intervalos de confianza en las predicciones

---

## Referencias

- Servicio Meteorológico Nacional — smn.conagua.gob.mx
- Köppen, W. (1936). Das geographische System der Klimate
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830
- Datos abiertos SMN: https://smn.conagua.gob.mx/es/climatologia/informacion-climatologica/informacion-estadistica-climatologica
