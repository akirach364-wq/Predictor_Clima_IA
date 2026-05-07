"""
Modelo de prediccion meteorologica para Chilpancingo, Guerrero.

El objetivo es predecir la temperatura media y precipitacion del dia
siguiente usando los datos historicos de los ultimos dias como entrada
al modelo. Esto es util porque nos permite anticipar si va a llover
o cuanto calor va a hacer, que es info relevante para planeacion
en salud publica, agricultura y transporte en la region.

Enfoque: Transformacion de serie temporal a problema de regresion supervisada.
Modelos comparados:
    - Random Forest Regressor (modelo principal)
    - Gradient Boosting Regressor
    - Ridge Regression (baseline)

Metricas de evaluacion:
    - MAE (Error Absoluto Medio)
    - RMSE (Raiz del Error Cuadratico Medio)
    - R² (Coeficiente de determinacion)

Autor: Josue Emanuel Diaz Rodriguez
Materia: Inteligencia Artificial
Instituto Tecnologico de Chilpancingo
Fecha: Mayo 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# importar el generador de datos
from generar_dataset import crear_dataset_completo

# configuracion basica de matplotlib para que se vean bien las graficas
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


# ─────────────────────────────────────────────────
# 1. INGENIERIA DE VARIABLES (FEATURE ENGINEERING)
# ─────────────────────────────────────────────────

def construir_features(df, variable_objetivo="temp_media", ventana_lags=7):
    """
    Transforma la serie temporal en un dataset de regresion supervisada.

    La idea es sencilla: si quiero predecir la temperatura de mañana,
    lo que mas me sirve son los datos de los ultimos dias. Por eso
    creamos 'lags' (rezagos) y estadisticas de ventana movil.

    Parametros:
        df: DataFrame con los datos climaticos
        variable_objetivo: columna que queremos predecir
        ventana_lags: cuantos dias hacia atras usar como features

    Retorna:
        X: matriz de features
        y: vector objetivo
        df_features: dataframe completo con todas las columnas creadas
    """
    df_feat = df.copy()
    col = variable_objetivo

    # --- Lags: los valores exactos de dias anteriores ---
    # son lo mas directo, "ayer fue X, que sera hoy?"
    for lag in [1, 2, 3, 7, 14]:
        df_feat[f"{col}_lag{lag}"] = df_feat[col].shift(lag)

    # --- Rolling statistics: tendencia reciente ---
    # el promedio de los ultimos 7 dias captura tendencias graduales
    df_feat[f"{col}_media_7d"] = df_feat[col].shift(1).rolling(7).mean()
    df_feat[f"{col}_media_14d"] = df_feat[col].shift(1).rolling(14).mean()
    df_feat[f"{col}_std_7d"] = df_feat[col].shift(1).rolling(7).std()
    df_feat[f"{col}_max_7d"] = df_feat[col].shift(1).rolling(7).max()
    df_feat[f"{col}_min_7d"] = df_feat[col].shift(1).rolling(7).min()

    # --- Variables temporales ---
    # el mes y el dia del año son muy importantes para clima estacional
    df_feat["mes_sin"] = np.sin(2 * np.pi * df_feat["mes"] / 12)
    df_feat["mes_cos"] = np.cos(2 * np.pi * df_feat["mes"] / 12)
    df_feat["dia_sin"] = np.sin(2 * np.pi * df_feat["dia_del_año"] / 365)
    df_feat["dia_cos"] = np.cos(2 * np.pi * df_feat["dia_del_año"] / 365)

    # --- Variables climaticas complementarias como features ---
    # la presion del dia anterior es un buen predictor de lluvia
    df_feat["presion_lag1"] = df_feat["presion_hpa"].shift(1)
    df_feat["humedad_lag1"] = df_feat["humedad_relativa"].shift(1)
    df_feat["precip_lag1"] = df_feat["precipitacion_mm"].shift(1)
    df_feat["precip_media7d"] = df_feat["precipitacion_mm"].shift(1).rolling(7).mean()

    # eliminar filas con NaN que quedaron por los lags
    df_feat = df_feat.dropna()

    # columnas que usaremos como features (excluyendo la variable objetivo y otras redundantes)
    columnas_excluir = [
        "temp_media", "temp_maxima", "temp_minima",
        "precipitacion_mm", "humedad_relativa",
        "velocidad_viento", "presion_hpa", "nubosidad_oktas",
        "tipo_dia", "año", "mes", "dia", "dia_semana", "dia_del_año"
    ]

    feature_cols = [c for c in df_feat.columns if c not in columnas_excluir]

    X = df_feat[feature_cols]
    y = df_feat[variable_objetivo]

    return X, y, df_feat, feature_cols


# ─────────────────────────────────────────────
# 2. ENTRENAMIENTO Y EVALUACION DE MODELOS
# ─────────────────────────────────────────────

def evaluar_modelo(modelo, X_train, X_test, y_train, y_test, nombre):
    """
    Entrena un modelo y calcula todas las metricas.
    Use esta funcion para no repetir codigo en cada modelo.
    """
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print(f"\n{'─'*40}")
    print(f"Modelo: {nombre}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    return {
        "nombre": nombre,
        "modelo": modelo,
        "predicciones": pred,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def entrenar_modelos_temperatura(df):
    """
    Entrena y compara tres modelos para predecir temperatura media.
    Retorna el mejor modelo y los resultados de evaluacion.
    """
    print("\n" + "="*50)
    print("PREDICCION DE TEMPERATURA MEDIA")
    print("="*50)

    X, y, df_feat, feat_cols = construir_features(df, "temp_media")

    # division temporal: no revolver futuro con pasado
    # usamos los primeros 80% para entrenar, los ultimos 20% para probar
    corte = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:corte], X.iloc[corte:]
    y_train, y_test = y.iloc[:corte], y.iloc[corte:]

    print(f"\nDatos de entrenamiento: {len(X_train)} dias")
    print(f"Datos de prueba: {len(X_test)} dias")
    print(f"Features usadas: {len(feat_cols)}")

    # escalar para Ridge (los arboles no lo necesitan pero no hace daño)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    resultados = []

    # --- Random Forest (modelo principal) ---
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    resultados.append(evaluar_modelo(rf, X_train, X_test, y_train, y_test, "Random Forest"))

    # --- Gradient Boosting ---
    gb = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    resultados.append(evaluar_modelo(gb, X_train, X_test, y_train, y_test, "Gradient Boosting"))

    # --- Ridge (baseline lineal) ---
    ridge = Ridge(alpha=1.0)
    resultados.append(evaluar_modelo(ridge, X_train_sc, X_test_sc, y_train, y_test, "Ridge Regression"))

    # seleccionar el mejor por R²
    mejor = max(resultados, key=lambda r: r["r2"])
    print(f"\n>>> Mejor modelo: {mejor['nombre']} (R² = {mejor['r2']:.4f})")

    return resultados, X_test, y_test, df_feat, feat_cols, mejor


def entrenar_modelos_precipitacion(df):
    """
    Mismo proceso pero para predecir precipitacion.
    Este es mas dificil porque la lluvia tiene mucha variabilidad.
    """
    print("\n" + "="*50)
    print("PREDICCION DE PRECIPITACION")
    print("="*50)

    X, y, df_feat, feat_cols = construir_features(df, "precipitacion_mm")

    corte = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:corte], X.iloc[corte:]
    y_train, y_test = y.iloc[:corte], y.iloc[corte:]

    print(f"\nDatos de entrenamiento: {len(X_train)} dias")
    print(f"Datos de prueba: {len(X_test)} dias")

    resultados = []

    rf_precip = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    resultados.append(evaluar_modelo(rf_precip, X_train, X_test, y_train, y_test, "Random Forest"))

    gb_precip = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.75,
        random_state=42
    )
    resultados.append(evaluar_modelo(gb_precip, X_train, X_test, y_train, y_test, "Gradient Boosting"))

    mejor = max(resultados, key=lambda r: r["r2"])
    print(f"\n>>> Mejor modelo precipitacion: {mejor['nombre']} (R² = {mejor['r2']:.4f})")

    return resultados, X_test, y_test, df_feat, mejor


# ─────────────────────────────────────────────────────
# 3. VISUALIZACIONES
# ─────────────────────────────────────────────────────

def grafica_serie_historica(df):
    """Muestra la temperatura y precipitacion historica de Chilpancingo."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Datos Climaticos Historicos — Chilpancingo, Guerrero (2018-2024)",
                 fontsize=14, fontweight="bold", y=0.98)

    # temperatura
    axes[0].plot(df.index, df["temp_media"], color="#c0392b", linewidth=0.6, alpha=0.7, label="Temp. media")
    media_movil = df["temp_media"].rolling(30).mean()
    axes[0].plot(df.index, media_movil, color="#922b21", linewidth=2, label="Media móvil 30d")
    axes[0].set_ylabel("Temperatura (°C)", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(8, 35)

    # precipitacion
    axes[1].bar(df.index, df["precipitacion_mm"], color="#2980b9", alpha=0.6, width=1, label="Precipitación diaria")
    precip_movil = df["precipitacion_mm"].rolling(30).mean()
    axes[1].plot(df.index, precip_movil, color="#1a5276", linewidth=1.8, label="Media móvil 30d")
    axes[1].set_ylabel("Precipitación (mm)", fontsize=11)
    axes[1].set_xlabel("Fecha", fontsize=11)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("graficas/01_serie_historica.png", bbox_inches="tight")
    plt.close()
    print("Grafica guardada: graficas/01_serie_historica.png")


def grafica_patron_mensual(df):
    """Boxplots de temperatura y precipitacion por mes para ver estacionalidad."""
    meses_nombres = ["Ene","Feb","Mar","Abr","May","Jun",
                     "Jul","Ago","Sep","Oct","Nov","Dic"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Patron Mensual del Clima — Chilpancingo, Guerrero",
                 fontsize=13, fontweight="bold")

    # temperatura por mes
    datos_temp = [df[df["mes"] == m]["temp_media"].values for m in range(1, 13)]
    bp = axes[0].boxplot(datos_temp, labels=meses_nombres, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#e74c3c")
        patch.set_alpha(0.6)
    axes[0].set_title("Temperatura Media por Mes", fontsize=11)
    axes[0].set_ylabel("°C")
    axes[0].set_xlabel("Mes")

    # precipitacion por mes
    precip_mensual = df.groupby("mes")["precipitacion_mm"].sum() / 7  # promedio anual
    colores = ["#aed6f1" if m not in [6,7,8,9,10] else "#2980b9" for m in range(1,13)]
    axes[1].bar(meses_nombres, precip_mensual, color=colores, edgecolor="white")
    axes[1].set_title("Precipitacion Promedio Mensual (mm)", fontsize=11)
    axes[1].set_ylabel("mm")
    axes[1].set_xlabel("Mes")

    plt.tight_layout()
    plt.savefig("graficas/02_patron_mensual.png", bbox_inches="tight")
    plt.close()
    print("Grafica guardada: graficas/02_patron_mensual.png")


def grafica_comparacion_modelos(resultados_temp, y_test_temp):
    """Compara el rendimiento de los modelos con barras de metricas."""
    nombres = [r["nombre"] for r in resultados_temp]
    mae_vals = [r["mae"] for r in resultados_temp]
    r2_vals = [r["r2"] for r in resultados_temp]
    rmse_vals = [r["rmse"] for r in resultados_temp]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Comparacion de Modelos — Prediccion de Temperatura",
                 fontsize=13, fontweight="bold")

    colores = ["#27ae60", "#e67e22", "#8e44ad"]

    # MAE
    bars = axes[0].bar(nombres, mae_vals, color=colores, edgecolor="white")
    axes[0].set_title("MAE (menor = mejor)", fontsize=10)
    axes[0].set_ylabel("°C")
    for bar, val in zip(bars, mae_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # RMSE
    bars = axes[1].bar(nombres, rmse_vals, color=colores, edgecolor="white")
    axes[1].set_title("RMSE (menor = mejor)", fontsize=10)
    axes[1].set_ylabel("°C")
    for bar, val in zip(bars, rmse_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # R²
    bars = axes[2].bar(nombres, r2_vals, color=colores, edgecolor="white")
    axes[2].set_title("R² (mayor = mejor)", fontsize=10)
    axes[2].set_ylim(0, 1.05)
    for bar, val in zip(bars, r2_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    for ax in axes:
        ax.set_xticklabels(nombres, rotation=15, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig("graficas/03_comparacion_modelos.png", bbox_inches="tight")
    plt.close()
    print("Grafica guardada: graficas/03_comparacion_modelos.png")


def grafica_real_vs_predicho(mejor_temp, X_test, y_test, df_feat):
    """
    Muestra real vs predicho para los ultimos meses del periodo de prueba.
    Esta grafica es la mas importante para interpretar que tan bien funciona.
    """
    pred = mejor_temp["predicciones"]
    nombre = mejor_temp["nombre"]

    # tomar solo los ultimos 90 dias para que la grafica sea legible
    n = 90
    fechas = df_feat.index[-len(y_test):][-n:]
    real_plot = y_test.values[-n:]
    pred_plot = pred[-n:]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle(f"Real vs Predicho — {nombre}\nPrediccion de Temperatura en Chilpancingo",
                 fontsize=13, fontweight="bold")

    # serie temporal
    axes[0].plot(fechas, real_plot, label="Real", color="#c0392b", linewidth=1.5, alpha=0.9)
    axes[0].plot(fechas, pred_plot, label="Predicho", color="#2980b9",
                 linewidth=1.5, linestyle="--", alpha=0.9)
    axes[0].fill_between(fechas, real_plot, pred_plot, alpha=0.15, color="#8e44ad")
    axes[0].set_ylabel("Temperatura (°C)")
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30)

    # scatter plot real vs predicho
    axes[1].scatter(real_plot, pred_plot, alpha=0.4, color="#27ae60", s=20)
    min_val = min(real_plot.min(), pred_plot.min()) - 1
    max_val = max(real_plot.max(), pred_plot.max()) + 1
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Prediccion perfecta")
    axes[1].set_xlabel("Temperatura Real (°C)")
    axes[1].set_ylabel("Temperatura Predicha (°C)")
    axes[1].set_title(f"Scatter Real vs Predicho  |  R² = {mejor_temp['r2']:.4f}", fontsize=10)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("graficas/04_real_vs_predicho.png", bbox_inches="tight")
    plt.close()
    print("Grafica guardada: graficas/04_real_vs_predicho.png")


def grafica_importancia_features(mejor_temp, feature_cols):
    """
    Si el modelo es Random Forest o Gradient Boosting, podemos ver
    cuales features fueron mas importantes para predecir.
    """
    if not hasattr(mejor_temp["modelo"], "feature_importances_"):
        print("El modelo no soporta feature importances, saltando grafica.")
        return

    importancias = mejor_temp["modelo"].feature_importances_
    df_imp = pd.DataFrame({"feature": feature_cols, "importancia": importancias})
    df_imp = df_imp.sort_values("importancia", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 7))
    colores = ["#e74c3c" if i >= len(df_imp)-3 else "#5d6d7e" for i in range(len(df_imp))]
    ax.barh(df_imp["feature"], df_imp["importancia"], color=colores)
    ax.set_title(f"Top 15 Variables Mas Importantes — {mejor_temp['nombre']}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importancia relativa")

    plt.tight_layout()
    plt.savefig("graficas/05_importancia_features.png", bbox_inches="tight")
    plt.close()
    print("Grafica guardada: graficas/05_importancia_features.png")


def grafica_mapa_calor_correlacion(df):
    """Correlacion entre las variables climaticas."""
    cols_num = ["temp_media", "temp_maxima", "temp_minima",
                "precipitacion_mm", "humedad_relativa",
                "velocidad_viento", "presion_hpa", "mes"]

    corr = df[cols_num].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax, linewidths=0.5,
                annot_kws={"size": 8})
    ax.set_title("Correlacion entre Variables Climaticas — Chilpancingo",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig("graficas/06_correlacion.png", bbox_inches="tight")
    plt.close()
    print("Grafica guardada: graficas/06_correlacion.png")


# ─────────────────────────────────────────────────────
# 4. PREDICCION A FUTURO (los proximos 30 dias)
# ─────────────────────────────────────────────────────

def predecir_30_dias(df, mejor_modelo_temp, mejor_modelo_precip,
                     feat_cols_temp, feat_cols_precip):
    """
    Genera una prediccion rodante para los proximos 30 dias.
    El 'forecast rolling' significa que cada dia predicho se usa
    como input para predecir el dia siguiente.
    """
    print("\n" + "="*50)
    print("PRONOSTICO A 30 DIAS")
    print("="*50)

    modelo_t = mejor_modelo_temp["modelo"]
    modelo_p = mejor_modelo_precip["modelo"]

    # usamos el ultimo mes de datos reales como base
    historico = df.copy()

    predicciones = []
    fecha_ultimo = df.index[-1]

    for dia_futuro in range(1, 31):
        fecha_pred = fecha_ultimo + pd.Timedelta(days=dia_futuro)
        mes_pred = fecha_pred.month
        dia_año_pred = fecha_pred.timetuple().tm_yday

        # construir el vector de features para este dia
        # usamos el historico acumulado (real + ya predicho)
        X_t, y_t, df_ext, fc_t = construir_features(historico, "temp_media")
        X_p, y_p, _,      fc_p = construir_features(historico, "precipitacion_mm")

        if len(X_t) == 0:
            break

        # tomar la ultima fila (el dia mas reciente disponible)
        x_temp_input = X_t.iloc[[-1]][feat_cols_temp] if feat_cols_temp else X_t.iloc[[-1]]
        x_prec_input = X_p.iloc[[-1]][feat_cols_precip] if feat_cols_precip else X_p.iloc[[-1]]

        temp_pred = modelo_t.predict(x_temp_input)[0]
        prec_pred = max(0, modelo_p.predict(x_prec_input)[0])

        # agregar prediccion al historico para el siguiente ciclo
        nueva_fila = {
            "año": fecha_pred.year, "mes": mes_pred,
            "dia": fecha_pred.day, "dia_semana": fecha_pred.weekday(),
            "dia_del_año": dia_año_pred,
            "temp_media": round(temp_pred, 1),
            "temp_maxima": round(temp_pred + 5, 1),
            "temp_minima": round(temp_pred - 6, 1),
            "precipitacion_mm": round(prec_pred, 1),
            "humedad_relativa": 65.0,  # valor neutro
            "velocidad_viento": 8.0,
            "presion_hpa": 877.0,
            "nubosidad_oktas": 3.0,
            "tipo_dia": "Predicho",
        }
        historico = pd.concat([historico, pd.DataFrame([nueva_fila], index=[fecha_pred])])

        predicciones.append({
            "fecha": fecha_pred.strftime("%Y-%m-%d"),
            "temp_predicha": round(temp_pred, 1),
            "precip_predicha": round(prec_pred, 1),
            "alerta": "🔴 Alta" if prec_pred > 20 else ("🟡 Media" if prec_pred > 5 else "🟢 Baja"),
        })

    df_pred = pd.DataFrame(predicciones)
    print("\nPronostico generado para los proximos 30 dias:\n")
    print(df_pred.to_string(index=False))

    # grafica del pronostico
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle("Pronostico Climatico — Chilpancingo, Guerrero (30 dias)",
                 fontsize=13, fontweight="bold")

    fechas_pred = pd.to_datetime(df_pred["fecha"])
    fechas_hist = df.index[-60:]

    axes[0].plot(fechas_hist, df.loc[fechas_hist, "temp_media"],
                 color="#c0392b", linewidth=1.5, label="Historico real")
    axes[0].plot(fechas_pred, df_pred["temp_predicha"],
                 color="#2980b9", linewidth=2, linestyle="--", label="Pronostico", marker="o", markersize=3)
    axes[0].axvline(fecha_ultimo, color="gray", linestyle=":", linewidth=1.5)
    axes[0].set_ylabel("Temperatura (°C)")
    axes[0].legend()

    axes[1].bar(fechas_hist, df.loc[fechas_hist, "precipitacion_mm"],
                color="#85c1e9", alpha=0.7, label="Historico real")
    axes[1].bar(fechas_pred, df_pred["precip_predicha"],
                color="#1a5276", alpha=0.7, label="Pronostico")
    axes[1].axvline(fecha_ultimo, color="gray", linestyle=":", linewidth=1.5, label="Inicio pronostico")
    axes[1].set_ylabel("Precipitacion (mm)")
    axes[1].set_xlabel("Fecha")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("graficas/07_pronostico_30dias.png", bbox_inches="tight")
    plt.close()
    print("\nGrafica guardada: graficas/07_pronostico_30dias.png")

    return df_pred


# ─────────────────────────────────────────────────────
# PUNTO DE ENTRADA PRINCIPAL
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("graficas", exist_ok=True)
    os.makedirs("datos", exist_ok=True)
    os.makedirs("modelos", exist_ok=True)

    print("=" * 60)
    print("  PREDICTOR CLIMATICO — CHILPANCINGO, GUERRERO")
    print("  Josue Emanuel Diaz Rodriguez")
    print("  Materia: Inteligencia Artificial")
    print("  Instituto Tecnologico de Chilpancingo")
    print("=" * 60)

    # cargar o generar el dataset
    ruta_datos = "datos/chilpancingo_clima_2018_2024.csv"
    if os.path.exists(ruta_datos):
        print("\nCargando dataset existente...")
        df = pd.read_csv(ruta_datos, index_col=0, parse_dates=True)
    else:
        df = crear_dataset_completo()
        df.to_csv(ruta_datos)

    print(f"Dataset cargado: {len(df)} registros ({df.index[0].date()} — {df.index[-1].date()})")

    # graficas exploratorias
    print("\nGenerando graficas exploratorias...")
    grafica_serie_historica(df)
    grafica_patron_mensual(df)
    grafica_mapa_calor_correlacion(df)

    # entrenar modelos de temperatura
    res_temp, X_test_t, y_test_t, df_feat_t, feat_cols_t, mejor_temp = entrenar_modelos_temperatura(df)

    # graficas de resultados temperatura
    grafica_comparacion_modelos(res_temp, y_test_t)
    grafica_real_vs_predicho(mejor_temp, X_test_t, y_test_t, df_feat_t)
    grafica_importancia_features(mejor_temp, feat_cols_t)

    # entrenar modelos de precipitacion
    res_precip, X_test_p, y_test_p, df_feat_p, mejor_precip = entrenar_modelos_precipitacion(df)

    # pronostico a 30 dias
    _, _, _, feat_cols_p = construir_features(df, "precipitacion_mm")
    df_pronostico = predecir_30_dias(df, mejor_temp, mejor_precip, feat_cols_t, feat_cols_p)
    df_pronostico.to_csv("datos/pronostico_30dias.csv", index=False)

    print("\n" + "="*60)
    print("EJECUCION COMPLETADA")
    print(f"  Graficas generadas: graficas/")
    print(f"  Dataset guardado: {ruta_datos}")
    print(f"  Pronostico: datos/pronostico_30dias.csv")
    print("="*60)