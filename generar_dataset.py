"""
generar_dataset.py
------------------
Este script genera el dataset de condiciones climaticas para Chilpancingo, Guerrero.
Tomé como referencia los datos historicos de la estacion meteorologica de la SMN
(Servicio Meteorologico Nacional) para el periodo 2018-2024.

Las variables se ajustaron a los rangos reales del clima de la ciudad:
- Altitud: ~1,350 msnm
- Clima: Semicalido subhumedo (Aw)
- Temporada de lluvias: Mayo - Octubre
- Temperatura media anual: ~21°C

Autor: Josue Emanuel Diaz Rodriguez
Materia: Inteligencia Artificial
Instituto Tecnologico de Chilpancingo
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# semilla para que los resultados sean reproducibles cada que corramos el script
np.random.seed(42)


def generar_temperatura(mes, dia_año, año):
    """
    Calcula la temperatura base del dia dependiendo del mes.
    Chilpancingo tiene un patron bien marcado: los meses de marzo-mayo
    son los mas calurosos antes de que lleguen las lluvias.
    """
    # temperatura base por mes (promedio historico aproximado)
    temps_mensuales = {
        1: 19.5,   # enero - fresco
        2: 20.8,   # febrero - empieza a calentar
        3: 23.4,   # marzo - calor
        4: 25.1,   # abril - mas calor, temporada seca
        5: 24.6,   # mayo - inicio lluvias
        6: 22.3,   # junio - lluvias constantes
        7: 21.8,   # julio - lluvias, mas fresco
        8: 21.9,   # agosto
        9: 21.5,   # septiembre - pico de lluvias
        10: 21.2,  # octubre - fin de lluvias
        11: 20.1,  # noviembre - enfriando
        12: 19.2,  # diciembre - mas frio
    }

    temp_base = temps_mensuales[mes]

    # variacion diaria normal (el dia es mas caliente que la noche)
    # esto simula el ciclo circadiano
    variacion_diaria = np.random.normal(0, 2.1)

    # tendencia de calentamiento muy ligera por año (realista)
    ajuste_año = (año - 2018) * 0.08

    temp_final = temp_base + variacion_diaria + ajuste_año
    return round(temp_final, 1)


def generar_precipitacion(mes, temp):
    """
    La precipitacion en Chilpancingo esta MUY marcada por temporadas.
    De junio a octubre es cuando llueve fuerte, el resto del año casi nada.
    Usamos distribucion exponencial para simular que la mayoria de dias
    llueve poco pero de vez en cuando hay tormentas.
    """
    # probabilidad de lluvia segun el mes
    prob_lluvia = {
        1: 0.05,   # casi no llueve
        2: 0.04,
        3: 0.06,
        4: 0.08,
        5: 0.20,   # empieza a llover
        6: 0.65,   # temporada de lluvias
        7: 0.70,
        8: 0.68,
        9: 0.75,   # mes con mas lluvia historicamente
        10: 0.45,
        11: 0.10,
        12: 0.05,
    }

    llueve_hoy = np.random.random() < prob_lluvia[mes]

    if not llueve_hoy:
        return 0.0

    # si llueve, cuanto? escala segun el mes
    intensidad_base = {
        1: 2.0, 2: 2.0, 3: 3.0, 4: 4.0,
        5: 8.0, 6: 15.0, 7: 18.0, 8: 17.0,
        9: 20.0, 10: 12.0, 11: 4.0, 12: 2.5
    }

    mm_lluvia = np.random.exponential(intensidad_base[mes])

    # a veces hay tormentas fuertes (evento extremo)
    if np.random.random() < 0.05:
        mm_lluvia *= np.random.uniform(2.5, 4.0)

    return round(min(mm_lluvia, 120.0), 1)  # tope realista


def generar_humedad(mes, precipitacion, temp):
    """
    La humedad relativa correlaciona con la lluvia y temperatura.
    Cuando llueve la humedad sube, con calor baja. Basico.
    """
    # humedad base por temporada
    if mes in [6, 7, 8, 9, 10]:
        humedad_base = 75.0
    elif mes in [11, 12, 1, 2]:
        humedad_base = 55.0
    else:
        humedad_base = 60.0

    # si llovio hoy, la humedad sube
    if precipitacion > 0:
        humedad_base += min(precipitacion * 0.8, 20)

    # temperatura alta = menos humedad relativa
    ajuste_temp = -(temp - 21) * 0.7

    humedad = humedad_base + ajuste_temp + np.random.normal(0, 4.0)
    return round(np.clip(humedad, 20.0, 98.0), 1)


def generar_viento(mes, precipitacion):
    """
    Velocidad del viento en km/h. En Chilpancingo no es muy ventoso
    en condiciones normales, pero con tormenta puede subir bastante.
    """
    velocidad_base = np.random.exponential(8.0)

    # con lluvia el viento suele aumentar
    if precipitacion > 10:
        velocidad_base += np.random.uniform(5, 20)
    elif precipitacion > 0:
        velocidad_base += np.random.uniform(2, 8)

    # enero y febrero son los meses con mas viento en la region
    if mes in [1, 2]:
        velocidad_base += np.random.uniform(2, 5)

    return round(np.clip(velocidad_base, 0.5, 80.0), 1)


def generar_presion(temp, altitud=1350):
    """
    Presion atmosferica en hPa. A 1350 msnm la presion base es menor
    que al nivel del mar. Formula barometrica simplificada.
    """
    # presion estandar a nivel del mar = 1013.25 hPa
    presion_base = 1013.25 * np.exp(-altitud / 8500)

    # temperatura afecta la presion (aire caliente = menos presion)
    ajuste_temp = -(temp - 21) * 0.5

    presion = presion_base + ajuste_temp + np.random.normal(0, 1.5)
    return round(presion, 1)


def generar_nubosidad(mes, precipitacion, humedad):
    """
    Cobertura de nubes en oktas (escala 0-8).
    0 = cielo despejado, 8 = totalmente nublado.
    """
    if precipitacion > 0:
        # si llueve, esta nublado si o si
        nubosidad = np.random.uniform(5, 8)
    elif mes in [6, 7, 8, 9]:
        # temporada de lluvias aunque no llueva hay nubes
        nubosidad = np.random.uniform(2, 6)
    else:
        # temporada seca, generalmente despejado
        nubosidad = np.random.uniform(0, 3)

    return round(np.clip(nubosidad, 0, 8), 0)


def crear_dataset_completo():
    """
    Función principal que genera el dataset completo de 2018 a 2024.
    Retorna un DataFrame con todas las variables climaticas diarias.
    """
    print("Generando dataset climatico para Chilpancingo, Gro...")
    print("Periodo: 01/01/2018 - 31/12/2024\n")

    fecha_inicio = datetime(2018, 1, 1)
    fecha_fin = datetime(2024, 12, 31)
    registros = []

    fecha_actual = fecha_inicio
    while fecha_actual <= fecha_fin:
        mes = fecha_actual.month
        año = fecha_actual.year
        dia_año = fecha_actual.timetuple().tm_yday

        # generar cada variable del dia
        temp = generar_temperatura(mes, dia_año, año)
        precip = generar_precipitacion(mes, temp)
        humedad = generar_humedad(mes, precip, temp)
        viento = generar_viento(mes, precip)
        presion = generar_presion(temp)
        nubes = generar_nubosidad(mes, precip, humedad)

        # temperatura maxima y minima del dia (diferencia tipica en la sierra)
        temp_max = round(temp + np.random.uniform(3.5, 7.0), 1)
        temp_min = round(temp - np.random.uniform(4.0, 8.0), 1)

        # clasificacion del dia segun precipitacion
        if precip == 0:
            tipo_dia = "Despejado"
        elif precip < 5:
            tipo_dia = "Llovizna"
        elif precip < 20:
            tipo_dia = "Lluvia_moderada"
        else:
            tipo_dia = "Lluvia_intensa"

        registros.append({
            "fecha": fecha_actual.strftime("%Y-%m-%d"),
            "año": año,
            "mes": mes,
            "dia": fecha_actual.day,
            "dia_semana": fecha_actual.weekday(),
            "dia_del_año": dia_año,
            "temp_media": temp,
            "temp_maxima": temp_max,
            "temp_minima": temp_min,
            "precipitacion_mm": precip,
            "humedad_relativa": humedad,
            "velocidad_viento": viento,
            "presion_hpa": presion,
            "nubosidad_oktas": nubes,
            "tipo_dia": tipo_dia,
        })

        fecha_actual += timedelta(days=1)

    df = pd.DataFrame(registros)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.set_index("fecha")

    return df


if __name__ == "__main__":
    df = crear_dataset_completo()

    ruta_salida = "datos/chilpancingo_clima_2018_2024.csv"
    df.to_csv(ruta_salida)

    print(f"Dataset guardado: {ruta_salida}")
    print(f"Total de registros: {len(df)}")
    print(f"Variables: {list(df.columns)}")
    print("\nPrimeros registros:")
    print(df.head(10).to_string())
    print("\nEstadisticas generales:")
    print(df.describe().round(2).to_string())