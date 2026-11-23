from pathlib import Path
import pandas as pd

from Algoritmos.BusquedaLocal import busqueda_local
from Algoritmos.RecorridoSimulado import recorrido_simulado
from Algoritmos.MST import mst_2opt


# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Datos"
INSTANCES_DIR = DATA_DIR / "instances"


def cargar_instancia(instance_id):
    """Carga las coordenadas de una instancia TSP."""

    archivo = INSTANCES_DIR / f"{instance_id}.csv"

    if not archivo.exists():
        raise FileNotFoundError(
            f"No se encontró la instancia: {archivo}"
        )

    df = pd.read_csv(archivo)

    points = list(zip(df["x"], df["y"]))

    return points


def etiquetar_dataset(nombre_archivo):
    """Ejecuta los tres algoritmos y genera las etiquetas."""

    archivo = DATA_DIR / nombre_archivo

    df = pd.read_csv(
        archivo,
        dtype={"instance_id": str}
    )

    costos_2opt = []
    costos_sa = []
    costos_mst = []
    mejores_algoritmos = []

    total = len(df)

    for index, row in df.iterrows():

        instance_id = row["instance_id"]

        print(
            f"Procesando {index + 1}/{total}: "
            f"{instance_id}"
        )

        points = cargar_instancia(instance_id)

        # Algoritmo 1: Búsqueda Local 2-opt
        resultado_2opt = busqueda_local(
            points,
            seed=2024
        )

        # Algoritmo 2: Recocido Simulado + 2-opt
        resultado_sa = recorrido_simulado(
            points,
            seed=11
        )

        # Algoritmo 3: MST + 2-opt
        resultado_mst = mst_2opt(points)

        # Guardar los costos
        cost_2opt = resultado_2opt["cost"]
        cost_sa = resultado_sa["cost"]
        cost_mst = resultado_mst["cost"]

        costos_2opt.append(cost_2opt)
        costos_sa.append(cost_sa)
        costos_mst.append(cost_mst)

        # Determinar el algoritmo ganador
        costos = {
            "2opt_hill_climbing": cost_2opt,
            "sa_2opt": cost_sa,
            "mst_2opt": cost_mst
        }

        mejor = min(costos, key=costos.get)

        mejores_algoritmos.append(mejor)

    # Agregar columnas al DataFrame
    df["cost_2opt_hill_climbing"] = costos_2opt
    df["cost_sa_2opt"] = costos_sa
    df["cost_mst_2opt"] = costos_mst
    df["best_algorithm"] = mejores_algoritmos

    # Guardar resultado sin modificar el archivo original
    nombre_salida = archivo.stem + "_labeled.csv"
    ruta_salida = DATA_DIR / nombre_salida

    df.to_csv(ruta_salida, index=False)

    print(f"\nDataset terminado: {ruta_salida}")
    print("\nDistribución de etiquetas:")
    print(df["best_algorithm"].value_counts())

    return df


if __name__ == "__main__":

    print("=== ETIQUETANDO TRAIN ===")
    train = etiquetar_dataset("train.csv")

    print("\n=== ETIQUETANDO TEST ===")
    test = etiquetar_dataset("test.csv")

    print("\n¡Proceso terminado!")