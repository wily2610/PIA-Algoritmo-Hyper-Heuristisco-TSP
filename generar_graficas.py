import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# ==========================================
# 1. RUTAS Y CARGA DE DATOS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Datos"
GRAFICAS_DIR = BASE_DIR / "Graficas"

GRAFICAS_DIR.mkdir(exist_ok=True)

train = pd.read_csv(DATA_DIR / "train_labeled.csv")
test = pd.read_csv(DATA_DIR / "test_labeled.csv")

datos = pd.concat([train, test], ignore_index=True)

comparacion = pd.read_csv(
    DATA_DIR / "comparacion_modelos.csv"
)

matriz = pd.read_csv(
    DATA_DIR / "matriz_confusion.csv",
    index_col=0
)

importancias = pd.read_csv(
    DATA_DIR / "importancia_caracteristicas.csv"
)

patrones_distribucion = pd.read_csv(
    DATA_DIR / "patrones_distribucion.csv",
    index_col=0
)

patrones_tamano = pd.read_csv(
    DATA_DIR / "patrones_tamano.csv",
    index_col=0
)

patrones_completos = pd.read_csv(
    DATA_DIR / "patrones_tamano_distribucion.csv",
    index_col=[0, 1]
)

sns.set_theme(style="whitegrid")

print("Generando graficas...")


# ==========================================
# 2. BALANCE DE CLASES
# ==========================================

plt.figure(figsize=(9, 5))

orden = datos["best_algorithm"].value_counts()

plt.bar(
    orden.index,
    orden.values
)

plt.title("Balance de clases - 500 instancias")
plt.xlabel("Algoritmo ganador")
plt.ylabel("Numero de instancias")
plt.xticks(rotation=15)

for i, valor in enumerate(orden.values):
    plt.text(i, valor + 3, str(valor), ha="center")

plt.tight_layout()
plt.savefig(GRAFICAS_DIR / "balance_clases.png", dpi=300)
plt.close()


# ==========================================
# 3. COMPARACION DE MODELOS
# ==========================================

plt.figure(figsize=(10, 6))

comparacion_grafica = comparacion.melt(
    id_vars="Modelo",
    value_vars=["Accuracy", "Balanced Accuracy", "F1 Macro"],
    var_name="Metrica",
    value_name="Valor"
)

sns.barplot(
    data=comparacion_grafica,
    x="Modelo",
    y="Valor",
    hue="Metrica"
)

plt.title("Comparacion de modelos - Validacion cruzada")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    GRAFICAS_DIR / "comparacion_modelos.png",
    dpi=300
)
plt.close()


# ==========================================
# 4. MATRIZ DE CONFUSION
# ==========================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Matriz de confusion - Conjunto de prueba")
plt.xlabel("Algoritmo predicho")
plt.ylabel("Algoritmo real")

plt.tight_layout()

plt.savefig(
    GRAFICAS_DIR / "matriz_confusion.png",
    dpi=300
)
plt.close()


# ==========================================
# 5. IMPORTANCIA DE CARACTERISTICAS
# ==========================================

importancias = importancias.sort_values(
    by="Importancia",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    importancias["Caracteristica"],
    importancias["Importancia"]
)

plt.title("Importancia de caracteristicas")
plt.xlabel("Importancia")
plt.ylabel("Caracteristica")

plt.tight_layout()

plt.savefig(
    GRAFICAS_DIR / "importancia_caracteristicas.png",
    dpi=300
)
plt.close()


# ==========================================
# 6. GANADORES POR DISTRIBUCION
# ==========================================

plt.figure(figsize=(10, 6))

patrones_distribucion.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Algoritmo ganador por distribucion")
plt.xlabel("Distribucion")
plt.ylabel("Numero de instancias")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    GRAFICAS_DIR / "patrones_distribucion.png",
    dpi=300
)
plt.close()


# ==========================================
# 7. GANADORES POR TAMANO
# ==========================================

plt.figure(figsize=(10, 6))

patrones_tamano.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Algoritmo ganador por numero de ciudades")
plt.xlabel("Numero de ciudades")
plt.ylabel("Numero de instancias")

plt.tight_layout()

plt.savefig(
    GRAFICAS_DIR / "patrones_tamano.png",
    dpi=300
)
plt.close()


# ==========================================
# 8. GANADORES POR TAMANO Y DISTRIBUCION
# ==========================================

plt.figure(figsize=(12, 7))

sns.heatmap(
    patrones_completos,
    annot=True,
    fmt="d",
    cmap="YlGnBu"
)

plt.title("Algoritmo ganador por tamano y distribucion")
plt.xlabel("Algoritmo ganador")
plt.ylabel("Distribucion y numero de ciudades")

plt.tight_layout()

plt.savefig(
    GRAFICAS_DIR / "patrones_completos.png",
    dpi=300
)
plt.close()


# ==========================================
# 9. FINAL
# ==========================================

print("\nGRAFICAS GENERADAS CORRECTAMENTE")

for archivo in GRAFICAS_DIR.iterdir():
    print(archivo.name)

print("\nProceso completado.")