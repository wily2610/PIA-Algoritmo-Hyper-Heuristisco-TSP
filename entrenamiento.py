from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =====================================================
# 1. RUTAS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Datos"

TRAIN_FILE = DATA_DIR / "train_labeled.csv"
TEST_FILE = DATA_DIR / "test_labeled.csv"


# =====================================================
# 2. CARGAR DATOS
# =====================================================

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print("=== DATOS CARGADOS ===")
print("TRAIN:", train.shape)
print("TEST:", test.shape)


# =====================================================
# 3. DEFINIR CARACTERÍSTICAS
# =====================================================

features_numericas = [
    "n",
    "mean_pairwise_dist",
    "std_coords",
    "mean_centroid_dist",
    "mean_nn_dist"
]

features_categoricas = [
    "distribution"
]

features = features_numericas + features_categoricas

X_train = train[features]
X_test = test[features]

y_train = train["best_algorithm"]
y_test = test["best_algorithm"]


print("\n=== DISTRIBUCIÓN DE CLASES TRAIN ===")
print(y_train.value_counts())

print("\n=== DISTRIBUCIÓN DE CLASES TEST ===")
print(y_test.value_counts())


# =====================================================
# 4. PREPROCESAMIENTO
# =====================================================

preprocesamiento = ColumnTransformer(
    transformers=[
        (
            "numericas",
            StandardScaler(),
            features_numericas
        ),
        (
            "categoricas",
            OneHotEncoder(handle_unknown="ignore"),
            features_categoricas
        )
    ]
)


# =====================================================
# 5. MODELO: ÁRBOL DE DECISIÓN
# =====================================================

modelo = Pipeline(
    steps=[
        ("preprocesamiento", preprocesamiento),
        (
            "clasificador",
            DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=6,
                min_samples_leaf=5,
                random_state=42
            )
        )
    ]
)


# =====================================================
# 6. VALIDACIÓN CRUZADA
# =====================================================

print("\n=== VALIDACIÓN CRUZADA ===")

cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

metricas = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro"
}

resultados_cv = cross_validate(
    modelo,
    X_train,
    y_train,
    cv=cv,
    scoring=metricas
)

accuracy_cv = resultados_cv["test_accuracy"].mean()
balanced_cv = resultados_cv["test_balanced_accuracy"].mean()
f1_cv = resultados_cv["test_f1_macro"].mean()

print("Modelo: Árbol de Decisión")
print(
    f"Accuracy: {accuracy_cv:.4f} "
    f"± {resultados_cv['test_accuracy'].std():.4f}"
)
print(
    f"Balanced Accuracy: {balanced_cv:.4f} "
    f"± {resultados_cv['test_balanced_accuracy'].std():.4f}"
)
print(
    f"F1 Macro: {f1_cv:.4f} "
    f"± {resultados_cv['test_f1_macro'].std():.4f}"
)
print("F1 Macro por fold:", resultados_cv["test_f1_macro"].round(4).tolist())


# =====================================================
# 7. BASELINE
# =====================================================

clase_mayoritaria = y_train.value_counts().idxmax()

predicciones_baseline = np.full(
    len(y_test),
    clase_mayoritaria,
    dtype=object
)

accuracy_baseline = accuracy_score(
    y_test,
    predicciones_baseline
)

balanced_baseline = balanced_accuracy_score(
    y_test,
    predicciones_baseline
)

f1_baseline = f1_score(
    y_test,
    predicciones_baseline,
    average="macro",
    zero_division=0
)

print("\n=== BASELINE ===")
print("Clase mayoritaria:", clase_mayoritaria)
print(f"Accuracy: {accuracy_baseline:.4f}")
print(f"Balanced Accuracy: {balanced_baseline:.4f}")
print(f"F1 Macro: {f1_baseline:.4f}")


# =====================================================
# 8. ENTRENAMIENTO Y EVALUACIÓN FINAL
# =====================================================

print("\n=== ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN ===")

modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

accuracy = accuracy_score(y_test, predicciones)

balanced = balanced_accuracy_score(
    y_test,
    predicciones
)

f1 = f1_score(
    y_test,
    predicciones,
    average="macro",
    zero_division=0
)

print("\n=== RESULTADOS EN TEST ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced:.4f}")
print(f"F1 Macro: {f1:.4f}")

print("\n=== REPORTE DE CLASIFICACIÓN ===")
print(
    classification_report(
        y_test,
        predicciones,
        zero_division=0
    )
)


# =====================================================
# 9. COMPARACIÓN CONTRA BASELINE
# =====================================================

comparacion = pd.DataFrame([
    {
        "Modelo": "Decision Tree",
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced,
        "F1 Macro": f1
    },
    {
        "Modelo": "Baseline",
        "Accuracy": accuracy_baseline,
        "Balanced Accuracy": balanced_baseline,
        "F1 Macro": f1_baseline
    }
])

comparacion.to_csv(
    DATA_DIR / "comparacion_modelos.csv",
    index=False
)


# =====================================================
# 10. MATRIZ DE CONFUSIÓN
# =====================================================

clases = sorted(y_train.unique().tolist())

matriz = confusion_matrix(
    y_test,
    predicciones,
    labels=clases
)

matriz_df = pd.DataFrame(
    matriz,
    index=clases,
    columns=clases
)

matriz_df.index.name = "Real"
matriz_df.columns.name = "Predicho"

matriz_df.to_csv(
    DATA_DIR / "matriz_confusion.csv"
)

print("\n=== MATRIZ DE CONFUSIÓN ===")
print(matriz_df)


# =====================================================
# 11. ANÁLISIS DE REGRET
# =====================================================

columnas_costos = {
    "2opt_hill_climbing": "cost_2opt_hill_climbing",
    "sa_2opt": "cost_sa_2opt",
    "mst_2opt": "cost_mst_2opt"
}

# Renombrar columnas de costo -> nombres de algoritmo
costos = test[list(columnas_costos.values())].rename(
    columns={v: k for k, v in columnas_costos.items()}
)

costo_optimo = costos.min(axis=1)

# Selección vectorizada del costo del algoritmo predicho
costo_seleccionado = pd.Series(
    costos.to_numpy()[
        np.arange(len(test)),
        costos.columns.get_indexer(predicciones)
    ],
    index=test.index
)

costo_baseline = costos[clase_mayoritaria]

regret = (costo_seleccionado - costo_optimo).clip(lower=0)

regret_baseline = (
    costo_baseline - costo_optimo
).clip(lower=0)

# Cuidado con costos óptimos = 0 (división por cero)
costo_optimo_safe = costo_optimo.replace(0, np.nan)

regret_relativo = (
    regret / costo_optimo_safe
) * 100

regret_baseline_relativo = (
    regret_baseline / costo_optimo_safe
) * 100

resultados_regret = pd.DataFrame({
    "instance_id": test["instance_id"],
    "algoritmo_predicho": predicciones,
    "algoritmo_real": y_test.values,
    "costo_optimo": costo_optimo.values,
    "costo_seleccionado": costo_seleccionado.values,
    "regret": regret.values,
    "regret_relativo_porcentaje": regret_relativo.values,
    "costo_baseline": costo_baseline.values,
    "regret_baseline": regret_baseline.values,
    "regret_baseline_relativo_porcentaje":
        regret_baseline_relativo.values
})

resultados_regret.to_csv(
    DATA_DIR / "resultados_regret.csv",
    index=False
)

n_cero = int((costo_optimo == 0).sum())

print("\n=== ANÁLISIS DE REGRET ===")
print(f"Regret medio del árbol: {regret.mean():.6f}")
print(f"Regret mediano: {regret.median():.6f}")
print(
    f"Regret relativo medio: "
    f"{regret_relativo.mean():.4f}%"
)
print(f"Regret medio baseline: {regret_baseline.mean():.6f}")
print(
    f"Regret relativo medio baseline: "
    f"{regret_baseline_relativo.mean():.4f}%"
)
print(
    "Instancias con regret cero:",
    (regret == 0).sum(),
    "de",
    len(test)
)
if n_cero:
    print(
        f"⚠️ {n_cero} instancias con costo óptimo = 0 "
        f"(excluidas del regret relativo)"
    )


# =====================================================
# 12. IMPORTANCIA DE CARACTERÍSTICAS
# =====================================================

clasificador = modelo.named_steps["clasificador"]

nombres_caracteristicas = (
    modelo.named_steps[
        "preprocesamiento"
    ].get_feature_names_out()
)

importancias = clasificador.feature_importances_

importancia_df = pd.DataFrame({
    "Caracteristica": nombres_caracteristicas,
    "Importancia": importancias
})

importancia_df["Caracteristica"] = (
    importancia_df["Caracteristica"]
    .str.replace("numericas__", "", regex=False)
    .str.replace("categoricas__", "", regex=False)
)

importancia_df = importancia_df.sort_values(
    by="Importancia",
    ascending=False
)

importancia_df.to_csv(
    DATA_DIR / "importancia_caracteristicas.csv",
    index=False
)

print("\n=== IMPORTANCIA DE CARACTERÍSTICAS ===")
print(importancia_df.to_string(index=False))


# =====================================================
# 13. ANÁLISIS DE PATRONES
# =====================================================

analisis_patrones = (
    train.groupby("best_algorithm")[features_numericas]
    .mean()
)

analisis_patrones.to_csv(
    DATA_DIR / "analisis_patrones.csv"
)


# Por distribución

patrones_distribucion = pd.crosstab(
    train["distribution"],
    train["best_algorithm"]
)

patrones_distribucion.to_csv(
    DATA_DIR / "patrones_distribucion.csv"
)


# Por tamaño de instancia (bins para evitar demasiadas filas)

train["n_bin"] = pd.qcut(
    train["n"],
    q=min(5, train["n"].nunique()),
    duplicates="drop"
)

patrones_tamano = pd.crosstab(
    train["n_bin"],
    train["best_algorithm"]
)

patrones_tamano.to_csv(
    DATA_DIR / "patrones_tamano.csv"
)


# Por tamaño y distribución

patrones_completos = pd.crosstab(
    [train["distribution"], train["n_bin"]],
    train["best_algorithm"]
)

patrones_completos.to_csv(
    DATA_DIR / "patrones_tamano_distribucion.csv"
)


# =====================================================
# 14. GUARDAR MODELO
# =====================================================

joblib.dump(
    modelo,
    DATA_DIR / "modelo_arbol.joblib"
)


# =====================================================
# 15. FINALIZACIÓN
# =====================================================

print("\n=== ARCHIVOS GENERADOS ===")

archivos = [
    "comparacion_modelos.csv",
    "matriz_confusion.csv",
    "resultados_regret.csv",
    "importancia_caracteristicas.csv",
    "analisis_patrones.csv",
    "patrones_distribucion.csv",
    "patrones_tamano.csv",
    "patrones_tamano_distribucion.csv",
    "modelo_arbol.joblib"
]

for archivo in archivos:
    print(archivo)

print("\n¡ENTRENAMIENTO TERMINADO!")