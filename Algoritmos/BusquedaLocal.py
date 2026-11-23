import random

from .tsp_helpers import (
    two_opt,
    random_tour,
    nearest_neighbor
)


def busqueda_local(points, seed=2024):
    """
    Ejecuta Búsqueda Local 2-opt desde dos inicializaciones:
    1. Ruta aleatoria.
    2. Vecino más cercano desde un nodo aleatorio.

    Devuelve el mejor recorrido y su costo.
    """

    n = len(points)
    rng = random.Random(seed)

    # Primera inicialización: ruta aleatoria
    initial_random = random_tour(n, rng)
    tour_random, improvements_random, cost_random = two_opt(
        points, initial_random
    )

    # Segunda inicialización: vecino más cercano
    start = rng.randrange(n)
    initial_nn = nearest_neighbor(points, start)
    tour_nn, improvements_nn, cost_nn = two_opt(
        points, initial_nn
    )

    # Elegir el mejor resultado
    if cost_random <= cost_nn:
        best_tour = tour_random
        best_cost = cost_random
        best_method = "aleatoria"
    else:
        best_tour = tour_nn
        best_cost = cost_nn
        best_method = "vecino_mas_cercano"

    return {
        "tour": best_tour,
        "cost": best_cost,
        "method": best_method,
        "cost_random": cost_random,
        "cost_nn": cost_nn,
        "improvements_random": improvements_random,
        "improvements_nn": improvements_nn
    }