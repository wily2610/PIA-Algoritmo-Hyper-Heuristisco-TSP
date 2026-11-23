import math
import random
import time

from .tsp_helpers import (
    tour_length,
    two_opt_move,
    random_tour
)


def enfriamiento_logaritmico(iteracion, temperatura_inicial):
    """Calcula la temperatura mediante enfriamiento logarítmico."""
    return temperatura_inicial / math.log(1 + iteracion + 1e-9)


def simulated_annealing(
    points,
    T0=100,
    max_iter=20_000,
    seed=11
):
    """
    Recocido Simulado con movimientos 2-opt
    y enfriamiento logarítmico.
    """

    start_time = time.perf_counter()
    rng = random.Random(seed)

    n = len(points)

    # Solución inicial aleatoria
    current_tour = random_tour(n, rng)
    current_cost = tour_length(points, current_tour)

    best_tour = current_tour[:]
    best_cost = current_cost

    history = [best_cost]

    for iteration in range(max_iter):

        # Temperatura actual
        temperature = enfriamiento_logaritmico(
            iteration, T0
        )

        # Elegir movimiento 2-opt
        i = rng.randrange(0, n - 2)
        k = rng.randrange(i + 2, n)

        # Evitar invertir la ruta completa
        if i == 0 and k == n - 1:
            history.append(best_cost)
            continue

        candidate_tour = two_opt_move(
            current_tour, i, k
        )

        candidate_cost = tour_length(
            points, candidate_tour
        )

        delta = candidate_cost - current_cost

        # Aceptar mejoras o movimientos probabilísticos
        if delta < 0 or rng.random() < math.exp(
            -delta / temperature
        ):
            current_tour = candidate_tour
            current_cost = candidate_cost

        # Guardar la mejor solución encontrada
        if current_cost < best_cost:
            best_tour = current_tour[:]
            best_cost = current_cost

        history.append(best_cost)

    elapsed = time.perf_counter() - start_time

    return {
        "tour": best_tour,
        "cost": best_cost,
        "history": history,
        "elapsed": elapsed
    }


def recorrido_simulado(points, seed=11):
    """Interfaz para utilizar SA desde el generador de etiquetas."""

    return simulated_annealing(
        points,
        T0=100,
        max_iter=20_000,
        seed=seed
    )