import math
import random
import itertools


def tour_length(points, tour):
    """Calcula la distancia total de un recorrido TSP."""
    return sum(
        math.dist(
            points[tour[i]],
            points[tour[(i + 1) % len(tour)]]
        )
        for i in range(len(tour))
    )


def two_opt_move(route, i, k):
    """Invierte un segmento de la ruta."""
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]


def two_opt(points, tour, max_iter=200_000):
    """Mejora una ruta mediante búsqueda local 2-opt."""
    n = len(tour)
    tour = tour[:]

    best_len = tour_length(points, tour)
    improvements = 0
    it = 0
    improved = True

    while improved and it < max_iter:
        improved = False

        for i in range(1, n - 1):
            for k in range(i + 1, n):

                if k - i < 2:
                    continue

                candidate = two_opt_move(tour, i, k)
                candidate_len = tour_length(points, candidate)
                it += 1

                if candidate_len < best_len - 1e-9:
                    tour = candidate
                    best_len = candidate_len
                    improvements += 1
                    improved = True
                    break

            if improved:
                break

    return tour, improvements, best_len


def nearest_neighbor(points, start=0):
    """Construye una ruta usando vecino más cercano."""
    n = len(points)
    unvisited = set(range(n)) - {start}
    tour = [start]

    while unvisited:
        last = tour[-1]

        nxt = min(
            unvisited,
            key=lambda city: math.dist(points[last], points[city])
        )

        tour.append(nxt)
        unvisited.remove(nxt)

    return tour


def random_tour(n, rng=None):
    """Genera una ruta aleatoria con todas las ciudades."""
    tour = list(range(n))

    if rng is None:
        rng = random

    rng.shuffle(tour)
    return tour


def random_cities(n, seed=42):
    """Genera ciudades aleatorias para experimentos."""
    rng = random.Random(seed)

    return [
        (rng.uniform(0, 100), rng.uniform(0, 100))
        for _ in range(n)
    ]


def brute_force_optimal(points):
    """Calcula la solución óptima por fuerza bruta."""
    n = len(points)
    best = None
    best_len = math.inf

    for perm in itertools.permutations(range(1, n)):
        tour = (0,) + perm
        length = tour_length(points, tour)

        if length < best_len:
            best = tour
            best_len = length

    return list(best), best_len