import math

from .tsp_helpers import two_opt, tour_length


def mst_prim(points):
    """
    Construye un árbol de expansión mínima utilizando Prim.
    Devuelve la lista de hijos de cada nodo.
    """

    n = len(points)

    if n == 0:
        return []

    visited = [False] * n
    children = [[] for _ in range(n)]

    visited[0] = True

    for _ in range(n - 1):
        min_distance = math.inf
        best_parent = -1
        best_node = -1

        for u in range(n):
            if visited[u]:
                for v in range(n):
                    if not visited[v]:
                        distance = math.dist(points[u], points[v])

                        if distance < min_distance:
                            min_distance = distance
                            best_parent = u
                            best_node = v

        children[best_parent].append(best_node)
        visited[best_node] = True

    return children


def preorder(children, root=0):
    """Genera un recorrido del árbol MST mediante preorden."""

    tour = []

    def visit(node):
        tour.append(node)

        for child in children[node]:
            visit(child)

    visit(root)

    return tour


def mst_2opt(points):
    """
    Ejecuta MST + recorrido preorden + 2-opt.
    Devuelve el recorrido mejorado y su costo.
    """

    # Construir el árbol MST
    children = mst_prim(points)

    # Obtener recorrido inicial
    initial_tour = preorder(children)

    # Calcular costo del MST convertido en recorrido
    initial_cost = tour_length(points, initial_tour)

    # Mejorar recorrido mediante 2-opt
    best_tour, improvements, best_cost = two_opt(
        points,
        initial_tour
    )

    return {
        "tour": best_tour,
        "cost": best_cost,
        "initial_cost": initial_cost,
        "improvements": improvements
    }