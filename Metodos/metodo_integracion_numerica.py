import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from Metodos.input_parser import (
    parse_real,
    parse_real_or_default,
    parse_int_or_default,
    build_numeric_function,
)


def evaluar_funcion_robusta(funcion_str, x_values, variable="x"):
    """Evalua una funcion numérica y reemplaza no-finitos por el límite simbólico cuando exista."""
    expr, f = build_numeric_function(funcion_str, variable)
    x_sym = sp.Symbol(variable)

    x_array = np.asarray(x_values, dtype=float)
    scalar_input = x_array.ndim == 0
    x_flat = x_array.reshape(-1)

    try:
        y_flat = np.array(f(x_flat), dtype=float).reshape(-1)
    except Exception:
        y_flat = np.array([float(f(float(xi))) for xi in x_flat], dtype=float)

    for idx, valor in enumerate(y_flat):
        if np.isfinite(valor):
            continue

        xi = float(x_flat[idx])
        limite_valor = np.nan

        try:
            limite = sp.limit(expr, x_sym, xi)
            limite_valor = float(sp.N(limite))
        except Exception:
            pass

        if not np.isfinite(limite_valor):
            try:
                limite_valor = float(f(float(xi)))
            except Exception:
                limite_valor = np.nan

        if not np.isfinite(limite_valor):
            raise ValueError(
                f"La funcion produjo valores no finitos en x={xi} y no se pudo obtener un limite finito"
            )

        y_flat[idx] = limite_valor

    if scalar_input:
        return float(y_flat[0])

    return y_flat.reshape(x_array.shape)


def _evaluar_malla(funcion_str, a, b, n):
    if n <= 0:
        raise ValueError("n debe ser un entero positivo")
    if b <= a:
        raise ValueError("Se requiere que b > a")

    x = np.linspace(a, b, n + 1)
    y = np.array(evaluar_funcion_robusta(funcion_str, x), dtype=float)

    h = (b - a) / n
    return x, y, h


def regla_trapecio(funcion_str, a, b, n):
    x, y, h = _evaluar_malla(funcion_str, a, b, n)
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return float(integral), x, y


def regla_rectangulo(funcion_str, a, b, n):
    if n <= 0:
        raise ValueError("n debe ser un entero positivo")
    if b <= a:
        raise ValueError("Se requiere que b > a")

    h = (b - a) / n
    x_mid = a + (np.arange(n) + 0.5) * h
    y_mid = np.array(evaluar_funcion_robusta(funcion_str, x_mid), dtype=float)

    integral = h * np.sum(y_mid)

    # Nodos de extremos para visualizar el intervalo en el dashboard.
    x_nodes = np.linspace(a, b, n + 1)
    y_nodes = np.array(evaluar_funcion_robusta(funcion_str, x_nodes), dtype=float)
    return float(integral), x_nodes, y_nodes


def regla_simpson_13(funcion_str, a, b, n):
    if n % 2 != 0:
        raise ValueError("Simpson 1/3 requiere n par")

    x, y, h = _evaluar_malla(funcion_str, a, b, n)
    integral = (h / 3.0) * (
        y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])
    )
    return float(integral), x, y


def regla_simpson_38(funcion_str, a, b, n):
    if n % 3 != 0:
        raise ValueError("Simpson 3/8 requiere n multiplo de 3")

    x, y, h = _evaluar_malla(funcion_str, a, b, n)
    pesos = np.ones_like(y)
    pesos[1:-1] = 3
    pesos[3:-1:3] = 2
    integral = (3.0 * h / 8.0) * np.sum(pesos * y)
    return float(integral), x, y


def regla_montecarlo(funcion_str, a, b, n, seed=None):
    if n <= 0:
        raise ValueError("n debe ser un entero positivo")
    if b <= a:
        raise ValueError("Se requiere que b > a")

    # Generar n puntos aleatorios uniformemente en [a, b]
    if seed is not None:
        np.random.seed(seed)
    x_random = np.random.uniform(a, b, n)
    y_random = np.array(evaluar_funcion_robusta(funcion_str, x_random), dtype=float)

    # Integral aproximada: (b-a) * promedio de f(x_i)
    integral = (b - a) * np.mean(y_random)

    # Desviación estándar de la estimación
    # Var(I) ≈ ((b-a)^2 / n) * Var(f(x_i))
    var_f = np.var(y_random, ddof=1)  # ddof=1 para muestra
    var_integral = ((b - a) ** 2 / n) * var_f
    std_integral = np.sqrt(var_integral)

    # Para visualización, ordenar los puntos
    sort_idx = np.argsort(x_random)
    x_nodes = x_random[sort_idx]
    y_nodes = y_random[sort_idx]

    return float(integral), float(std_integral), x_nodes, y_nodes


def graficar_area(funcion_str, x, y, titulo):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, y, linewidth=2, label=f"f(x) = {funcion_str}")
    ax.fill_between(x, y, 0, alpha=0.25, color="tab:blue", label="Area aproximada")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def ejecutar_integracion_numerica():
    print("\n" + "=" * 80)
    print("INTEGRACION NUMERICA")
    print("=" * 80)
    print("Metodos disponibles:")
    print("1. Rectangulo (punto medio)")
    print("2. Trapecio")
    print("3. Simpson 1/3")
    print("4. Simpson 3/8")
    print("5. Monte Carlo")

    opcion = input("Seleccione metodo (1-5): ").strip()
    if opcion not in {"1", "2", "3", "4"}:
        print("Opcion no valida")
        return

    funcion = input("Ingrese f(x): ").strip()
    if not funcion:
        print("Debe ingresar una funcion")
        return

    try:
        a = parse_real(input("Limite inferior a: "), "a")
        b = parse_real(input("Limite superior b: "), "b")
        n = parse_int_or_default(input("Cantidad de intervalos n (default 6): "), 6, "n")

        if opcion == "1":
            valor, x, y = regla_rectangulo(funcion, a, b, n)
            nombre = "Rectangulo (punto medio)"
        elif opcion == "2":
            valor, x, y = regla_trapecio(funcion, a, b, n)
            nombre = "Trapecio"
        elif opcion == "3":
            valor, x, y = regla_simpson_13(funcion, a, b, n)
            nombre = "Simpson 1/3"
        elif opcion == "4":
            valor, x, y = regla_simpson_38(funcion, a, b, n)
            nombre = "Simpson 3/8"
        elif opcion == "5":
            seed_input = input("Semilla (opcional, enter para aleatorio): ").strip()
            seed = int(seed_input) if seed_input else None
            valor, std, x, y = regla_montecarlo(funcion, a, b, n, seed=seed)
            nombre = "Monte Carlo"
        else:
            print("Opcion no valida")
            return

        print(f"\nResultado ({nombre}): {valor:.12g}")
        if nombre == "Monte Carlo":
            print(f"Desviacion estandar: {std:.7g}")

        mostrar_grafico = input("¿Desea ver el grafico del area? (s/n): ").strip().lower()
        if mostrar_grafico in {"s", "si", "sí"}:
            graficar_area(funcion, x, y, f"Integracion por {nombre}")

    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    ejecutar_integracion_numerica()
