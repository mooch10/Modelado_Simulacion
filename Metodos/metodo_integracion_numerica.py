import numpy as np
import matplotlib.pyplot as plt

from Metodos.input_parser import (
    parse_real,
    parse_real_or_default,
    parse_int_or_default,
    build_numeric_function,
)


def _evaluar_malla(funcion_str, a, b, n):
    if n <= 0:
        raise ValueError("n debe ser un entero positivo")
    if b <= a:
        raise ValueError("Se requiere que b > a")

    _, f = build_numeric_function(funcion_str)
    x = np.linspace(a, b, n + 1)
    y = np.array(f(x), dtype=float)

    if not np.all(np.isfinite(y)):
        raise ValueError("La funcion produjo valores no finitos en el intervalo")

    h = (b - a) / n
    return x, y, h


def regla_trapecio(funcion_str, a, b, n):
    x, y, h = _evaluar_malla(funcion_str, a, b, n)
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return float(integral), x, y


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
    print("1. Trapecio")
    print("2. Simpson 1/3")
    print("3. Simpson 3/8")

    opcion = input("Seleccione metodo (1-3): ").strip()
    if opcion not in {"1", "2", "3"}:
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
            valor, x, y = regla_trapecio(funcion, a, b, n)
            nombre = "Trapecio"
        elif opcion == "2":
            valor, x, y = regla_simpson_13(funcion, a, b, n)
            nombre = "Simpson 1/3"
        else:
            valor, x, y = regla_simpson_38(funcion, a, b, n)
            nombre = "Simpson 3/8"

        print(f"\nResultado ({nombre}): {valor:.12g}")

        mostrar_grafico = input("¿Desea ver el grafico del area? (s/n): ").strip().lower()
        if mostrar_grafico in {"s", "si", "sí"}:
            graficar_area(funcion, x, y, f"Integracion por {nombre}")

    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    ejecutar_integracion_numerica()
