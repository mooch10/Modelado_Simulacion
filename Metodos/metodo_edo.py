import numpy as np
import sympy as sp

from Metodos.input_parser import parse_real, parse_real_or_default, parse_int_or_default

"""
╔════════════════════════════════════════════════════════════════╗
║       MACHETE: ECUACIONES DIFERENCIALES ORDINARIAS              ║
║  (Euler, Runge-Kutta 4)                                        ║
╚════════════════════════════════════════════════════════════════╝

DEFINICIÓN:
Resuelven dy/dx = f(x,y) numéricamente a partir de condición inicial.

UTILIDAD:
• Resolver EDO sin solución analítica.
• Modelar sistemas dinámicos (física, biología, economía).

PASOS (Euler):
1. Condición inicial: (x₀, y₀)
2. Para cada paso i: y_{i+1} = y_i + h·f(x_i, y_i)
3. Completar n iteraciones

PASOS (RK4 - Runge-Kutta 4to orden):
1. Condición inicial: (x₀, y₀)
2. Calcular k₁,k₂,k₃,k₄ usando pendientes en puntos estratégicos
3. y_{i+1} = y_i + (h/6)·(k₁ + 2k₂ + 2k₃ + k₄)
4. Completar n iteraciones

FÓRMULAS:
Euler: y_{i+1} = y_i + h·f(x_i, y_i)
RK4: k₁ = f(x_i, y_i)
     k₂ = f(x_i + h/2, y_i + h·k₁/2)
     k₃ = f(x_i + h/2, y_i + h·k₂/2)
     k₄ = f(x_i + h, y_i + h·k₃)
     y_{i+1} = y_i + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)

REQUISITOS:
• f(x,y) calculable
• h > 0 (tamaño paso)
• n > 0 (iteraciones)
• RK4 es más preciso que Euler
"""

ALLOWED_LOCALS = {
    "e": sp.E,
    "pi": sp.pi,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
}


def construir_funcion_edo(funcion_str):
    x, y = sp.symbols("x y")
    expr = sp.sympify(funcion_str, locals=ALLOWED_LOCALS)
    invalid = expr.free_symbols - {x, y}
    if invalid:
        raise ValueError(f"Variables no permitidas: {invalid}")
    f = sp.lambdify((x, y), expr, "numpy")
    return expr, f


def metodo_euler(funcion_str, x0, y0, h, n):
    if n <= 0:
        raise ValueError("n debe ser positivo")
    if h == 0:
        raise ValueError("h no puede ser 0")

    _, f = construir_funcion_edo(funcion_str)
    filas = []

    x = float(x0)
    y = float(y0)
    filas.append({"Iteracion": 0, "x": x, "y": y, "k1": float(f(x, y))})

    for i in range(1, n + 1):
        k1 = float(f(x, y))
        y = y + h * k1
        x = x + h
        filas.append({"Iteracion": i, "x": float(x), "y": float(y), "k1": k1})

    return filas


def metodo_rk4(funcion_str, x0, y0, h, n):
    if n <= 0:
        raise ValueError("n debe ser positivo")
    if h == 0:
        raise ValueError("h no puede ser 0")

    _, f = construir_funcion_edo(funcion_str)
    filas = []

    x = float(x0)
    y = float(y0)
    filas.append(
        {
            "Iteracion": 0,
            "x": x,
            "y": y,
            "k1": float("nan"),
            "k2": float("nan"),
            "k3": float("nan"),
            "k4": float("nan"),
        }
    )

    for i in range(1, n + 1):
        k1 = float(f(x, y))
        k2 = float(f(x + h / 2.0, y + h * k1 / 2.0))
        k3 = float(f(x + h / 2.0, y + h * k2 / 2.0))
        k4 = float(f(x + h, y + h * k3))

        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x = x + h

        filas.append(
            {
                "Iteracion": i,
                "x": float(x),
                "y": float(y),
                "k1": k1,
                "k2": k2,
                "k3": k3,
                "k4": k4,
            }
        )

    return filas


def ejecutar_edo():
    print("\n" + "=" * 80)
    print("EDO - PROBLEMA DE VALOR INICIAL")
    print("=" * 80)
    print("1. Euler")
    print("2. Runge-Kutta 4 (RK4)")

    opcion = input("Seleccione metodo (1-2): ").strip()
    if opcion not in {"1", "2"}:
        print("Opcion no valida")
        return

    try:
        funcion = input("Ingrese y' = f(x, y): ").strip()
        if not funcion:
            raise ValueError("Debe ingresar una funcion")

        x0 = parse_real(input("x0: "), "x0")
        y0 = parse_real(input("y0: "), "y0")
        h = parse_real_or_default(input("Paso h (default 0.1): "), 0.1, "h")
        n = parse_int_or_default(input("Cantidad de pasos n (default 10): "), 10, "n")

        if opcion == "1":
            filas = metodo_euler(funcion, x0, y0, h, n)
            nombre = "Euler"
        else:
            filas = metodo_rk4(funcion, x0, y0, h, n)
            nombre = "RK4"

        ultimo = filas[-1]
        print(f"\nResultado ({nombre}): y({ultimo['x']:.7g}) = {ultimo['y']:.12g}")
        print(f"Pasos realizados: {len(filas) - 1}")

    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    ejecutar_edo()
