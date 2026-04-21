import numpy as np
import sympy as sp

from Metodos.input_parser import parse_real, parse_real_or_default, parse_int_or_default

"""
╔════════════════════════════════════════════════════════════════╗
║       MACHETE: ECUACIONES DIFERENCIALES ORDINARIAS              ║
║  (Euler, Heun, Runge-Kutta 4)                                  ║
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

PASOS (Heun / RK2 mejorado):
1. Condición inicial: (x₀, y₀)
2. Predictor de Euler: y* = y_i + h·f(x_i, y_i)
3. Corrector (promedio de pendientes):
    y_{i+1} = y_i + (h/2)[f(x_i, y_i) + f(x_i+h, y*)]
4. Completar n iteraciones

PASOS (RK4 - Runge-Kutta 4to orden):
1. Condición inicial: (x₀, y₀)
2. Calcular k₁,k₂,k₃,k₄ usando pendientes en puntos estratégicos
3. y_{i+1} = y_i + (h/6)·(k₁ + 2k₂ + 2k₃ + k₄)
4. Completar n iteraciones

FÓRMULAS:
Euler: y_{i+1} = y_i + h·f(x_i, y_i)
Heun:  k₁ = f(x_i, y_i)
    y* = y_i + h·k₁
    k₂ = f(x_i+h, y*)
    y_{i+1} = y_i + (h/2)(k₁ + k₂)
RK4: k₁ = f(x_i, y_i)
     k₂ = f(x_i + h/2, y_i + h·k₁/2)
     k₃ = f(x_i + h/2, y_i + h·k₂/2)
     k₄ = f(x_i + h, y_i + h·k₃)
     y_{i+1} = y_i + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)

REQUISITOS:
• f(x,y) calculable
• h > 0 (tamaño paso)
• n > 0 (iteraciones)
• Heun mejora a Euler usando predictor-corrector
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
    "cbrt": lambda x: sp.real_root(x, 3),
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


def _validar_parametros_edo(h, n):
    if n <= 0:
        raise ValueError("n debe ser positivo")
    if h == 0:
        raise ValueError("h no puede ser 0")


def metodo_euler(funcion_str, x0, y0, h, n):
    _validar_parametros_edo(h, n)

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


def metodo_heun(funcion_str, x0, y0, h, n):
    _validar_parametros_edo(h, n)

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
            "y_predictor": float("nan"),
            "k2": float("nan"),
        }
    )

    for i in range(1, n + 1):
        k1 = float(f(x, y))
        y_predictor = y + h * k1
        k2 = float(f(x + h, y_predictor))

        y = y + (h / 2.0) * (k1 + k2)
        x = x + h

        filas.append(
            {
                "Iteracion": i,
                "x": float(x),
                "y": float(y),
                "k1": k1,
                "y_predictor": float(y_predictor),
                "k2": k2,
            }
        )

    return filas


def metodo_rk4(funcion_str, x0, y0, h, n):
    _validar_parametros_edo(h, n)

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
    print("2. Heun (RK2 mejorado)")
    print("3. Runge-Kutta 4 (RK4)")
    print("4. Comparar Euler vs Heun vs RK4")

    opcion = input("Seleccione metodo (1-4): ").strip()
    if opcion not in {"1", "2", "3", "4"}:
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
        elif opcion == "2":
            filas = metodo_heun(funcion, x0, y0, h, n)
            nombre = "Heun"
        elif opcion == "3":
            filas = metodo_rk4(funcion, x0, y0, h, n)
            nombre = "RK4"
        else:
            filas_euler = metodo_euler(funcion, x0, y0, h, n)
            filas_heun = metodo_heun(funcion, x0, y0, h, n)
            filas_rk4 = metodo_rk4(funcion, x0, y0, h, n)

            y_euler = float(filas_euler[-1]["y"])
            y_heun = float(filas_heun[-1]["y"])
            y_rk4 = float(filas_rk4[-1]["y"])
            x_fin = float(filas_rk4[-1]["x"])

            print("\n" + "=" * 90)
            print("COMPARATIVA EDO: EULER vs HEUN vs RK4".center(90))
            print("=" * 90)
            print(f"f(x,y) = {funcion}")
            print(f"x0 = {x0}, y0 = {y0}, h = {h}, n = {n}")
            print("-" * 90)
            print(f"{'Método':<20}{'x final':>15}{'y final':>25}")
            print("-" * 90)
            print(f"{'Euler':<20}{x_fin:>15.6g}{y_euler:>25.12g}")
            print(f"{'Heun':<20}{x_fin:>15.6g}{y_heun:>25.12g}")
            print(f"{'RK4':<20}{x_fin:>15.6g}{y_rk4:>25.12g}")
            print("-" * 90)
            print(f"|Heun - Euler| = {abs(y_heun - y_euler):.12g}")
            print(f"|RK4  - Euler| = {abs(y_rk4 - y_euler):.12g}")
            print(f"|RK4  - Heun | = {abs(y_rk4 - y_heun):.12g}")
            print("=" * 90)
            return

        ultimo = filas[-1]
        print(f"\nResultado ({nombre}): y({ultimo['x']:.6g}) = {ultimo['y']:.12g}")
        print(f"Pasos realizados: {len(filas) - 1}")

    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    ejecutar_edo()

