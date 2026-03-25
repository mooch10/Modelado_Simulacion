import numpy as np
import matplotlib.pyplot as plt
from Metodos.input_parser import (
    parse_real,
    parse_real_or_default,
    parse_int_or_default,
    build_numeric_function,
)

def _evaluar_numero(valor):
    """Convierte resultados de lambdify a float real y valida finitud."""
    if isinstance(valor, complex):
        if abs(valor.imag) > 1e-12:
            raise ValueError("La función devolvió un valor complejo")
        valor = valor.real

    valor = float(valor)
    if not np.isfinite(valor):
        raise ValueError("La función devolvió un valor no finito")
    return valor


def evaluar_funcion(g_num, x):
    """
    Evalúa la función definida por el usuario en un valor dado de x.
    Soporta funciones matemáticas básicas como sin, cos, exp, etc.
    """
    try:
        return _evaluar_numero(g_num(x))
    except Exception as e:
        raise ValueError(f"Error al evaluar la función: {e}")


def metodo_punto_fijo(g_str, x0, tol=1e-6, max_iter=100):
    """
    Encuentra la raíz de f(x) = 0 usando el método de Punto Fijo.
    Requiere que la función g(x) converja, es decir, |g'(x)| < 1 en el intervalo.
    Retorna una tupla (raiz, tabla_datos).
    """
    print(f"Iniciando método de Punto Fijo con x0 = {x0}")
    _, g_num = build_numeric_function(g_str)
    x = x0
    tabla_datos = []

    for iteracion in range(max_iter):
        try:
            x_nuevo = evaluar_funcion(g_num, x)
        except ValueError as e:
            raise ValueError(f"Error en la evaluación de g(x): {e}")

        # Calcular el error absoluto
        error = abs(x_nuevo - x)
        tabla_datos.append((iteracion + 1, x, x_nuevo, error))

        print(f"Iteración {iteracion + 1}: x = {x_nuevo}")

        if error < tol:
            print(f"Convergido a la raíz: {x_nuevo} después de {iteracion + 1} iteraciones")
            imprimir_tabla(tabla_datos)
            return x_nuevo, tabla_datos

        x = x_nuevo

    print(f"Máximo de iteraciones alcanzado. Valor aproximado: {x}")
    imprimir_tabla(tabla_datos)
    return x, tabla_datos

def imprimir_tabla(datos):
    """
    Imprime una tabla con los datos de cada iteración del método de punto fijo.
    """
    print("\nTabla de Iteraciones:")
    print("=" * 70)
    print(f"{'Iter':^5} {'x_n':^20} {'x_(n+1)':^20} {'Error':^20}")
    print("=" * 70)
    for iteracion, x_n, x_n1, error in datos:
        print(f"{iteracion:^5} {x_n:^20.10f} {x_n1:^20.10f} {error:^20.2e}")
    print("=" * 70)

def graficar_funcion(func_str, a, b, raiz=None):
    """
    Grafica la función f(x) en el intervalo [a, b].
    Si se proporciona una raíz, la marca en el gráfico.
    """
    _, f_num = build_numeric_function(func_str)
    x_vals = np.linspace(a, b, 1000)
    y_vals = []

    for x in x_vals:
        try:
            y = evaluar_funcion(f_num, x)
            y_vals.append(y)
        except:
            y_vals.append(np.nan)  # Para evitar errores en la gráfica

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label=f'f(x) = {func_str}')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    if raiz is not None:
        plt.axvline(raiz, color='red', linestyle='--', label=f'Raíz aproximada: {raiz:.6f}')
        plt.plot(raiz, 0, 'ro')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gráfico de la función')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print("Búsqueda de Raíces usando Método de Punto Fijo")
    print("===============================================")

    # Obtener entradas del usuario
    func_str = input("Ingresa la función f(x) = 0 (ej. x**2 - 2): ")
    g_str = input("Ingresa la función g(x) para el punto fijo (ej. x - (x**2 - 2)/(2*x)): ")
    x0 = parse_real(input("Ingresa el valor inicial x0: "), "x0")
    tol_input = input("Ingresa la tolerancia (por defecto 1e-6): ")
    tol = parse_real_or_default(tol_input, 1e-6, "tolerancia")
    max_iter_input = input("Ingresa el máximo de iteraciones (por defecto 100): ")
    max_iter = parse_int_or_default(max_iter_input, 100, "máximo de iteraciones")

    try:
        raiz, tabla_datos = metodo_punto_fijo(g_str, x0, tol, max_iter)
        print(f"\nRaíz aproximada encontrada: {raiz}")

        # Preguntar si quiere graficar
        graficar = input("¿Quieres ver el gráfico de la función? (s/n): ").lower()
        if graficar == 's':
            a = parse_real(input("Ingresa el límite inferior para el gráfico: "), "a")
            b = parse_real(input("Ingresa el límite superior para el gráfico: "), "b")
            graficar_funcion(func_str, a, b, raiz)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
