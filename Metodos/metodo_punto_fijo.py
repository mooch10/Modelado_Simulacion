import math
import re
import numpy as np
import matplotlib.pyplot as plt
from Metodos.input_parser import parse_real, parse_real_or_default, parse_int_or_default

def evaluar_funcion(func_str, x):
    """
    Evalúa la función definida por el usuario en un valor dado de x.
    Soporta funciones matemáticas básicas como sin, cos, exp, etc.
    """
    # corrija casos típicos donde se olvida el operador '*' entre un número y
    # la constante e, por ejemplo "0.4e**x**2" -> "0.4*e**x**2".
    func_str = re.sub(r'(?P<num>\d)(?P<e>e\*\*)', r'\g<num>*\g<e>', func_str)

    try:
        # Define nombres permitidos por seguridad
        nombres_permitidos = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "exp": math.exp,
            "log": math.log,
            "sqrt": math.sqrt,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "pow": pow
        }
        # Evalúa la cadena de la función con x y nombres permitidos
        return eval(func_str, {"__builtins__": None}, {**nombres_permitidos, "x": x})
    except Exception as e:
            # Provide a clearer hint for common syntax mistakes
            msg = str(e)
            if 'invalid decimal literal' in msg:
                msg += (
                    " - revisa que uses el operador '*' cuando multiplies.\n"
                    "    Por ejemplo: en lugar de escribir '0.4e**x**2' usa '0.4*e**x**2' o '0.4*math.e**x**2'."
                    "\n    Evita la notación científica con 'e' sin multiplicar."
                )
            raise ValueError(f"Error al evaluar la función: {msg}")


def metodo_punto_fijo(g_str, x0, tol=1e-6, max_iter=100):
    """
    Encuentra la raíz de f(x) = 0 usando el método de Punto Fijo.
    Requiere que la función g(x) converja, es decir, |g'(x)| < 1 en el intervalo.
    Retorna una tupla (raiz, tabla_datos).
    """
    print(f"Iniciando método de Punto Fijo con x0 = {x0}")
    x = x0
    tabla_datos = []

    for iteracion in range(max_iter):
        try:
            x_nuevo = evaluar_funcion(g_str, x)
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
    x_vals = np.linspace(a, b, 1000)
    y_vals = []

    for x in x_vals:
        try:
            y = evaluar_funcion(func_str, x)
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
