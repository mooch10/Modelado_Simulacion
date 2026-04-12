import matplotlib.pyplot as plt
import numpy as np
from Metodos.input_parser import (
    parse_real,
    parse_real_or_default,
    parse_int_or_default,
    build_numeric_function,
)

"""
╔════════════════════════════════════════════════════════════════╗
║                    MACHETE: MÉTODO BISECCIÓN                   ║
╚════════════════════════════════════════════════════════════════╝

DEFINICIÓN:
Método para hallar raíces de f(x)=0 dividiendo repetidamente un intervalo
en dos mitades, descartando la que no contiene la raíz.

UTILIDAD:
Encontrar raíces de ecuaciones no lineales cuando f es continua.
Garantiza convergencia si f(a)·f(b) < 0.

PASOS:
1. Verificar que f(a)·f(b) < 0 (signos opuestos)
2. Calcular punto medio: c = (a+b)/2
3. Evaluar f(c)
4. Si |f(c)| < tol → convergió
5. Si f(a)·f(c) < 0 → nueva región [a,c]
6. Si no → nueva región [c,b]
7. Repetir hasta convergencia

FÓRMULA:
c = (a + b) / 2
Error absoluto: |c_nuevo - c_anterior|

REQUISITOS:
• f(x) continua en [a,b]
• f(a)·f(b) < 0 (signos opuestos)
• Tolerancia > 0
• Máximo de iteraciones definido
"""

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


def evaluar_funcion(f_num, x):
    """
    Evalúa la función definida por el usuario en un valor dado de x.
    Soporta funciones matemáticas básicas como sin, cos, exp, etc.
    """
    try:
        return _evaluar_numero(f_num(x))
    except Exception as e:
        raise ValueError(f"Error al evaluar la función: {e}")

def metodo_biseccion(func_str, a, b, tol=1e-6, max_iter=100):
    """
    Encuentra la raíz de f(x) = 0 usando el método de Bisección.
    Requiere que f(a) * f(b) < 0.
    Retorna una tupla (raiz, tabla_datos).
    """
    _, f_num = build_numeric_function(func_str)

    fa = evaluar_funcion(f_num, a)
    fb = evaluar_funcion(f_num, b)

    if fa * fb >= 0:
        raise ValueError("La función debe tener signos opuestos en a y b (f(a) * f(b) < 0)")

    print(f"Iniciando método de Bisección con intervalo [{a}, {b}]")
    print(f"f({a}) = {fa}, f({b}) = {fb}")

    # Lista para almacenar los datos de cada iteración
    tabla_datos = []
    raiz = None

    for iteracion in range(max_iter):
        c = (a + b) / 2
        fc = evaluar_funcion(f_num, c)

        # Agregar datos a la tabla
        tabla_datos.append((iteracion + 1, a, b, c, fa, fb, fc))

        print(f"Iteración {iteracion + 1}: c = {c}, f(c) = {fc}")

        if abs(fc) < tol:
            print(f"Convergido a la raíz: {c} después de {iteracion + 1} iteraciones")
            raiz = c
            # Imprimir tabla
            imprimir_tabla(tabla_datos)
            return raiz, tabla_datos

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    print(f"Máximo de iteraciones alcanzado. Raíz aproximada: {(a + b) / 2}")
    raiz = (a + b) / 2
    # Agregar la última iteración si no convergió
    tabla_datos.append((max_iter + 1, a, b, raiz, fa, fb, evaluar_funcion(f_num, raiz)))
    imprimir_tabla(tabla_datos)
    return raiz, tabla_datos

def imprimir_tabla(datos):
    """
    Imprime una tabla con los datos de cada iteración del método de bisección.
    """
    print("\nTabla de Iteraciones:")
    print("=" * 60)
    print(f"{'Iter':^5} {'a':^15} {'b':^15} {'c':^15} {'f(c)':^10}")
    print("=" * 60)
    for iteracion, a, b, c, fa, fb, fc in datos:
        print(f"{iteracion:^5} {a:^15.6f} {b:^15.6f} {c:^15.6f} {fc:^10.6f}")
    print("=" * 60)

def graficar_funcion(func_str, a, b, raiz):
    """
    Grafica la función y marca la raíz encontrada.
    """
    try:
        _, f_num = build_numeric_function(func_str)

        # Crear puntos x para el gráfico
        x = np.linspace(a, b, 300)
        y = []
        
        for xi in x:
            try:
                y.append(evaluar_funcion(f_num, xi))
            except:
                y.append(np.nan)
        
        y = np.array(y)
        
        # Crear la figura
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', linewidth=2, label=f'f(x) = {func_str}')
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Marcar la raíz encontrada
        plt.plot(raiz, 0, 'ro', markersize=10, label=f'Raíz: {raiz:.6f}')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.title('Gráfico del Método de Bisección', fontsize=14)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error al graficar: {e}")

def main():
    print("Búsqueda de Raíces usando Método de Bisección")
    print("============================================")

    # Obtener entradas del usuario
    func_str = input("Ingresa la función en términos de x (ej. x**2 - 2): ")
    a = parse_real(input("Ingresa el límite inferior a: "), "a")
    b = parse_real(input("Ingresa el límite superior b: "), "b")
    tol_input = input("Ingresa la tolerancia (por defecto 1e-6): ")
    tol = parse_real_or_default(tol_input, 1e-6, "tolerancia")
    max_iter_input = input("Ingresa el máximo de iteraciones (por defecto 100): ")
    max_iter = parse_int_or_default(max_iter_input, 100, "máximo de iteraciones")

    try:
        raiz, tabla_datos = metodo_biseccion(func_str, a, b, tol, max_iter)
        print(f"\nRaíz aproximada encontrada: {raiz}")
        
        # Preguntar si desea ver el gráfico
        ver_grafico = input("\n¿Deseas ver el gráfico de la función? (s/n): ").lower()
        if ver_grafico == 's':
            graficar_funcion(func_str, a, b, raiz)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()