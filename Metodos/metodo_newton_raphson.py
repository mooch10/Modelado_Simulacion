import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
from Metodos.input_parser import (
    parse_real,
    parse_real_or_default,
    parse_int_or_default,
    build_numeric_function,
)

def metodo_newton_raphson(
    funcion_str: str,
    x0: float,
    tolerancia: float = 1e-6,
    max_iteraciones: int = 100,
    variable: str = 'x'
) -> Tuple[float, List[dict], bool]:
    """
    Implementa el método de Newton-Raphson para encontrar raíces.
    
    Parámetros:
    -----------
    funcion_str : str
        Función como string (ej: "x**3 - 2*x - 5")
    x0 : float
        Aproximación inicial
    tolerancia : float
        Error admitido (default: 1e-6)
    max_iteraciones : int
        Número máximo de iteraciones (default: 100)
    variable : str
        Variable de la función (default: 'x')
    
    Retorna:
    --------
    raiz : float
        La raíz encontrada
    iteraciones : list
        Lista de diccionarios con datos de cada iteración
    convergencia : bool
        True si convergió, False si no
    """
    
    x = sp.Symbol(variable)
    try:
        f_expr, f = build_numeric_function(funcion_str, variable)
    except Exception:
        print(f"Error: No se pudo interpretar la función '{funcion_str}'")
        return None, [], False
    
    # Calcular la derivada automáticamente
    f_prima_expr = sp.diff(f_expr, x)
    
    # Convertir a funciones evaluables
    f_prima = sp.lambdify(x, f_prima_expr, 'numpy')
    
    print(f"\n{'='*80}")
    print(f"MÉTODO DE NEWTON-RAPHSON")
    print(f"{'='*80}")
    print(f"Función: f({variable}) = {funcion_str}")
    print(f"Derivada: f'({variable}) = {f_prima_expr}")
    print(f"Aproximación inicial (x₀): {x0}")
    print(f"Tolerancia de error: {tolerancia}")
    print(f"Máximo de iteraciones: {max_iteraciones}")
    print(f"{'='*80}\n")
    
    iteraciones = []
    x_actual = x0
    convergencia = False
    
    for i in range(max_iteraciones):
        # Evaluar función y derivada
        f_x = f(x_actual)
        f_prima_x = f_prima(x_actual)
        
        # Verificar que la derivada no sea cero
        if abs(f_prima_x) < 1e-15:
            print(f"Error: La derivada es muy cercana a cero en x = {x_actual}")
            break
        
        # Calcular la siguiente aproximación
        x_siguiente = x_actual - (f_x / f_prima_x)
        
        # Calcular error
        error = abs(x_siguiente - x_actual)
        
        # Guardar datos de la iteración
        iteraciones.append({
            'Iteración': i,
            'x_n': x_actual,
            'f(x_n)': f_x,
            "f'(x_n)": f_prima_x,
            'x_(n+1)': x_siguiente,
            'Error |x_(n+1) - x_n|': error
        })
        
        # Verificar convergencia
        if error < tolerancia:
            convergencia = True
            print(f" Convergencia alcanzada en iteración {i}")
            x_actual = x_siguiente
            break
        
        x_actual = x_siguiente
    
    if not convergencia:
        print(f" No se alcanzó convergencia después de {max_iteraciones} iteraciones")
    
    return x_actual, iteraciones, convergencia


def mostrar_tabla_iteraciones(iteraciones: List[dict]) -> None:
    """Muestra las iteraciones en una tabla formateada"""
    if not iteraciones:
        print("No hay iteraciones que mostrar")
        return
    
    print("\n" + "="*160)
    print("TABLA DE ITERACIONES".center(160))
    print("="*160)
    
    # Encabezados centrados
    print(f"{'Iteración':^12} {'x_n':^20} {'f(x_n)':^20} {'f´(x_n)':^20} {'x_(n+1)':^20} {'Error':^20}")
    print("-"*160)
    
    # Datos centrados
    for i, iter_data in enumerate(iteraciones):
        x_n = iter_data['x_n']
        f_xn = iter_data['f(x_n)']
        f_prima_xn = iter_data["f'(x_n)"]
        x_n1 = iter_data['x_(n+1)']
        error = iter_data['Error |x_(n+1) - x_n|']
        
        print(f"{i:^12} {x_n:^20.10g} {f_xn:^20.10g} {f_prima_xn:^20.10g} {x_n1:^20.10g} {error:^20.10g}")
    
    print("="*160 + "\n")


def graficar_funcion(funcion_str: str, raiz: float, variable: str = 'x') -> None:
    """Grafica la función y marca la raíz encontrada"""
    try:
        _, f = build_numeric_function(funcion_str, variable)
    except Exception as e:
        raise ValueError(f"No se pudo interpretar la función para graficar: {e}")
    
    # Determinar rango automáticamente basado en la raíz
    margen = max(abs(raiz) * 0.5, 2)
    x_vals = np.linspace(raiz - margen, raiz + margen, 1000)
    y_vals = f(x_vals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f({variable}) = {funcion_str}')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    plt.plot(raiz, 0, 'ro', markersize=10, label=f'Raíz: {variable} = {raiz:.10e}')
    plt.grid(True, alpha=0.3)
    plt.xlabel(variable, fontsize=12)
    plt.ylabel('f(' + variable + ')', fontsize=12)
    plt.title('Método de Newton-Raphson', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def main():
    """Función principal - interfaz interactiva"""
    print("\n" + "="*80)
    print("CALCULADORA - MÉTODO DE NEWTON-RAPHSON")
    print("="*80)
    
    # Obtener entrada del usuario
    print("\nIngrese la función como expresión Python:")
    print("Ejemplos: x**3 - 2*x - 5, sin(x) - x/2, exp(x) - 3*x")
    funcion = input("\nf(x) = ").strip()
    
    if not funcion:
        print("Error: Debe ingresar una función")
        return
    
    try:
        x0 = parse_real(input("\nAproximación inicial (x₀): ").strip(), "x₀")
    except ValueError:
        print("Error: x₀ debe ser un número")
        return
    
    try:
        tol_input = input("\nTolerancia de error (default 1e-6): ").strip()
        tolerancia = parse_real_or_default(tol_input, 1e-6, "tolerancia")
    except ValueError:
        print("Error: Tolerancia debe ser un número")
        return
    
    try:
        iter_input = input("\nMáximo de iteraciones (default 100): ").strip()
        max_iter = parse_int_or_default(iter_input, 100, "máximo de iteraciones")
    except ValueError:
        print("Error: Máximo de iteraciones debe ser un entero")
        return
    
    # Ejecutar método
    raiz, iteraciones, convergencia = metodo_newton_raphson(
        funcion, 
        x0, 
        tolerancia, 
        max_iter
    )
    
    # Mostrar resultados
    if iteraciones:
        mostrar_tabla_iteraciones(iteraciones)
        
        print("\n" + "="*80)
        print("RESULTADO FINAL")
        print("="*80)
        print(f"Raíz encontrada: x = {raiz:.15e}")
        print(f"Convergencia: {' Sí' if convergencia else ' No'}")
        print(f"Número de iteraciones: {len(iteraciones)}")
        print("="*80 + "\n")
        
        # Opción de graficar
        graficar = input("¿Desea graficar la función? (s/n): ").strip().lower()
        if graficar == 's' or graficar == 'si':
            try:
                graficar_funcion(funcion, raiz)
            except Exception as e:
                print(f"Error al graficar: {e}")
    else:
        print("No se pudieron obtener resultados.")


if __name__ == "__main__":
    main()
