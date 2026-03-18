import math
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


def evaluar_funcion(func_str, x):
    """
    Evalúa la función definida por el usuario en un valor dado de x.
    Soporta funciones matemáticas básicas como sin, cos, exp, etc.
    """
    # Corrija casos típicos donde se olvida el operador '*' entre un número y
    # la constante e, por ejemplo "0.4e**x**2" -> "0.4*e**x**2"
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
        msg = str(e)
        if 'invalid decimal literal' in msg:
            msg += (
                " - revisa que uses el operador '*' cuando multiplies.\n"
                "    Por ejemplo: en lugar de escribir '0.4e**x**2' usa '0.4*e**x**2' o '0.4*math.e**x**2'."
                "\n    Evita la notación científica con 'e' sin multiplicar."
            )
        raise ValueError(f"Error al evaluar la función: {msg}")


def metodo_aitken(
    g_str: str,
    x0: float,
    tolerancia: float = 1e-6,
    max_iteraciones: int = 100
) -> Tuple[float, List[Dict], bool]:
    """
    Implementa el método de aceleración de Aitken para encontrar raíces.
    
    El método de Aitken acelera la convergencia de la iteración de punto fijo
    usando la fórmula de aceleración:
    
    x* = x₀ - (x₁ - x₀)² / (x₂ - 2x₁ + x₀)
    
    Parámetros:
    -----------
    g_str : str
        Función de iteración como string (ej: "cos(x)")
    x0 : float
        Aproximación inicial
    tolerancia : float
        Error admitido (default: 1e-6)
    max_iteraciones : int
        Número máximo de iteraciones (default: 100)
    
    Retorna:
    --------
    raiz : float
        La raíz encontrada
    iteraciones : list
        Lista de diccionarios con datos de cada iteración
    convergencia : bool
        True si convergió, False si no
    """
    
    print(f"\n{'='*100}")
    print(f"{'MÉTODO DE ACELERACIÓN DE AITKEN':^100}")
    print(f"{'='*100}")
    print(f"Función de iteración: g(x) = {g_str}")
    print(f"Aproximación inicial (x₀): {x0}")
    print(f"Tolerancia de error: {tolerancia}")
    print(f"Máximo de iteraciones: {max_iteraciones}")
    print(f"{'='*100}\n")
    
    iteraciones = []
    x_anterior = x0
    convergencia = False
    
    try:
        # Calcular primeras dos iteraciones
        x_actual = evaluar_funcion(g_str, x_anterior)
        x_siguiente = evaluar_funcion(g_str, x_actual)
        
        for i in range(max_iteraciones):
            # Calcular el denominador de la fórmula de Aitken
            denominador = x_siguiente - 2 * x_actual + x_anterior
            
            # Verificar que no sea singular
            if abs(denominador) < 1e-15:
                # Si el denominador es muy pequeño, continuar con iteración normal
                x_acelerado = x_siguiente
                es_acelerado = False
            else:
                # Aplicar fórmula de Aitken
                numerador = (x_actual - x_anterior) ** 2
                x_acelerado = x_anterior - numerador / denominador
                es_acelerado = True
            
            # Calcular error con respecto a la iteración anterior
            error = abs(x_acelerado - x_anterior)
            
            # Guardar datos de la iteración
            iteraciones.append({
                'Iteración': i + 1,
                'x_(n-1)': x_anterior,
                'x_n': x_actual,
                'x_(n+1)': x_siguiente,
                'x_acelerado': x_acelerado,
                'Error': error,
                'Acelerado': '✓' if es_acelerado else '✗'
            })
            
            print(f"Iteración {i + 1}: x_acelerado = {x_acelerado:.10f}, Error = {error:.2e}")
            
            # Verificar convergencia
            if error < tolerancia:
                convergencia = True
                print(f"\n✓ Convergencia alcanzada en iteración {i + 1}")
                break
            
            # Actualizar valores para la siguiente iteración
            x_anterior = x_actual
            x_actual = x_siguiente
            x_siguiente = evaluar_funcion(g_str, x_acelerado)
        
        if not convergencia:
            print(f"\n⚠ No se alcanzó convergencia después de {max_iteraciones} iteraciones")
            print(f"Valor aproximado final: {x_acelerado}")
        
        return x_acelerado, iteraciones, convergencia
    
    except ValueError as e:
        print(f"Error durante la evaluación: {e}")
        return None, [], False


def mostrar_tabla_iteraciones(iteraciones: List[Dict]) -> None:
    """Muestra las iteraciones en una tabla formateada y centrada"""
    if not iteraciones:
        print("No hay iteraciones que mostrar")
        return
    
    print("\n" + "="*150)
    print("TABLA DE ITERACIONES - MÉTODO DE AITKEN".center(150))
    print("="*150)
    
    # Encabezados centrados
    print(f"{'Iter':^6} {'x_(n-1)':^20} {'x_n':^20} {'x_(n+1)':^20} {'x_acelerado':^25} {'Error':^20} {'Ace.':^7}")
    print("-"*150)
    
    # Datos centrados
    for iter_data in iteraciones:
        iter_num = iter_data['Iteración']
        x_n_minus_1 = iter_data['x_(n-1)']
        x_n = iter_data['x_n']
        x_n_plus_1 = iter_data['x_(n+1)']
        x_acelerado = iter_data['x_acelerado']
        error = iter_data['Error']
        es_acelerado = iter_data['Acelerado']
        
        print(f"{iter_num:^6} {x_n_minus_1:^20.10g} {x_n:^20.10g} {x_n_plus_1:^20.10g} {x_acelerado:^25.10g} {error:^20.10g} {es_acelerado:^7}")
    
    print("="*150 + "\n")


def graficar_metodo_aitken(
    g_str: str,
    x0: float,
    raiz: float,
    iteraciones: List[Dict]
) -> None:
    """
    Grafica la convergencia del método de Aitken.
    Muestra la función g(x), la línea y=x, y los puntos de iteración.
    """
    try:
        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Primer subplot: Gráfica de g(x) y la línea y=x
        margen = max(abs(raiz) * 0.5, 2)
        x_vals = np.linspace(raiz - margen, raiz + margen, 1000)
        
        # Evaluar g(x)
        g_vals = []
        for x in x_vals:
            try:
                y = evaluar_funcion(g_str, x)
                g_vals.append(y)
            except:
                g_vals.append(np.nan)
        
        g_vals = np.array(g_vals)
        
        ax1.plot(x_vals, g_vals, 'b-', linewidth=2, label=f'g(x) = {g_str}')
        ax1.plot(x_vals, x_vals, 'r--', linewidth=2, label='y = x')
        ax1.plot(raiz, raiz, 'go', markersize=10, label=f'Raíz encontrada: {raiz:.10f}')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.set_title('Método de Aitken - Gráfica de g(x)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.set_xlim(raiz - margen, raiz + margen)
        
        # Segundo subplot: Convergencia (Error vs Iteración)
        if iteraciones:
            iteracion_nums = [iter_data['Iteración'] for iter_data in iteraciones]
            errores = [iter_data['Error'] for iter_data in iteraciones]
            
            ax2.semilogy(iteracion_nums, errores, 'bo-', linewidth=2, markersize=8, label='Error absoluto')
            ax2.axhline(y=1e-6, color='r', linestyle='--', linewidth=2, label='Tolerancia (1e-6)')
            ax2.grid(True, alpha=0.3, which='both')
            ax2.set_xlabel('Iteración', fontsize=12)
            ax2.set_ylabel('Error (escala logarítmica)', fontsize=12)
            ax2.set_title('Convergencia del Método', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.set_xticks(range(1, len(iteracion_nums) + 1))
        
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"Error al graficar: {e}")


def main():
    """Función principal con interfaz de usuario"""
    print(f"\n{'*'*100}")
    print(f"{'MÉTODO DE ACELERACIÓN DE AITKEN':^100}")
    print(f"{'*'*100}")
    
    # Solicitar entrada del usuario
    try:
        g_str = input("\nIngrese la función de iteración g(x) (ej: cos(x), x**2/10 + 1, sqrt(x)): ").strip()
        
        if not g_str:
            print("Error: Debe ingresar una función")
            return
        
        x0 = float(input("Ingrese la aproximación inicial x₀: "))
        
        tolerancia_input = input("Ingrese la tolerancia de error (Enter para 1e-6): ").strip()
        tolerancia = float(tolerancia_input) if tolerancia_input else 1e-6
        
        max_iter_input = input("Ingrese el máximo de iteraciones (Enter para 100): ").strip()
        max_iteraciones = int(max_iter_input) if max_iter_input else 100
        
        # Ejecutar el método
        raiz, iteraciones, convergencia = metodo_aitken(g_str, x0, tolerancia, max_iteraciones)
        
        if raiz is not None:
            # Mostrar tabla
            mostrar_tabla_iteraciones(iteraciones)
            
            # Resumen final
            print(f"{'='*100}")
            print(f"RESUMEN DE RESULTADOS".center(100))
            print(f"{'='*100}")
            print(f"{'Raíz encontrada:':.<50} {raiz:.15f}")
            print(f"{'Número de iteraciones:':.<50} {len(iteraciones)}")
            print(f"{'Convergencia:':.<50} {'✓ SÍ' if convergencia else '✗ NO'}")
            print(f"{'='*100}\n")
            
            # Ofrecer opción de visualización
            mostrar_grafico = input("¿Desea visualizar el gráfico de convergencia? (s/n): ").strip().lower()
            if mostrar_grafico in ['s', 'si', 'sí', 'yes', 'y']:
                graficar_metodo_aitken(g_str, x0, raiz, iteraciones)
    
    except ValueError as e:
        print(f"Error de entrada: {e}")
    except KeyboardInterrupt:
        print("\n\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Error inesperado: {e}")


if __name__ == "__main__":
    main()
