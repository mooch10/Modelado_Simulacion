import math
import re
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Callable, Tuple, List, Dict

# Funciones del método de Newton-Raphson
def evaluar_funcion_newton(funcion_str: str, x: float, variable: str = 'x') -> float:
    """Evalúa la función simbólica en un punto"""
    x_sym = sp.Symbol(variable)
    f_expr = sp.sympify(funcion_str, locals={"e": sp.E, "pi": sp.pi})
    f = sp.lambdify(x_sym, f_expr, 'numpy')
    return f(x)

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
    
    # Convertir string a expresión simbólica
    x = sp.Symbol(variable)
    try:
        # Si el usuario usa "e" para el número de Euler, mapearlo a sympy.E
        # para evitar errores al lambdificar (numpy.log no puede manejar símbolos).
        f_expr = sp.sympify(funcion_str, locals={"e": sp.E, "pi": sp.pi})
    except Exception:
        print(f"Error: No se pudo interpretar la función '{funcion_str}'")
        return None, [], False
    
    # Calcular la derivada automáticamente
    f_prima_expr = sp.diff(f_expr, x)
    
    # Convertir a funciones evaluables
    f = sp.lambdify(x, f_expr, 'numpy')
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

def mostrar_tabla_newton(iteraciones: List[dict]) -> None:
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

def graficar_newton(funcion_str: str, raiz: float, variable: str = 'x') -> None:
    """Grafica la función y marca la raíz encontrada"""
    x = sp.Symbol(variable)
    f_expr = sp.sympify(funcion_str)
    f = sp.lambdify(x, f_expr, 'numpy')
    
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

# Funciones del método de Aitken
def evaluar_funcion_aitken(func_str, x):
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
    convergencia = False
    
    try:
        # Usamos x0 como semilla inicial
        x0_iter = x0
        
        for i in range(max_iteraciones):
            # Calcular los tres valores requeridos para la aceleración de Aitken
            x1 = evaluar_funcion_aitken(g_str, x0_iter)
            x2 = evaluar_funcion_aitken(g_str, x1)
            
            # Calcular el denominador de la fórmula de Aitken
            denominador = x2 - 2 * x1 + x0_iter
            
            # Verificar que no sea singular
            if abs(denominador) < 1e-15:
                # Si el denominador es muy pequeño, usar la iteración simple
                x_acelerado = x2
                es_acelerado = False
            else:
                # Aplicar fórmula de Aitken
                numerador = (x1 - x0_iter) ** 2
                x_acelerado = x0_iter - numerador / denominador
                es_acelerado = True
            
            # Calcular error con respecto a la semilla (x0_iter)
            error = abs(x_acelerado - x0_iter)
            
            # Guardar datos de la iteración
            iteraciones.append({
                'Iteración': i + 1,
                'x_(n-1)': x0_iter,
                'x_n': x1,
                'x_(n+1)': x2,
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
            
            # Usar el valor acelerado como nueva semilla para la siguiente iteración
            x0_iter = x_acelerado
        
        if not convergencia:
            print(f"\n⚠ No se alcanzó convergencia después de {max_iteraciones} iteraciones")
            print(f"Valor aproximado final: {x_acelerado}")
        
        return x_acelerado, iteraciones, convergencia
    
    except ValueError as e:
        print(f"Error durante la evaluación: {e}")
        return None, [], False

def mostrar_tabla_aitken(iteraciones: List[Dict]) -> None:
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

def graficar_aitken(
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
                y = evaluar_funcion_aitken(g_str, x)
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

# Funciones del método de Bisección
def evaluar_funcion_biseccion(func_str, x):
    """
    Evalúa la función definida por el usuario en un valor dado de x.
    Soporta funciones matemáticas básicas como sin, cos, exp, etc.
    """
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
            "pow": pow,
            "x": x
        }
        # Evalúa la cadena de la función con x y nombres permitidos
        return eval(func_str, {"__builtins__": {}}, nombres_permitidos)
    except Exception as e:
        raise ValueError(f"Error al evaluar la función: {e}")

def metodo_biseccion(func_str, a, b, tol=1e-6, max_iter=100):
    """
    Encuentra la raíz de f(x) = 0 usando el método de Bisección.
    Requiere que f(a) * f(b) < 0.
    Retorna una tupla (raiz, tabla_datos).
    """
    fa = evaluar_funcion_biseccion(func_str, a)
    fb = evaluar_funcion_biseccion(func_str, b)

    if fa * fb >= 0:
        raise ValueError("La función debe tener signos opuestos en a y b (f(a) * f(b) < 0)")

    print(f"Iniciando método de Bisección con intervalo [{a}, {b}]")
    print(f"f({a}) = {fa}, f({b}) = {fb}")

    # Lista para almacenar los datos de cada iteración
    tabla_datos = []
    raiz = None

    for iteracion in range(max_iter):
        c = (a + b) / 2
        fc = evaluar_funcion_biseccion(func_str, c)

        # Agregar datos a la tabla
        tabla_datos.append((iteracion + 1, a, b, c, fa, fb, fc))

        print(f"Iteración {iteracion + 1}: c = {c}, f(c) = {fc}")

        if abs(fc) < tol:
            print(f"Convergido a la raíz: {c} después de {iteracion + 1} iteraciones")
            raiz = c
            # Imprimir tabla
            imprimir_tabla_biseccion(tabla_datos)
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
    tabla_datos.append((max_iter + 1, a, b, raiz, fa, fb, evaluar_funcion_biseccion(func_str, raiz)))
    imprimir_tabla_biseccion(tabla_datos)
    return raiz, tabla_datos

def imprimir_tabla_biseccion(datos):
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

def graficar_biseccion(func_str, a, b, raiz):
    """
    Grafica la función y marca la raíz encontrada.
    """
    try:
        # Crear puntos x para el gráfico
        x = np.linspace(a, b, 300)
        y = []
        
        for xi in x:
            try:
                y.append(evaluar_funcion_biseccion(func_str, xi))
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

# Funciones del método de Punto Fijo
def evaluar_funcion_punto_fijo(func_str, x):
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
            x_nuevo = evaluar_funcion_punto_fijo(g_str, x)
        except ValueError as e:
            raise ValueError(f"Error en la evaluación de g(x): {e}")

        # Calcular el error absoluto
        error = abs(x_nuevo - x)
        tabla_datos.append((iteracion + 1, x, x_nuevo, error))

        print(f"Iteración {iteracion + 1}: x = {x_nuevo}")

        if error < tol:
            print(f"Convergido a la raíz: {x_nuevo} después de {iteracion + 1} iteraciones")
            imprimir_tabla_punto_fijo(tabla_datos)
            return x_nuevo, tabla_datos

        x = x_nuevo

    print(f"Máximo de iteraciones alcanzado. Valor aproximado: {x}")
    imprimir_tabla_punto_fijo(tabla_datos)
    return x, tabla_datos

def imprimir_tabla_punto_fijo(datos):
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

def graficar_punto_fijo(func_str, a, b, raiz=None):
    """
    Grafica la función f(x) en el intervalo [a, b].
    Si se proporciona una raíz, la marca en el gráfico.
    """
    x_vals = np.linspace(a, b, 1000)
    y_vals = []

    for x in x_vals:
        try:
            y = evaluar_funcion_punto_fijo(func_str, x)
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

def comparativa_metodos():
    """Ejecuta los 4 métodos con la misma función y compara resultados"""
    print("\n" + "="*100)
    print("COMPARATIVA DE LOS 4 MÉTODOS".center(100))
    print("="*100)
    
    # Recolectar parámetros comunes
    print("\nIngrese los parámetros para ejecutar los 4 métodos:")
    print("-" * 100)
    
    funcion = input("\n1. Función principal f(x) (ej: x**3 - 2*x - 5): ").strip()
    if not funcion:
        print("Error: Debe ingresar una función")
        return
    
    g_iteracion = input("2. Función de iteración g(x) (para Aitken y Punto Fijo, ej: cos(x)): ").strip()
    if not g_iteracion:
        print("Error: Debe ingresar una función de iteración")
        return
    
    try:
        x0 = float(input("3. Aproximación inicial x₀: "))
    except ValueError:
        print("Error: x₀ debe ser un número")
        return
    
    try:
        a = float(input("4. Límite inferior a (para Bisección): "))
        b = float(input("5. Límite superior b (para Bisección): "))
    except ValueError:
        print("Error: Los límites deben ser números")
        return
    
    try:
        tol_input = input("6. Tolerancia de error (default 1e-6): ").strip()
        tolerancia = float(tol_input) if tol_input else 1e-6
    except ValueError:
        print("Error: Tolerancia debe ser un número")
        return
    
    try:
        iter_input = input("7. Máximo de iteraciones (default 100): ").strip()
        max_iteraciones = int(iter_input) if iter_input else 100
    except ValueError:
        print("Error: Máximo de iteraciones debe ser un entero")
        return
    
    # Diccionario para almacenar resultados
    resultados = {}
    
    # Ejecutar Newton-Raphson
    print("\n" + "-"*100)
    print("Ejecutando Método de Newton-Raphson...")
    print("-"*100)
    try:
        raiz_nr, iter_nr, conv_nr = metodo_newton_raphson(funcion, x0, tolerancia, max_iteraciones)
        if raiz_nr is not None:
            resultados['Newton-Raphson'] = {
                'raiz': raiz_nr,
                'iteraciones': len(iter_nr),
                'error': iter_nr[-1]['Error |x_(n+1) - x_n|'] if iter_nr else 0,
                'convergencia': conv_nr
            }
    except Exception as e:
        print(f"Error en Newton-Raphson: {e}")
    
    # Ejecutar Aitken
    print("\n" + "-"*100)
    print("Ejecutando Método de Aceleración de Aitken...")
    print("-"*100)
    try:
        raiz_aitken, iter_aitken, conv_aitken = metodo_aitken(g_iteracion, x0, tolerancia, max_iteraciones)
        if raiz_aitken is not None:
            resultados['Aitken'] = {
                'raiz': raiz_aitken,
                'iteraciones': len(iter_aitken),
                'error': iter_aitken[-1]['Error'] if iter_aitken else 0,
                'convergencia': conv_aitken
            }
    except Exception as e:
        print(f"Error en Aitken: {e}")
    
    # Ejecutar Bisección
    print("\n" + "-"*100)
    print("Ejecutando Método de Bisección...")
    print("-"*100)
    try:
        raiz_bis, iter_bis = metodo_biseccion(funcion, a, b, tolerancia, max_iteraciones)
        if raiz_bis is not None:
            resultados['Bisección'] = {
                'raiz': raiz_bis,
                'iteraciones': len(iter_bis),
                'error': abs(iter_bis[-1][6]) if iter_bis else 0,
                'convergencia': True
            }
    except Exception as e:
        print(f"Error en Bisección: {e}")
    
    # Ejecutar Punto Fijo
    print("\n" + "-"*100)
    print("Ejecutando Método de Punto Fijo...")
    print("-"*100)
    try:
        raiz_pf, iter_pf = metodo_punto_fijo(g_iteracion, x0, tolerancia, max_iteraciones)
        if raiz_pf is not None:
            resultados['Punto Fijo'] = {
                'raiz': raiz_pf,
                'iteraciones': len(iter_pf),
                'error': iter_pf[-1][3] if iter_pf else 0,
                'convergencia': True
            }
    except Exception as e:
        print(f"Error en Punto Fijo: {e}")
    
    # Mostrar tabla comparativa
    print("\n\n" + "="*130)
    print("TABLA COMPARATIVA DE RESULTADOS".center(130))
    print("="*130)
    print(f"{'Método':^25} {'Raíz encontrada':^30} {'Iteraciones':^15} {'Error final':^25} {'Convergencia':^15}")
    print("-"*130)
    
    for metodo, datos in resultados.items():
        raiz = datos['raiz']
        iteraciones = datos['iteraciones']
        error = datos['error']
        convergencia = "✓ Sí" if datos['convergencia'] else "✗ No"
        
        print(f"{metodo:^25} {raiz:^30.15e} {iteraciones:^15} {error:^25.10e} {convergencia:^15}")
    
    print("="*130)
    
    # Mostrar análisis adicional
    if resultados:
        print("\n" + "="*130)
        print("ANÁLISIS COMPARATIVO".center(130))
        print("="*130)
        
        # Raíz promedio
        raices = [datos['raiz'] for datos in resultados.values()]
        raiz_promedio = np.mean(raices)
        print(f"\nRaíz promedio: {raiz_promedio:.15e}")
        
        # Método más rápido (menos iteraciones)
        metodo_rapido = min(resultados.items(), key=lambda x: x[1]['iteraciones'])
        print(f"Método más rápido (menos iteraciones): {metodo_rapido[0]} con {metodo_rapido[1]['iteraciones']} iteraciones")
        
        # Método con menor error final
        metodo_preciso = min(resultados.items(), key=lambda x: x[1]['error'])
        print(f"Método más preciso (menor error): {metodo_preciso[0]} con error de {metodo_preciso[1]['error']:.10e}")
        
        # Divergencia entre raíces
        if len(raices) > 1:
            divergencia_max = max(raices) - min(raices)
            print(f"Divergencia máxima entre raíces: {divergencia_max:.10e}")
        
        print("="*130 + "\n")


def main():
    while True:
        print("\n" + "="*80)
        print("MENÚ PRINCIPAL - MÉTODOS NUMÉRICOS PARA RAÍCES")
        print("="*80)
        print("1. Método de Newton-Raphson")
        print("2. Método de Aceleración de Aitken")
        print("3. Método de Bisección")
        print("4. Método de Punto Fijo")
        print("5. Comparativa de los 4 métodos")
        print("6. Salir")
        print("="*80)
        
        try:
            opcion = int(input("Selecciona una opción (1-6): ").strip())
        except ValueError:
            print("Error: Debe ingresar un número entero.")
            continue
        
        if opcion == 1:
            # Newton-Raphson
            print("\nIngrese la función como expresión Python:")
            print("Ejemplos: x**3 - 2*x - 5, sin(x) - x/2, exp(x) - 3*x")
            funcion = input("\nf(x) = ").strip()
            
            if not funcion:
                print("Error: Debe ingresar una función")
                continue
            
            try:
                x0 = float(input("\nAproximación inicial (x₀): ").strip())
            except ValueError:
                print("Error: x₀ debe ser un número")
                continue
            
            try:
                tol_input = input("\nTolerancia de error (default 1e-6): ").strip()
                tolerancia = float(tol_input) if tol_input else 1e-6
            except ValueError:
                print("Error: Tolerancia debe ser un número")
                continue
            
            try:
                iter_input = input("\nMáximo de iteraciones (default 100): ").strip()
                max_iter = int(iter_input) if iter_input else 100
            except ValueError:
                print("Error: Máximo de iteraciones debe ser un entero")
                continue
            
            # Ejecutar método
            raiz, iteraciones, convergencia = metodo_newton_raphson(
                funcion, 
                x0, 
                tolerancia, 
                max_iter
            )
            
            # Mostrar resultados
            if iteraciones:
                mostrar_tabla_newton(iteraciones)
                
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
                        graficar_newton(funcion, raiz)
                    except Exception as e:
                        print(f"Error al graficar: {e}")
        
        elif opcion == 2:
            # Aitken
            try:
                g_str = input("\nIngrese la función de iteración g(x) (ej: cos(x), x**2/10 + 1, sqrt(x)): ").strip()
                
                if not g_str:
                    print("Error: Debe ingresar una función")
                    continue
                
                x0 = float(input("Ingrese la aproximación inicial x₀: "))
                
                tolerancia_input = input("Ingrese la tolerancia de error (Enter para 1e-6): ").strip()
                tolerancia = float(tolerancia_input) if tolerancia_input else 1e-6
                
                max_iter_input = input("Ingrese el máximo de iteraciones (Enter para 100): ").strip()
                max_iteraciones = int(max_iter_input) if max_iter_input else 100
                
                # Ejecutar el método
                raiz, iteraciones, convergencia = metodo_aitken(g_str, x0, tolerancia, max_iteraciones)
                
                if raiz is not None:
                    # Mostrar tabla
                    mostrar_tabla_aitken(iteraciones)
                    
                    # Resumen final
                    print(f"{'='*100}")
                    print(f"RESUMEN DE RESULTADOS".center(100))
                    print(f"{'='*100}")
                    print(f"{'Raíz encontrada:':.<50} {raiz:.15f}")
                    print(f"{'Número de iteraciones:':.<50} {len(iteraciones)}")
                    print(f"{'Convergencia:':.<50} {' SÍ' if convergencia else '✗ NO'}")
                    print(f"{'='*100}\n")
                    
                    # Ofrecer opción de visualización
                    mostrar_grafico = input("¿Desea visualizar el gráfico de convergencia? (s/n): ").strip().lower()
                    if mostrar_grafico in ['s', 'si', 'sí', 'yes', 'y']:
                        graficar_aitken(g_str, x0, raiz, iteraciones)
            
            except ValueError as e:
                print(f"Error de entrada: {e}")
        
        elif opcion == 3:
            # Bisección
            func_str = input("Ingresa la función en términos de x (ej. x**2 - 2): ")
            a = float(input("Ingresa el límite inferior a: "))
            b = float(input("Ingresa el límite superior b: "))
            tol_input = input("Ingresa la tolerancia (por defecto 1e-6): ")
            tol = float(tol_input) if tol_input else 1e-6
            max_iter_input = input("Ingresa el máximo de iteraciones (por defecto 100): ")
            max_iter = int(max_iter_input) if max_iter_input else 100

            try:
                raiz, tabla_datos = metodo_biseccion(func_str, a, b, tol, max_iter)
                print(f"\nRaíz aproximada encontrada: {raiz}")
                
                # Preguntar si desea ver el gráfico
                ver_grafico = input("\n¿Deseas ver el gráfico de la función? (s/n): ").lower()
                if ver_grafico == 's':
                    graficar_biseccion(func_str, a, b, raiz)
            except ValueError as e:
                print(f"Error: {e}")
        
        elif opcion == 4:
            # Punto Fijo
            func_str = input("Ingresa la función f(x) = 0 (ej. x**2 - 2): ")
            g_str = input("Ingresa la función g(x) para el punto fijo (ej. x - (x**2 - 2)/(2*x)): ")
            x0 = float(input("Ingresa el valor inicial x0: "))
            tol_input = input("Ingresa la tolerancia (por defecto 1e-6): ")
            tol = float(tol_input) if tol_input else 1e-6
            max_iter_input = input("Ingresa el máximo de iteraciones (por defecto 100): ")
            max_iter = int(max_iter_input) if max_iter_input else 100

            try:
                raiz, tabla_datos = metodo_punto_fijo(g_str, x0, tol, max_iter)
                print(f"\nRaíz aproximada encontrada: {raiz}")

                # Preguntar si quiere graficar
                graficar = input("¿Quieres ver el gráfico de la función? (s/n): ").lower()
                if graficar == 's':
                    a = float(input("Ingresa el límite inferior para el gráfico: "))
                    b = float(input("Ingresa el límite superior para el gráfico: "))
                    graficar_punto_fijo(func_str, a, b, raiz)
            except ValueError as e:
                print(f"Error: {e}")
        
        elif opcion == 5:
            # Comparativa de los 4 métodos
            comparativa_metodos()
        
        elif opcion == 6:
            print("¡Hasta luego!")
            break
        
        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()