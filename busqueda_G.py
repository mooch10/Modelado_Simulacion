import sympy as sp
import numpy as np
from sympy import symbols, solve, diff, lambdify, simplify
from itertools import combinations
from metodo_punto_fijo import metodo_punto_fijo


def obtener_despejes_punto_fijo(f_str, variable='x'):
    """
    Analiza una función f(x) = 0 y propone múltiples despejes válidos para g(x).
    
    Estrategias de despeje:
    1. Despejar x directamente: x = g(x)
    2. Manipulaciones algebraicas: sumar/restar términos
    3. Despejar de diferentes formas combinando términos
    4. Usar propiedades de logaritmo y exponencial
    
    Retorna una lista de tuplas (g_str, validez, descripcion)
    """
    try:
        x = symbols(variable)
        
        # Parsear la función ingresada
        # Reemplazar notaciones comunes
        f_str = f_str.replace('^', '**')
        f_str = f_str.replace('**x**', '**x**')  # Mantener notación de potencias
        
        # Convertir a expresión sympy
        f_expr = sp.sympify(f_str)
        
        despejes = []
        
        print(f"\n{'='*70}")
        print(f"Analizando f(x) = {f_expr}")
        print(f"{'='*70}\n")
        
        # ESTRATEGIA 1: Despejar x directamente de f(x) = 0
        print("Estrategia 1: Despejar x directamente...")
        try:
            soluciones = solve(f_expr, x)
            if soluciones:
                for sol in soluciones:
                    if x in sol.free_symbols:  # Solo si depende de x
                        g_str = str(sol)
                        despejes.append({
                            'g_str': g_str,
                            'expr': sol,
                            'metodo': 'Despeje directo',
                            'formula': f'x = {g_str}'
                        })
                        print(f"  [+] Encontrado: g(x) = {g_str}")
        except Exception:
            pass
        
        # ESTRATEGIA 2: Mover un término a cada lado: x = f(x) + c
        print("\nEstrategia 2: Despejes por reorganizacion...")
        try:
            f_expanded = sp.expand(f_expr)
            terminos = sp.Add.make_args(f_expanded)
            
            # Para cada término que contenga x, intentar aislarlo
            for i, termino_principal in enumerate(terminos):
                if termino_principal.has(x):
                    # Suma de los otros términos (con signo opuesto)
                    otros_terminos = sum([t for j, t in enumerate(terminos) if j != i])
                    
                    # Intenta despejar el término principal
                    try:
                        # Si es un término lineal en x
                        coef_x = sp.diff(termino_principal, x)
                        if coef_x != 0 and not coef_x.has(x):
                            # Es lineal: x = -otros_terminos / coef_x
                            solucion = -otros_terminos / coef_x
                            solucion = simplify(solucion)
                            if x in solucion.free_symbols:
                                g_str = str(solucion)
                                # Agregar si no está duplicado
                                if not any(str(d['expr']) == g_str for d in despejes):
                                    despejes.append({
                                        'g_str': g_str,
                                        'expr': solucion,
                                        'metodo': 'Reorganizacion lineal',
                                        'formula': f'x = {g_str}'
                                    })
                                    print(f"  [+] Encontrado: g(x) = {g_str}")
                    except:
                        pass
        except Exception:
            pass
        
        # ESTRATEGIA 3: Movimientos alternativos (x = algo)
        print("\nEstrategia 3: Despejes alternativos...")
        try:
            f_expanded = sp.expand(f_expr)
            
            # Intenta: x = f_expr (si es posible reorganizar)
            # Por ejemplo: si f(x) = x^2 - 2x - 3, intentar x = 2x + 3 - x^2, etc.
            
            terminos = sp.Add.make_args(f_expanded)
            
            # Generar combinaciones: mover cada termino a un lado
            for movidos in range(1, len(terminos)):
                for indices_movidos in combinations(range(len(terminos)), movidos):
                    try:
                        terminos_movidos = sum([terminos[j] for j in indices_movidos])
                        terminos_restantes = sum([terminos[j] for j in range(len(terminos)) if j not in indices_movidos])
                        
                        # Intenta resolver: terminos_movidos = -terminos_restantes
                        if terminos_movidos.has(x):
                            sol = solve(terminos_movidos + terminos_restantes, x)
                            if sol:
                                for s in sol:
                                    if x in s.free_symbols:
                                        g_str = str(s)
                                        if len(g_str) < 120 and not any(str(simplify(d['expr'] - s)) == '0' for d in despejes):
                                            despejes.append({
                                                'g_str': g_str,
                                                'expr': s,
                                                'metodo': 'Reorganizacion alternativa',
                                                'formula': f'x = {g_str}'
                                            })
                    except:
                        pass
        except Exception:
            pass
        
        # ESTRATEGIA 4: Usando propiedades especiales (log, exp, etc.)
        print("\nEstrategia 4: Despejes especiales...")
        try:
            # Si hay logaritm, intentar exponencial
            if f_expr.has(sp.log):
                terms_with_log = [t for t in sp.Add.make_args(f_expanded) if t.has(sp.log)]
                terms_without_log = [t for t in sp.Add.make_args(f_expanded) if not t.has(sp.log)]
                if terms_with_log and terms_without_log:
                    # Por ejemplo: log(x) = k pueden convertirse a x = exp(k)
                    pass
            
            # Si hay exponencial
            if f_expr.has(sp.exp):
                sol = solve(f_expr, x)
                for s in sol:
                    if x in s.free_symbols and not any(str(d['expr']) == str(s) for d in despejes):
                        g_str = str(s)
                        if len(g_str) < 120:
                            despejes.append({
                                'g_str': g_str,
                                'expr': s,
                                'metodo': 'Despeje especial (exp)',
                                'formula': f'x = {g_str}'
                            })
        except:
            pass
        
        # Eliminar duplicados basados en expresión simplificada
        despejes_unicos = []
        expresiones_vistas = []
        for d in despejes:
            expr_simp = simplify(d['expr'])
            es_duplicado = False
            for expr_vista in expresiones_vistas:
                if simplify(expr_simp - expr_vista) == 0:
                    es_duplicado = True
                    break
            if not es_duplicado:
                expresiones_vistas.append(expr_simp)
                despejes_unicos.append(d)
        
        return despejes_unicos, x, f_expr
        
    except Exception as exc:
        print(f"Error al analizar la función: {exc}")
        return [], None, None





def validar_convergencia(g_str, x_sym, punto_prueba=1.0, intervalo=0.5):
    """
    Valida si g(x) es convergente en un intervalo.
    Criterio: |g'(x)| < 1 en el intervalo [punto_prueba - intervalo, punto_prueba + intervalo]
    
    Retorna: (es_convergente, derivada_str, valor_derivada_en_punto)
    """
    try:
        x = symbols('x')
        g_expr = sp.sympify(g_str)
        
        # Calcular derivada
        g_prima = diff(g_expr, x)
        g_prima_str = str(g_prima)
        
        # Convertir a función para evaluar
        g_prima_func = lambdify(x, g_prima, 'numpy')
        
        # Evaluar en varios puntos del intervalo
        puntos = np.linspace(punto_prueba - intervalo, punto_prueba + intervalo, 10)
        valores_derivada = []
        
        for p in puntos:
            try:
                val = g_prima_func(p)
                if np.isfinite(val):
                    valores_derivada.append(abs(val))
            except:
                pass
        
        if valores_derivada:
            max_derivada = max(valores_derivada)
            valor_en_punto = g_prima_func(punto_prueba)
            es_convergente = max_derivada < 1.0
            
            return es_convergente, g_prima_str, valor_en_punto, max_derivada
        else:
            return False, g_prima_str, None, None
            
    except Exception as exc:
        return False, f"Error: {str(exc)}", None, None


def mostrar_opciones_despeje(f_str):
    """
    Interfaz principal: muestra todas las opciones de despeje para una función f(x).
    Valida cada opción y permite al usuario seleccionar una para usar en el método.
    """
    despejes, x_sym, f_expr = obtener_despejes_punto_fijo(f_str)
    
    if not despejes:
        print("\n[!] No se encontraron despejes validos para esta funcion.")
        return None
    
    print(f"\n{'='*70}")
    print(f"OPCIONES DE DESPEJE ENCONTRADAS: {len(despejes)}")
    print(f"{'='*70}\n")
    
    opciones_validas = []
    
    for idx, despeje in enumerate(despejes, 1):
        g_str = despeje['g_str']
        print(f"\n{'-'*70}")
        print(f"OPCION {idx}: {despeje['metodo']}")
        print(f"g(x) = {g_str}")
        print(f"Formula: {despeje['formula']}")
        
        # Validar convergencia
        es_convergente, g_prima_str, valor_derivada, max_derivada = validar_convergencia(g_str, x_sym)
        
        print(f"\nDerivada: g'(x) = {g_prima_str}")
        
        if valor_derivada is not None:
            print(f"  * |g'(x=1)| aprox. {abs(valor_derivada):.6f}")
            print(f"  * max|g'(x)| en [0.5, 1.5] aprox. {max_derivada:.6f}")
        
        if es_convergente:
            print(f"  [V] CONVERGENTE: |g'(x)| < 1 en el intervalo")
            opciones_validas.append({
                'indice': idx,
                'g_str': g_str,
                'metodo': despeje['metodo'],
                'derivada': g_prima_str,
                'valor_derivada': valor_derivada,
                'max_derivada': max_derivada
            })
        else:
            print(f"  [X] DIVERGENTE: |g'(x)| >= 1 (riesgo de no converger)")
    
    print(f"\n{'='*70}")
    print(f"Resumen: {len(opciones_validas)} opciones convergentes de {len(despejes)} totales")
    print(f"{'='*70}\n")
    
    return opciones_validas


def menu_interactivo():
    """
    Menu interactivo para el usuario.
    """
    print("\n" + "="*70)
    print("PROGRAMA PARA ENCONTRAR OPCIONES DE DESPEJE - METODO DEL PUNTO FIJO")
    print("="*70)
    
    while True:
        print("\n1. Ingresar nueva funcion f(x)")
        print("2. Salir")
        opcion = input("\nSelecciona una opcion: ").strip()
        
        if opcion == '2':
            print("Hasta luego!")
            break
        elif opcion == '1':
            print("\nIngresa la funcion f(x) = 0")
            print("Ejemplos: x**2 - 2*x - 3, sin(x) - x, exp(x) - 3*x")
            print("Operadores: +, -, *, /, ** (potencia)")
            print("Funciones: sin, cos, tan, exp, log, sqrt, abs, pow")
            print("Constantes: pi, e")
            
            f_str = input("\nf(x) = ").strip()
            
            opciones = mostrar_opciones_despeje(f_str)
            
            if opciones:
                print("\nDeseas probar con el metodo del punto fijo?")
                print("1. Si, seleccionar una opcion")
                print("2. No, volver al menu")
                
                opcion2 = input("\nSelecciona una opcion: ").strip()
                
                if opcion2 == '1':
                    while True:
                        try:
                            idx = int(input(f"\nSelecciona el numero de opcion (1-{len(opciones)}): ").strip())
                            if 1 <= idx <= len(opciones):
                                opcion_seleccionada = opciones[idx - 1]
                                
                                print(f"\nOpcion seleccionada: g(x) = {opcion_seleccionada['g_str']}")
                                
                                x0 = float(input("Ingresa valor inicial x0: ").strip())
                                tol = float(input("Ingresa tolerancia (ej: 1e-6): ").strip())
                                max_iter = int(input("Ingresa maximo de iteraciones (ej: 100): ").strip())
                                
                                # Ejecutar metodo del punto fijo
                                raiz, tabla = metodo_punto_fijo(
                                    opcion_seleccionada['g_str'],
                                    x0,
                                    tol=tol,
                                    max_iter=max_iter
                                )
                                
                                print(f"\n[V] Raiz encontrada: x = {raiz}")
                                
                                break
                            else:
                                print(f"Por favor, ingresa un numero entre 1 y {len(opciones)}")
                        except ValueError:
                            print("Ingresa un numero valido")
        else:
            print("Opcion no valida. Intenta de nuevo.")


if __name__ == "__main__":
    menu_interactivo()
