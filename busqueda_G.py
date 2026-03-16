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
                    # RECHAZAR si es constante pura (no tiene variables)
                    # ACEPTAR si contiene x o es una expresión con parámetros
                    if sol.free_symbols:  # Tiene símbolos (variables)
                        g_str = str(sol)
                        despejes.append({
                            'g_str': g_str,
                            'expr': sol,
                            'metodo': 'Despeje directo',
                            'formula': f'x = {g_str}'
                        })
                        print(f"  [+] Encontrado: g(x) = {g_str}")
            else:
                print("  (sin soluciones)")
        except Exception as e:
            print(f"  (error: {type(e).__name__})")
        
        # ESTRATEGIA 2: Movimientos simples de términos - MÍS AGRESIVA
        print("\nEstrategia 2: Despejes por movimiento de términos...")
        try:
            f_expanded = sp.expand(f_expr)
            terminos = sp.Add.make_args(f_expanded)
            
            # Estrategia: para cada término con x, aislarlo
            # Si f = a*x + b*f(x) + c = 0, entonces x = (-b*f(x) - c) / a
            
            for i, termino in enumerate(terminos):
                if termino.has(x):
                    # Intenta despejar este término
                    otros = sum([terminos[j] for j in range(len(terminos)) if j != i])
                    
                    try:
                        # Intenta resolver: termino = -otros para x
                        sols = solve(termino + otros, x)
                        if sols:
                            for sol in sols:
                                # SOLO aceptar si la solución contiene x (es función de x)
                                if sol.has(x):
                                    g_str = str(sol)
                                    # Aceptar si no es duplicado
                                    if len(g_str) < 150:
                                        es_duplicado = any(
                                            str(simplify(d['expr'] - sol)) == '0' 
                                            for d in despejes
                                        )
                                        if not es_duplicado:
                                            despejes.append({
                                                'g_str': g_str,
                                                'expr': sol,
                                                'metodo': 'Movimiento de términos',
                                                'formula': f'x = {g_str}'
                                            })
                                            print(f"  [+] Encontrado: g(x) = {g_str}")
                    except:
                        pass
        except Exception:
            print("  (error en procesamiento)")
        
        # ESTRATEGIA 3: Crear despejes x = ... moviendo un término
        print("\nEstrategia 3: Reorganizaciones algebraicas...")
        try:
            f_expanded = sp.expand(f_expr)
            terminos = list(sp.Add.make_args(f_expanded))
            
            # Para cada combinación de términos a mover
            if len(terminos) >= 2:
                # Estrategia: mover los términos SIN x a un lado
                terminos_con_x = [t for t in terminos if t.has(x)]
                terminos_sin_x = [t for t in terminos if not t.has(x)]
                
                if terminos_con_x and terminos_sin_x:
                    # f = (términos con x) + (términos sin x) = 0
                    # x = (algo) - (términos sin x) / (coef de términos con x)
                    for tc in terminos_con_x:
                        try:
                            suma_sin_x = sum(terminos_sin_x)
                            sols = solve(tc + suma_sin_x, x)
                            if sols:
                                for sol in sols:
                                    # SOLO aceptar si la solución contiene x (es función de x)
                                    if sol.has(x):
                                        g_str = str(sol)
                                        if len(g_str) < 150:
                                            es_duplicado = any(
                                                str(simplify(d['expr'] - sol)) == '0'
                                                for d in despejes
                                            )
                                            if not es_duplicado:
                                                despejes.append({
                                                    'g_str': g_str,
                                                    'expr': sol,
                                                    'metodo': 'Reorganización algebraica',
                                                    'formula': f'x = {g_str}'
                                                })
                                                print(f"  [+] Encontrado: g(x) = {g_str}")
                        except:
                            pass
        except Exception:
            print("  (error en procesamiento)")
        
        # ESTRATEGIA 4: Despejes especiales para funciones trascendentales
        print("\nEstrategia 4: Funciones especiales (exp, log, trig)...")
        try:
            # Si la función tiene exponencial: e^(f) = g, entonces x = ln(g)/f
            # Si tiene log: ln(f) = g, entonces x = e^g / f
            # Si tiene sin/cos: podemos intentar arcsin, arccos
            
            if f_expr.has(sp.exp):
                # Intenta e^(expr) = 0 tipo funciones
                sols = solve(f_expr, x)
                for sol in sols:
                    # SOLO aceptar si contiene x
                    if sol.has(x):
                        g_str = str(sol)
                        if len(g_str) < 150:
                            es_duplicado = any(
                                str(simplify(d['expr'] - sol)) == '0'
                                for d in despejes
                            )
                            if not es_duplicado:
                                despejes.append({
                                    'g_str': g_str,
                                    'expr': sol,
                                    'metodo': 'Despeje especial (exp)',
                                    'formula': f'x = {g_str}'
                                })
                                print(f"  [+] Encontrado: g(x) = {g_str}")
            
            if f_expr.has(sp.log):
                sols = solve(f_expr, x)
                for sol in sols:
                    # SOLO aceptar si contiene x
                    if sol.has(x):
                        g_str = str(sol)
                        if len(g_str) < 150:
                            es_duplicado = any(
                                str(simplify(d['expr'] - sol)) == '0'
                                for d in despejes
                            )
                            if not es_duplicado:
                                despejes.append({
                                    'g_str': g_str,
                                    'expr': sol,
                                    'metodo': 'Despeje especial (log)',
                                    'formula': f'x = {g_str}'
                                })
                                print(f"  [+] Encontrado: g(x) = {g_str}")
        except Exception:
            print("  (error en procesamiento)")
        
        # ESTRATEGIA 5: Generación manual de despejes para funciones comunes
        print("\nEstrategia 5: Despejes manuales para funciones comunes...")
        try:
            # Si la función es del tipo a*x + b*f(x) + c = 0
            # Podemos generar x = (-b*f(x) - c) / a
            
            f_expanded = sp.expand(f_expr)
            terminos = sp.Add.make_args(f_expanded)
            
            # Encontrar términos lineales en x
            for t in terminos:
                try:
                    derivada_t = sp.diff(t, x)
                    # Si la derivada no contiene x, es potencialmente lineal
                    if not derivada_t.has(x) and derivada_t != 0:
                        # Es lineal en x, podemos despejar
                        otros = sum([t2 for t2 in terminos if t2 != t])
                        sol = -otros / derivada_t
                        sol = simplify(sol)
                        
                        # SOLO aceptar si la solución contiene x
                        if sol.has(x):
                            g_str = str(sol)
                            
                            if len(g_str) < 150:
                                es_duplicado = any(
                                    str(simplify(d['expr'] - sol)) == '0'
                                    for d in despejes
                                )
                                if not es_duplicado:
                                    despejes.append({
                                        'g_str': g_str,
                                        'expr': sol,
                                        'metodo': 'Despeje lineal',
                                        'formula': f'x = {g_str}'
                                    })
                                    print(f"  [+] Encontrado: g(x) = {g_str}")
                except:
                    pass
        except Exception:
            print("  (error en procesamiento)")
        
        # ESTRATEGIA 6: Reorganizaciones alternativas para generar más despejes
        print("\nEstrategia 6: Despejes alternativos por factorización...")
        try:
            f_expanded = sp.expand(f_expr)
            terminos = sp.Add.make_args(f_expanded)
            terminos_con_x = [t for t in terminos if t.has(x)]
            terminos_sin_x = sum([t for t in terminos if not t.has(x)])
            
            # Si hay múltiples términos con x, intentar despejes alternativos
            if len(terminos_con_x) >= 2:
                # Intenta: para cada par de términos con x, despeja uno en términos del otro
                for i, t1 in enumerate(terminos_con_x):
                    for j, t2 in enumerate(terminos_con_x):
                        if i != j:
                            # Intenta despejar x de: t1 + t2 + resto = 0
                            # Como: t1 = -(t2 + resto)
                            # Desde: x = f(x)
                            try:
                                otros_terminos = sum([terminos_con_x[k] for k in range(len(terminos_con_x)) if k != i]) + terminos_sin_x
                                sols = solve(t1 + otros_terminos, x)
                                if sols:
                                    for sol in sols:
                                        if sol.has(x) and len(str(sol)) < 200:
                                            es_duplicado = any(
                                                str(simplify(d['expr'] - sol)) == '0'
                                                for d in despejes
                                            )
                                            if not es_duplicado:
                                                g_str = str(sol)
                                                despejes.append({
                                                    'g_str': g_str,
                                                    'expr': sol,
                                                    'metodo': 'Reorganización alternativa',
                                                    'formula': f'x = {g_str}'
                                                })
                                                print(f"  [+] Encontrado: g(x) = {g_str}")
                            except:
                                pass
        except Exception:
            print("  (error en procesamiento)")
        
        # ESTRATEGIA 7: Despejes por reordenamiento de potencias y productos
        print("\nEstrategia 7: Reordenamiento de potencias...")
        try:
            f_expanded = sp.expand(f_expr)
            terminos = list(sp.Add.make_args(f_expanded))
            
            # Para funciones como x^2 - a, genera x = a/x
            # Para funciones como x^3 - a, genera x = a/x^2
            # etc.
            
            terminos_con_x = [t for t in terminos if t.has(x)]
            terminos_sin_x = [t for t in terminos if not t.has(x)]
            
            if terminos_con_x and terminos_sin_x:
                for t_con_x in terminos_con_x:
                    try:
                        # Obtén el grado de x en este término
                        grado = sp.degree(t_con_x, x)
                        
                        # Si es un monomio en x (como x^2, 2*x^3, etc.)
                        if grado > 0:
                            # Extrae el coeficiente
                            coef = sp.LC(t_con_x, x)  # Leading coefficient
                            
                            # Si hay un término sin x
                            suma_sin_x = sum(terminos_sin_x)
                            
                            # Genera: t_con_x = -suma_sin_x
                            # Si t_con_x = a*x^n, entonces x^n = -suma_sin_x / a
                            # Así que: x = (-suma_sin_x / a)^(1/n)
                            # O alternativamente: x = -suma_sin_x / (a * x^(n-1))
                            
                            # Estrategia: si x^2 = k, genera x = k/x (iterativo)
                            if grado == 2 and coef == 1:
                                # x^2 = -suma_sin_x / coef
                                # x = (-suma_sin_x / coef) / x
                                sol = -suma_sin_x / x
                                sol = simplify(sol)
                                if sol.has(x):
                                    g_str = str(sol)
                                    if len(g_str) < 200:
                                        es_duplicado = any(
                                            str(simplify(d['expr'] - sol)) == '0'
                                            for d in despejes
                                        )
                                        if not es_duplicado:
                                            despejes.append({
                                                'g_str': g_str,
                                                'expr': sol,
                                                'metodo': 'Reordenamiento potencia',
                                                'formula': f'x = {g_str}'
                                            })
                                            print(f"  [+] Encontrado: g(x) = {g_str}")
                            
                            # Estrategia general: x^n = rhs => x = rhs/x^(n-1) si n >= 2
                            elif grado >= 2:
                                # x^grado = -suma_sin_x / coef
                                # x = (-suma_sin_x / coef) / x^(grado-1)
                                rhs = -suma_sin_x / coef
                                divisor = x ** (grado - 1)
                                sol = rhs / divisor
                                sol = simplify(sol)
                                if sol.has(x):
                                    g_str = str(sol)
                                    if len(g_str) < 200:
                                        es_duplicado = any(
                                            str(simplify(d['expr'] - sol)) == '0'
                                            for d in despejes
                                        )
                                        if not es_duplicado:
                                            despejes.append({
                                                'g_str': g_str,
                                                'expr': sol,
                                                'metodo': 'Reordenamiento potencia',
                                                'formula': f'x = {g_str}'
                                            })
                                            print(f"  [+] Encontrado: g(x) = {g_str}")
                    except:
                        pass
        except Exception:
            print("  (error en procesamiento)")
        
        # ESTRATEGIA 8: Despejes por promediación e interpolación
        print("\nEstrategia 8: Transformaciones por promediación...")
        try:
            # Para ecuaciones del tipo x = f(x), a veces podemos usar
            # x = (x + f(x)) / 2 u otras combinaciones que convergen mejor
            
            # Si tenemos despejes candidatos, genera variaciones
            if despejes:
                despejes_nuevos = []
                for d in despejes:
                    try:
                        g_expr = d['expr']
                        
                        # Si g tiene la forma "k/x" o similar, intenta promediación
                        # x = (x + f(x)) / 2
                        g_promedio = (x + g_expr) / 2
                        g_promedio = simplify(g_promedio)
                        
                        if g_promedio.has(x):
                            g_str = str(g_promedio)
                            if len(g_str) < 200:
                                es_duplicado = any(
                                    str(simplify(d2['expr'] - g_promedio)) == '0'
                                    for d2 in despejes + despejes_nuevos
                                )
                                if not es_duplicado:
                                    despejes_nuevos.append({
                                        'g_str': g_str,
                                        'expr': g_promedio,
                                        'metodo': 'Promediación',
                                        'formula': f'x = {g_str}'
                                    })
                                    print(f"  [+] Encontrado: g(x) = {g_str}")
                    except:
                        pass
                
                # Agregar los despejes nuevos a la lista principal
                despejes.extend(despejes_nuevos)
        except Exception:
            print("  (error en procesamiento)")
        
        # ESTRATEGIA 9: Despejes por suma de constantes a ambos lados
        print("\nEstrategia 9: Despejes por equilibrio...")
        try:
            # Para x^2 - 2 = 0, podemos generar:
            # x^2 + c*x = 2 + c*x => x*(x + c) = 2 + c*x => x = (2 + c*x)/(x + c)
            # Buscamos encontrar combinaciones que converjan mejor
            
            f_expanded = sp.expand(f_expr)
            terminos = list(sp.Add.make_args(f_expanded))
            terminos_con_x = [t for t in terminos if t.has(x)]
            
            # Si hay únicamente un término con x (como x^2) y constantes
            if len(terminos_con_x) == 1 and len(terminos) >= 2:
                t_x = terminos_con_x[0]
                suma_const = sum([t for t in terminos if not t.has(x)])
                
                try:
                    grado = sp.degree(t_x, x)
                    
                    # Para x^2 - a, genera x = a/(x+k) para varios k
                    if grado == 2:
                        coef = sp.LC(t_x, x)
                        
                        # Suma constante es -suma_const (porque f = t_x + suma_const = 0)
                        for k_val in [0.5, 1, 1.5]:
                            try:
                                # x*(x + k) = -suma_const/coef
                                # x = (-suma_const/coef) / (x + k)
                                sol = (-suma_const / coef) / (x + k_val)
                                sol = simplify(sol)
                                
                                if sol.has(x):
                                    g_str = str(sol)
                                    if len(g_str) < 200:
                                        es_duplicado = any(
                                            str(simplify(d['expr'] - sol)) == '0'
                                            for d in despejes
                                        )
                                        if not es_duplicado:
                                            despejes.append({
                                                'g_str': g_str,
                                                'expr': sol,
                                                'metodo': 'Equilibrio',
                                                'formula': f'x = {g_str}'
                                            })
                                            print(f"  [+] Encontrado: g(x) = {g_str}")
                            except:
                                pass
                except:
                    pass
        except Exception:
            print("  (error en procesamiento)")
        
        # ESTRATEGIA 10: Despejes para ecuaciones sin término independiente
        print("\nEstrategia 10: Despejes sin término independiente...")
        try:
            f_expanded = sp.expand(f_expr)
            terminos = list(sp.Add.make_args(f_expanded))
            
            # Si NO hay términos sin x (como x**2 = 0 o x**3 - 2*x = 0)
            terminos_sin_x = [t for t in terminos if not t.has(x)]
            
            if not terminos_sin_x:  # No hay constantes
                # Para x^n = 0 o polinomios puros, genera x = x/2, x = x*(1/2), etc.
                terminos_con_x = [t for t in terminos if t.has(x)]
                
                if terminos_con_x:
                    # Intenta: x = x/2, x = x/3, etc.
                    for divisor in [1.5, 2, 2.5]:
                        try:
                            sol = x / divisor
                            g_str = str(sol)
                            
                            es_duplicado = any(
                                str(simplify(d['expr'] - sol)) == '0'
                                for d in despejes
                            )
                            if not es_duplicado:
                                despejes.append({
                                    'g_str': g_str,
                                    'expr': sol,
                                    'metodo': 'Iteración simple',
                                    'formula': f'x = {g_str}'
                                })
                                print(f"  [+] Encontrado: g(x) = {g_str}")
                        except:
                            pass
        except Exception:
            print("  (error en procesamiento)")
        
        # Eliminar duplicados basados en expresión simplificada
        despejes_unicos = []
        expresiones_vistas = []
        for d in despejes:
            expr_simp = simplify(d['expr'])
            es_duplicado = False
            
            try:
                for expr_vista in expresiones_vistas:
                    if simplify(expr_simp - expr_vista) == 0:
                        es_duplicado = True
                        break
            except:
                pass
            
            if not es_duplicado:
                expresiones_vistas.append(expr_simp)
                despejes_unicos.append(d)
        
        print(f"\nTotal encontrados: {len(despejes_unicos)} despejes únicos")
        return despejes_unicos, x, f_expr
        
    except Exception as exc:
        print(f"Error fatal al analizar la función: {exc}")
        return [], None, None





def validar_convergencia(g_str, x_sym, punto_prueba=1.0, intervalo=0.5):
    """
    Valida si g(x) es convergente en un intervalo.
    Criterio: |g'(x)| < 1 en el intervalo [punto_prueba - intervalo, punto_prueba + intervalo]
    
    Também prueba en otros intervalos si el primero no converge.
    
    Retorna: (es_convergente, derivada_str, valor_derivada_en_punto, max_derivada)
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
            
            # Si no converge en [0.5, 1.5], intenta otros intervalo
            if not es_convergente and punto_prueba == 1.0:
                # Intenta en [1, 2] y [2, 3]
                for nuevo_punto in [1.5, 2.0, 2.5]:
                    puntos2 = np.linspace(nuevo_punto - 0.5, nuevo_punto + 0.5, 10)
                    valores_der2 = []
                    
                    for p in puntos2:
                        try:
                            val = g_prima_func(p)
                            if np.isfinite(val):
                                valores_der2.append(abs(val))
                        except:
                            pass
                    
                    if valores_der2 and max(valores_der2) < 1.0:
                        # Encontró convergencia en otro intervalo
                        max_derivada = max(valores_der2)
                        es_convergente = True
                        break
            
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
    opciones_todas = []
    
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
        
        opcion_dict = {
            'indice': idx,
            'g_str': g_str,
            'metodo': despeje['metodo'],
            'derivada': g_prima_str,
            'valor_derivada': valor_derivada,
            'max_derivada': max_derivada,
            'es_convergente': es_convergente
        }
        
        opciones_todas.append(opcion_dict)
        
        if es_convergente:
            print(f"  [V] CONVERGENTE: |g'(x)| < 1 en el intervalo")
            opciones_validas.append(opcion_dict)
        else:
            print(f"  [X] DIVERGENTE: |g'(x)| >= 1 (pero puedes probar de todas formas)")
    
    print(f"\n{'='*70}")
    print(f"Resumen: {len(opciones_validas)} opciones convergentes de {len(despejes)} totales")
    print(f"{'='*70}\n")
    
    # Si no hay convergentes, retorna todas como opciones
    # Así el usuario puede intentar aunque no converjan en [0.5, 1.5]
    if not opciones_validas:
        print("⚠ No hay opciones garantizadas convergentes, pero puedes probar cualquiera.")
        return opciones_todas
    
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
                                
                                x0_input = input("Ingresa valor inicial x0 (ej: 1): ").strip()
                                x0 = float(x0_input) if x0_input else 1.0
                                
                                tol_input = input("Ingresa tolerancia (por defecto 1e-6): ").strip()
                                tol = float(tol_input) if tol_input else 1e-6
                                
                                max_iter_input = input("Ingresa maximo de iteraciones (por defecto 100): ").strip()
                                max_iter = int(max_iter_input) if max_iter_input else 100
                                
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
                        except ValueError as e:
                            print("Ingresa valores numericos validos")
        else:
            print("Opcion no valida. Intenta de nuevo.")


if __name__ == "__main__":
    menu_interactivo()
