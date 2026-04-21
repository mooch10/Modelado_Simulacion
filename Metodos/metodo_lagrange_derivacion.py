import numpy as np
import sympy as sp
import math

"""
╔════════════════════════════════════════════════════════════════╗
║         MACHETE: INTERPOLACIÓN POLINOMIAL LAGRANGE              ║
╚════════════════════════════════════════════════════════════════╝

DEFINICIÓN:
Técnica que construye un polinomio de grado n que pasa por n+1
puntos dados, permitiendo estimar valores intermedios.

UTILIDAD:
• Interpolar valores entre puntos conocidos.
• Aproximar derivadas numéricamente.
• Reconstruir funciones a partir de datos discretos.

PASOS:
1. Obtener n+1 puntos (x_i, y_i) distintos en x
2. Construir bases de Lagrange L_i(x)
3. Polinomio: P(x) = Σ y_i · L_i(x)
4. Evaluar P(x) en puntos deseados
5. Derivar P(x) para obtener velocidades

FÓRMULA (Base Lagrange):
L_i(x) = ∏_{j≠i} (x - x_j) / (x_i - x_j)
P(x) = Σ_{i=0}^n y_i · L_i(x)

REQUISITOS:
• Mínimo 2 puntos
• Puntos con x distintos
• Valores y finitos
• Se asume continuidad entre puntos
"""

def _locals_simbolicos():
    """Diccionario de funciones/constantes permitidas para sympify."""
    return {
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


def _parsear_expresion(texto, variable=None):
    """Parsea una expresion con soporte de e, pi y funciones trigonométricas."""
    expr = sp.sympify(texto, locals=_locals_simbolicos())

    if variable is None:
        if expr.free_symbols:
            raise ValueError("La expresion no debe contener variables")
    else:
        simbolos_invalidos = expr.free_symbols - {variable}
        if simbolos_invalidos:
            raise ValueError(f"Se encontraron variables no permitidas: {simbolos_invalidos}")

    return expr


def normalizar_puntos(x_vals, y_vals):
    """Valida y ordena puntos preservando forma simbólica (pi, e, etc.)."""
    if len(x_vals) != len(y_vals):
        raise ValueError("x e y deben tener la misma cantidad de datos")

    if len(x_vals) < 2:
        raise ValueError("Se requieren al menos 2 puntos")

    x_sym = [sp.nsimplify(v, [sp.pi, sp.E], rational=True) for v in x_vals]
    y_sym = [sp.nsimplify(v, [sp.pi, sp.E], rational=True) for v in y_vals]

    if len(set(map(str, x_sym))) != len(x_sym):
        raise ValueError("Los valores de x deben ser distintos")

    orden = sorted(range(len(x_sym)), key=lambda i: float(sp.N(x_sym[i])))
    x_ord = np.array([x_sym[i] for i in orden], dtype=object)
    y_ord = np.array([y_sym[i] for i in orden], dtype=object)
    return x_ord, y_ord


def bases_lagrange(x_vals):
    """Construye las bases de Lagrange L_i(x)."""
    x = sp.Symbol("x")
    n = len(x_vals)
    bases = []

    for i in range(n):
        li = 1
        for j in range(n):
            if i != j:
                li *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
        bases.append(sp.simplify(li))

    return bases


def polinomio_lagrange(x_vals, y_vals):
    """Retorna el polinomio interpolante de Lagrange."""
    x_vals, y_vals = normalizar_puntos(x_vals, y_vals)
    bases = bases_lagrange(x_vals)

    p = 0
    for yi, li in zip(y_vals, bases):
        p += yi * li

    return sp.expand(p), bases


def tabla_diferencias_divididas(x_vals, y_vals):
    """Calcula la tabla de diferencias divididas de Newton."""
    x_vals, y_vals = normalizar_puntos(x_vals, y_vals)
    x_num = np.array([float(sp.N(v)) for v in x_vals], dtype=float)
    y_num = np.array([float(sp.N(v)) for v in y_vals], dtype=float)
    n = len(x_vals)
    tabla = np.zeros((n, n), dtype=float)
    tabla[:, 0] = y_num

    for j in range(1, n):
        for i in range(n - j):
            numerador = tabla[i + 1, j - 1] - tabla[i, j - 1]
            denominador = x_num[i + j] - x_num[i]
            tabla[i, j] = numerador / denominador

    return x_num, tabla


def polinomio_newton_desde_dd(x_vals, tabla_dd):
    """Construye el polinomio de Newton usando la diagonal superior de DD."""
    x = sp.Symbol("x")
    n = len(x_vals)

    p = tabla_dd[0, 0]
    termino = 1

    for j in range(1, n):
        termino *= (x - x_vals[j - 1])
        p += tabla_dd[0, j] * termino

    return sp.expand(p)


def _ordenar_descendente(expr):
    """Retorna una expresion polinomica expandida y ordenada por grado descendente."""
    x = sp.Symbol("x")
    expr_expandida = sp.expand(expr)

    try:
        pol = sp.Poly(expr_expandida, x)
        return pol.as_expr()
    except sp.PolynomialError:
        return expr_expandida


def _expr_a_texto_decimal(expr, decimales=7):
    """Formatea expresiones: preserva pi simbólico y, si no aplica, usa decimales con coma."""
    def _formatear_decimal(valor):
        txt = f"{valor:.{decimales}f}"
        txt = txt.rstrip("0").rstrip(".")
        if txt == "-0":
            txt = "0"
        return txt.replace(".", ",")

    x = sp.Symbol("x")
    expr_ordenada = _ordenar_descendente(expr)

    # Si se puede reconstruir una forma con pi de manera estable, priorizar salida simbólica.
    try:
        expr_pi = sp.nsimplify(expr_ordenada, [sp.pi], rational=True)
        if expr_pi.has(sp.pi):
            return str(_ordenar_descendente(sp.expand(expr_pi)))
    except Exception:
        pass

    try:
        pol = sp.Poly(expr_ordenada, x)
        partes = []

        for potencia, coef in sorted(pol.terms(), key=lambda t: t[0][0], reverse=True):
            grado = potencia[0]
            coef_num = float(coef)
            signo = "-" if coef_num < 0 else "+"
            coef_abs = abs(coef_num)
            coef_txt = _formatear_decimal(coef_abs)

            if grado == 0:
                termino = f"{coef_txt}"
            elif grado == 1:
                termino = f"{coef_txt}*x"
            else:
                termino = f"{coef_txt}*x**{grado}"

            partes.append((signo, termino))

        if not partes:
            return "0"

        primer_signo, primer_termino = partes[0]
        salida = ("-" if primer_signo == "-" else "") + primer_termino

        for signo, termino in partes[1:]:
            salida += f" {signo} {termino}"

        return salida

    except sp.PolynomialError:
        expr_num = float(sp.N(expr_ordenada))
        return _formatear_decimal(expr_num)


def _formato_suma_lagrange(y_vals, bases_ordenadas):
    """Construye la representacion P(x)=sum(y_i*L_i(x)) en texto legible."""
    terminos = []
    for i, yi in enumerate(y_vals):
        yi_txt = _expr_a_texto_decimal(yi)
        li_txt = _expr_a_texto_decimal(bases_ordenadas[i])
        terminos.append(f"({yi_txt})*({li_txt})")
    return " + ".join(terminos)


def derivada_lagrange_local(x_sub, y_sub, x_eval):
    """Aproxima la derivada usando polinomio local de Lagrange con 3 puntos."""
    x = sp.Symbol("x")
    p_local, _ = polinomio_lagrange(x_sub, y_sub)
    dp = sp.diff(p_local, x)
    valor = float(dp.subs(x, float(x_eval)))
    return valor, sp.expand(p_local), sp.expand(dp)


def aproximar_derivada_tres_formas(x_vals, y_vals, forma, x_objetivo=None):
    """
    Aproxima f'(x) con 3 puntos por Lagrange en tres formas:
    - adelante: usa los primeros 3 puntos y evalua en x0
    - atras: usa los ultimos 3 puntos y evalua en xn
    - centrada: usa 3 puntos alrededor de un x interior
    """
    x_vals, y_vals = normalizar_puntos(x_vals, y_vals)
    x_num = np.array([float(sp.N(v)) for v in x_vals], dtype=float)

    if len(x_vals) < 3:
        raise ValueError("Se requieren al menos 3 puntos para derivar por aproximacion")

    forma = forma.strip().lower()

    if forma == "adelante":
        x_sub = x_vals[:3]
        y_sub = y_vals[:3]
        x_eval = x_sub[0]

    elif forma == "atras":
        x_sub = x_vals[-3:]
        y_sub = y_vals[-3:]
        x_eval = x_sub[-1]

    elif forma == "centrada":
        if len(x_vals) < 3:
            raise ValueError("No hay suficientes puntos para forma centrada")

        if x_objetivo is None:
            idx = len(x_vals) // 2
        else:
            idx = int(np.argmin(np.abs(x_num - float(x_objetivo))))

        if idx == 0:
            idx = 1
        if idx == len(x_vals) - 1:
            idx = len(x_vals) - 2

        x_sub = x_vals[idx - 1: idx + 2]
        y_sub = y_vals[idx - 1: idx + 2]
        x_eval = x_sub[1]

    else:
        raise ValueError("Forma no valida. Use: adelante, atras o centrada")

    derivada, p_local, dp_local = derivada_lagrange_local(x_sub, y_sub, x_eval)

    return {
        "forma": forma,
        "x_evaluacion": float(x_eval),
        "x_sub": x_sub,
        "y_sub": y_sub,
        "derivada": derivada,
        "polinomio_local": p_local,
        "derivada_polinomio_local": dp_local,
    }


def _leer_lista_flotantes(mensaje):
    texto = input(mensaje).strip()
    partes = [p.strip() for p in texto.split(",") if p.strip()]
    valores = []

    for p in partes:
        expr = _parsear_expresion(p)
        valores.append(float(sp.N(expr)))

    return valores


def _leer_lista_expresiones(mensaje):
    texto = input(mensaje).strip()
    partes = [p.strip() for p in texto.split(",") if p.strip()]
    valores = []
    for p in partes:
        valores.append(_parsear_expresion(p))
    return valores


def _cargar_puntos_interpolacion():
    """Permite cargar datos por lista de y o por funcion f(x)."""
    print("\nCarga de datos para Lagrange:")
    print("1. Ingresar x e y como listas")
    print("2. Ingresar x y una funcion f(x) para calcular y")

    modo = input("Seleccione modo (1-2): ").strip()

    x_vals = _leer_lista_expresiones(
        "Ingrese x separados por coma (admite expresiones como pi/2, e, sqrt(2), cbrt(8)): "
    )
    funcion_referencia = None

    if modo == "2":
        x = sp.Symbol("x")
        funcion = input(
            "Ingrese f(x) (ej: sin(x), exp(x), x**2 + pi, cos(x)+e): "
        ).strip()

        if not funcion:
            raise ValueError("Debe ingresar una funcion f(x)")

        f_expr = _parsear_expresion(funcion, variable=x)
        funcion_referencia = f_expr
        y_vals = [sp.simplify(f_expr.subs(x, xv)) for xv in x_vals]
    else:
        y_vals = _leer_lista_expresiones(
            "Ingrese y=f(x) separados por coma (admite expresiones numericas como pi, e, sqrt(2), cbrt(8)): "
        )

    x_norm, y_norm = normalizar_puntos(x_vals, y_vals)
    return x_norm, y_norm, funcion_referencia


def _comparar_error_local(x_vals, y_vals, funcion_referencia=None):
    """Compara el error local de interpolacion en un punto x* especifico."""
    x = sp.Symbol("x")
    p_lagrange, _ = polinomio_lagrange(x_vals, y_vals)
    p_lagrange = _ordenar_descendente(p_lagrange)

    x_eval = float(_leer_lista_flotantes("Ingrese el punto x* (ej: pi/3): ")[0])
    p_eval = float(sp.N(p_lagrange.subs(x, x_eval)))

    print("\n" + "=" * 90)
    print("COMPARACION DE ERROR LOCAL".center(90))
    print("=" * 90)

    print(f"x* = {str(sp.N(x_eval, 12)).replace('.', ',')}")
    print(f"P(x*) = {str(sp.N(p_eval, 12)).replace('.', ',')}")

    valor_real = None
    fuente = ""

    if funcion_referencia is not None:
        valor_real = float(sp.N(funcion_referencia.subs(x, x_eval)))
        fuente = f"f(x) = {funcion_referencia}"
    else:
        usar_manual = input(
            "No hay funcion exacta cargada. ¿Desea ingresar f(x*) manualmente? (s/n): "
        ).strip().lower()
        if usar_manual in ["s", "si", "sí"]:
            valor_real = float(_leer_lista_flotantes("Ingrese f(x*) (admite expresion): ")[0])
            fuente = "valor manual"

    if valor_real is None:
        print("No se puede calcular error sin un valor real de referencia.")
        print("=" * 90)
        return

    error_abs = abs(valor_real - p_eval)
    error_rel = (error_abs / abs(valor_real)) if abs(valor_real) > 1e-15 else float("nan")

    print(f"Valor real ({fuente}) = {str(sp.N(valor_real, 12)).replace('.', ',')}")
    print(f"Error absoluto = {str(sp.N(error_abs, 12)).replace('.', ',')}")
    if np.isnan(error_rel):
        print("Error relativo = no definido (valor real cercano a 0)")
    else:
        print(f"Error relativo = {str(sp.N(error_rel, 12)).replace('.', ',')}")
    print("=" * 90)


def _max_abs_polinomio_en_intervalo(poly_expr, a, b):
    """Calcula max|poly(x)| en [a,b] usando puntos críticos del polinomio."""
    x = sp.Symbol("x")
    poly_expr = sp.expand(poly_expr)
    dpoly = sp.diff(poly_expr, x)

    candidatos = [float(a), float(b)]

    if dpoly != 0:
        try:
            for r in sp.nroots(dpoly):
                if abs(sp.im(r)) < 1e-10:
                    rr = float(sp.re(r))
                    if float(a) <= rr <= float(b):
                        candidatos.append(rr)
        except Exception:
            pass

    f_poly = sp.lambdify(x, sp.Abs(poly_expr), "numpy")
    vals = np.array([float(f_poly(c)) for c in candidatos], dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        raise ValueError("No se pudo evaluar |w(x)| en el intervalo")

    return float(np.max(vals))


def _cota_error_global_teorica(x_vals, funcion_expr, a, b, muestras=4000, m_manual=None):
    """
    Calcula una cota teorica global del error de interpolacion de Lagrange:
    ||f - P_n||_inf <= M/(n+1)! * max|w(x)|,  w(x)=prod(x-x_i)
    usando maximos aproximados por muestreo en [a,b].
    """
    x = sp.Symbol("x")
    n = len(x_vals) - 1

    if b <= a:
        raise ValueError("El intervalo debe cumplir a < b")

    deriv_orden = sp.diff(funcion_expr, x, n + 1) if funcion_expr is not None else None
    w_expr = sp.Integer(1)
    for xi in x_vals:
        w_expr *= (x - float(xi))
    w_expr = sp.expand(w_expr)

    if m_manual is not None:
        m_aprox = abs(float(m_manual))
        fuente_m = "manual"
    else:
        if deriv_orden is None:
            raise ValueError("Debe ingresar f(x) o un valor manual para M")
        xs = np.linspace(float(a), float(b), int(muestras))
        d_fun = sp.lambdify(x, sp.Abs(deriv_orden), "numpy")
        d_vals = np.array(d_fun(xs), dtype=float)
        d_vals = d_vals[np.isfinite(d_vals)]
        if d_vals.size == 0:
            raise ValueError("No se pudo evaluar f^(n+1) en el intervalo indicado")
        m_aprox = float(np.max(d_vals))
        fuente_m = "automatica (muestreo)"

    w_max_aprox = _max_abs_polinomio_en_intervalo(w_expr, a, b)
    cota = (m_aprox * w_max_aprox) / math.factorial(n + 1)

    return {
        "n": n,
        "derivada_orden": deriv_orden,
        "w_expr": w_expr,
        "M_aprox": m_aprox,
        "fuente_M": fuente_m,
        "Wmax_aprox": w_max_aprox,
        "cota_global": cota,
        "intervalo": (float(a), float(b)),
    }


def _error_real_maximo_intervalo(x_vals, y_vals, funcion_expr, a, b, muestras=4000):
    """Estima el error real máximo |f(x)-P_n(x)| en [a,b] por muestreo denso."""
    x = sp.Symbol("x")
    p_lagrange, _ = polinomio_lagrange(x_vals, y_vals)

    xs = np.linspace(float(a), float(b), int(muestras))
    f_fun = sp.lambdify(x, funcion_expr, "numpy")
    p_fun = sp.lambdify(x, p_lagrange, "numpy")

    f_vals = np.array(f_fun(xs), dtype=float)
    p_vals = np.array(p_fun(xs), dtype=float)
    err_vals = np.abs(f_vals - p_vals)
    err_vals = err_vals[np.isfinite(err_vals)]

    if err_vals.size == 0:
        raise ValueError("No se pudo estimar el error real en el intervalo")

    return float(np.max(err_vals))


def _mostrar_cota_error_global(x_vals, y_vals, funcion_referencia=None):
    """Interfaz para calcular y mostrar la cota de error global (teórica)."""
    x = sp.Symbol("x")

    modo_m = input(
        "¿Cómo desea obtener M=max|f^(n+1)|? 1) Automático con f(x)  2) Manual: "
    ).strip()

    m_manual = None

    if modo_m == "2":
        m_manual = float(_leer_lista_flotantes("Ingrese M (admite expresion): ")[0])
    else:
        if funcion_referencia is None:
            func_txt = input(
                "No hay funcion exacta cargada. Ingrese f(x) para cota teorica: "
            ).strip()
            if not func_txt:
                print("Se requiere f(x) para calcular la cota teorica")
                return
            funcion_referencia = _parsear_expresion(func_txt, variable=x)

    print("\nIntervalo para cota global [a,b].")
    print("Si presiona Enter, se usa [min(x_i), max(x_i)].")

    a_txt = input("a (Enter por defecto): ").strip()
    b_txt = input("b (Enter por defecto): ").strip()

    if a_txt:
        a = float(sp.N(_parsear_expresion(a_txt)))
    else:
        a = float(min(float(sp.N(v)) for v in x_vals))

    if b_txt:
        b = float(sp.N(_parsear_expresion(b_txt)))
    else:
        b = float(max(float(sp.N(v)) for v in x_vals))

    resultado = _cota_error_global_teorica(x_vals, funcion_referencia, a, b, m_manual=m_manual)

    print("\n" + "=" * 95)
    print("COTA DE ERROR GLOBAL TEORICA (LAGRANGE)".center(95))
    print("=" * 95)
    print(f"Grado del interpolante n = {resultado['n']}")
    if resultado["derivada_orden"] is not None:
        print(f"f^(n+1)(x) = {resultado['derivada_orden']}")
    else:
        print("f^(n+1)(x): no ingresada (se usa M manual)")
    print(f"w(x) = {resultado['w_expr']}")
    print(
        f"Intervalo: [{_expr_a_texto_decimal(resultado['intervalo'][0])}, "
        f"{_expr_a_texto_decimal(resultado['intervalo'][1])}]"
    )
    print(
        f"M ({resultado['fuente_M']}) = max|f^(n+1)(x)| = "
        f"{_expr_a_texto_decimal(resultado['M_aprox'])}"
    )
    print(f"Wmax (criticos de w) = max|w(x)| = {_expr_a_texto_decimal(resultado['Wmax_aprox'])}")
    print(f"Cota global <= {_expr_a_texto_decimal(resultado['cota_global'])}")
    print("Nota: si M es automático, se estima por muestreo; con M manual la cota es más controlable.")
    
    if funcion_referencia is not None:
        error_real = _error_real_maximo_intervalo(
            x_vals,
            y_vals,
            funcion_referencia,
            a,
            b,
            muestras=4000,
        )

        razon = (resultado["cota_global"] / error_real) if error_real > 1e-15 else float("inf")

        print("-" * 95)
        print("COMPARACION EN PARALELO: COTA TEORICA VS ERROR REAL".center(95))
        print("-" * 95)
        print(f"Error real máximo (numérico) = {_expr_a_texto_decimal(error_real)}")
        print(f"Cota teórica global         = {_expr_a_texto_decimal(resultado['cota_global'])}")
        if np.isfinite(razon):
            print(f"Relación cota/error real    = {_expr_a_texto_decimal(razon)}")
        else:
            print("Relación cota/error real    = infinito (error real ~ 0)")

    print("=" * 95)


def _imprimir_tabla_dd(x_vals, tabla):
    n = len(x_vals)
    encabezado = ["i", "x_i", "f[x_i]"] + [f"DD orden {k}" for k in range(1, n)]

    print("\n" + "=" * 110)
    print("TABLA DE DIFERENCIAS DIVIDIDAS".center(110))
    print("=" * 110)
    print(" | ".join(f"{h:^15}" for h in encabezado))
    print("-" * 110)

    for i in range(n):
        fila = [f"{i:^15}", f"{x_vals[i]:^15.8g}"]
        for j in range(n):
            if i <= n - j - 1:
                fila.append(f"{tabla[i, j]:^15.8g}")
            else:
                fila.append(f"{'':^15}")
        print(" | ".join(fila))

    print("=" * 110)


def ejecutar_metodo_lagrange():
    """Interfaz de consola para Lagrange + derivacion + diferencias divididas."""
    print("\n" + "=" * 90)
    print("MODULO DE INTERPOLACION Y DERIVACION NUMERICA (LAGRANGE + DD)".center(90))
    print("=" * 90)

    try:
        x_vals, y_vals, funcion_referencia = _cargar_puntos_interpolacion()
    except ValueError as e:
        print(f"Error de datos: {e}")
        return

    while True:
        print("\n" + "=" * 90)
        print("SUBMENU LAGRANGE / DERIVACION / DIFERENCIAS DIVIDIDAS")
        print("=" * 90)
        print("1. Mostrar bases de Lagrange L_i(x)")
        print("2. Construir polinomio interpolante de Lagrange")
        print("3. Derivar por aproximacion (adelante, atras, centrada)")
        print("4. Diferencias divididas y polinomio de Newton")
        print("5. Comparar error local en un punto x*")
        print("6. Calcular cota de error global teorica")
        print("7. Reingresar puntos")
        print("8. Volver al menu principal")

        opcion = input("Seleccione una opcion (1-8): ").strip()

        if opcion == "1":
            try:
                _, bases = polinomio_lagrange(x_vals, y_vals)
                bases_ordenadas = [_ordenar_descendente(li) for li in bases]

                print("\nBases de Lagrange:")
                for i, li in enumerate(bases_ordenadas):
                    print(f"L_{i}(x) = {_expr_a_texto_decimal(li)}")

                print("\nSuma de polinomios (forma de interpolacion):")
                print(f"P(x) = {_formato_suma_lagrange(y_vals, bases_ordenadas)}")
            except Exception as e:
                print(f"Error: {e}")

        elif opcion == "2":
            try:
                p, _ = polinomio_lagrange(x_vals, y_vals)
                p = _ordenar_descendente(p)
                print("\nPolinomio interpolante de Lagrange:")
                print(f"P(x) = {_expr_a_texto_decimal(p)}")

                evaluar = input("¿Desea evaluar P(x) en un punto? (s/n): ").strip().lower()
                if evaluar in ["s", "si", "sí"]:
                    x_eval = _leer_lista_flotantes("Ingrese x* (admite pi/2, e, sqrt(2), etc.): ")[0]
                    x = sp.Symbol("x")
                    y_eval = float(sp.N(p.subs(x, x_eval)))
                    print(f"P({x_eval}) = {y_eval}")
            except Exception as e:
                print(f"Error: {e}")

        elif opcion == "3":
            print("\nFormas disponibles: adelante, atras, centrada")
            forma = input("Ingrese forma: ").strip().lower()
            x_obj = None

            if forma == "centrada":
                x_txt = input("x objetivo interior (Enter para usar el central): ").strip()
                if x_txt:
                    x_obj = float(sp.N(_parsear_expresion(x_txt)))

            try:
                resultado = aproximar_derivada_tres_formas(x_vals, y_vals, forma, x_obj)
                p_local = _ordenar_descendente(resultado['polinomio_local'])
                dp_local = _ordenar_descendente(resultado['derivada_polinomio_local'])
                print("\nResultado de derivacion aproximada:")
                print(f"Forma: {resultado['forma']}")
                print(f"Puntos usados x: {resultado['x_sub']}")
                print(f"Puntos usados y: {resultado['y_sub']}")
                print(f"Polinomio local: {_expr_a_texto_decimal(p_local)}")
                print(f"d/dx polinomio local: {_expr_a_texto_decimal(dp_local)}")
                print(f"f'({resultado['x_evaluacion']}) ≈ {resultado['derivada']}")

                x = sp.Symbol("x")
                if funcion_referencia is not None:
                    derivada_real_expr = sp.diff(funcion_referencia, x)
                    derivada_real = float(sp.N(derivada_real_expr.subs(x, resultado["x_evaluacion"])))
                    error_abs = abs(resultado["derivada"] - derivada_real)
                    print(f"f'({resultado['x_evaluacion']}) real = {derivada_real}")
                    print(f"Error |derivada aproximada - derivada real| = {error_abs}")
                else:
                    print("Error |derivada aproximada - derivada real| = no disponible (sin f(x) exacta)")
            except Exception as e:
                print(f"Error: {e}")

        elif opcion == "4":
            try:
                x_dd, tabla = tabla_diferencias_divididas(x_vals, y_vals)
                _imprimir_tabla_dd(x_dd, tabla)

                p_newton = polinomio_newton_desde_dd(x_dd, tabla)
                p_newton = _ordenar_descendente(p_newton)
                print("\nPolinomio de Newton (desde DD):")
                print(f"P_N(x) = {_expr_a_texto_decimal(p_newton)}")

                evaluar = input("¿Desea evaluar P_N(x) en un punto? (s/n): ").strip().lower()
                if evaluar in ["s", "si", "sí"]:
                    x_eval = _leer_lista_flotantes("Ingrese x* (admite pi/2, e, sqrt(2), etc.): ")[0]
                    x = sp.Symbol("x")
                    y_eval = float(sp.N(p_newton.subs(x, x_eval)))
                    print(f"P_N({x_eval}) = {y_eval}")
            except Exception as e:
                print(f"Error: {e}")

        elif opcion == "5":
            try:
                _comparar_error_local(x_vals, y_vals, funcion_referencia)
            except Exception as e:
                print(f"Error: {e}")

        elif opcion == "6":
            try:
                _mostrar_cota_error_global(x_vals, y_vals, funcion_referencia)
            except Exception as e:
                print(f"Error: {e}")

        elif opcion == "7":
            try:
                x_vals, y_vals, funcion_referencia = _cargar_puntos_interpolacion()
                print("Puntos actualizados correctamente")
            except ValueError as e:
                print(f"Error de datos: {e}")

        elif opcion == "8":
            break

        else:
            print("Opcion no valida")
