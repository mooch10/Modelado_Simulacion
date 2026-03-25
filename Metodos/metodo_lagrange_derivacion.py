import numpy as np
import sympy as sp


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
    """Valida y ordena los puntos por x para evitar inconsistencias."""
    x_arr = np.array(x_vals, dtype=float)
    y_arr = np.array(y_vals, dtype=float)

    if x_arr.size != y_arr.size:
        raise ValueError("x e y deben tener la misma cantidad de datos")

    if x_arr.size < 2:
        raise ValueError("Se requieren al menos 2 puntos")

    if len(np.unique(x_arr)) != x_arr.size:
        raise ValueError("Los valores de x deben ser distintos")

    orden = np.argsort(x_arr)
    return x_arr[orden], y_arr[orden]


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
    n = len(x_vals)
    tabla = np.zeros((n, n), dtype=float)
    tabla[:, 0] = y_vals

    for j in range(1, n):
        for i in range(n - j):
            numerador = tabla[i + 1, j - 1] - tabla[i, j - 1]
            denominador = x_vals[i + j] - x_vals[i]
            tabla[i, j] = numerador / denominador

    return x_vals, tabla


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
    """Formatea expresiones con coma decimal y cantidad fija de decimales."""
    x = sp.Symbol("x")
    expr_ordenada = _ordenar_descendente(expr)

    try:
        pol = sp.Poly(expr_ordenada, x)
        partes = []

        for potencia, coef in sorted(pol.terms(), key=lambda t: t[0][0], reverse=True):
            grado = potencia[0]
            coef_num = float(coef)
            signo = "-" if coef_num < 0 else "+"
            coef_abs = abs(coef_num)
            coef_txt = f"{coef_abs:.{decimales}f}".replace(".", ",")

            if grado == 0:
                termino = f"{coef_txt}"
            elif grado == 1:
                termino = f"{coef_txt}*x"
            else:
                termino = f"{coef_txt}*x**{grado}"

            partes.append((signo, termino))

        if not partes:
            return f"{0:.{decimales}f}".replace(".", ",")

        primer_signo, primer_termino = partes[0]
        salida = ("-" if primer_signo == "-" else "") + primer_termino

        for signo, termino in partes[1:]:
            salida += f" {signo} {termino}"

        return salida

    except sp.PolynomialError:
        expr_num = float(sp.N(expr_ordenada))
        return f"{expr_num:.{decimales}f}".replace(".", ",")


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
            idx = int(np.argmin(np.abs(x_vals - float(x_objetivo))))

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


def _cargar_puntos_interpolacion():
    """Permite cargar datos por lista de y o por funcion f(x)."""
    print("\nCarga de datos para Lagrange:")
    print("1. Ingresar x e y como listas")
    print("2. Ingresar x y una funcion f(x) para calcular y")

    modo = input("Seleccione modo (1-2): ").strip()

    x_vals = _leer_lista_flotantes(
        "Ingrese x separados por coma (admite expresiones como pi/2, e, sqrt(2)): "
    )

    if modo == "2":
        x = sp.Symbol("x")
        funcion = input(
            "Ingrese f(x) (ej: sin(x), exp(x), x**2 + pi, cos(x)+e): "
        ).strip()

        if not funcion:
            raise ValueError("Debe ingresar una funcion f(x)")

        f_expr = _parsear_expresion(funcion, variable=x)
        y_vals = [float(sp.N(f_expr.subs(x, xv))) for xv in x_vals]
    else:
        y_vals = _leer_lista_flotantes(
            "Ingrese y=f(x) separados por coma (admite expresiones numericas como pi, e, sqrt(2)): "
        )

    return normalizar_puntos(x_vals, y_vals)


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
        x_vals, y_vals = _cargar_puntos_interpolacion()
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
        print("5. Reingresar puntos")
        print("6. Volver al menu principal")

        opcion = input("Seleccione una opcion (1-6): ").strip()

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
                    x_eval = float(input("Ingrese x*: "))
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
                    x_obj = float(x_txt)

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
                    x_eval = float(input("Ingrese x*: "))
                    x = sp.Symbol("x")
                    y_eval = float(sp.N(p_newton.subs(x, x_eval)))
                    print(f"P_N({x_eval}) = {y_eval}")
            except Exception as e:
                print(f"Error: {e}")

        elif opcion == "5":
            try:
                x_vals, y_vals = _cargar_puntos_interpolacion()
                print("Puntos actualizados correctamente")
            except ValueError as e:
                print(f"Error de datos: {e}")

        elif opcion == "6":
            break

        else:
            print("Opcion no valida")
