import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from Metodos.input_parser import parse_real, parse_real_or_default, parse_int_or_default

from Metodos.metodo_newton_raphson import (
    metodo_newton_raphson,
    mostrar_tabla_iteraciones as mostrar_tabla_newton,
    graficar_funcion as graficar_newton,
)
from Metodos.metodo_aitken import (
    metodo_aitken,
    mostrar_tabla_iteraciones as mostrar_tabla_aitken,
    graficar_metodo_aitken as graficar_aitken,
)
from Metodos.metodo_de_biseccion import (
    metodo_biseccion,
    graficar_funcion as graficar_biseccion,
)
from Metodos.metodo_punto_fijo import (
    metodo_punto_fijo,
    graficar_funcion as graficar_punto_fijo,
)
from Metodos.metodo_lagrange_derivacion import ejecutar_metodo_lagrange
from Metodos.metodo_integracion_numerica import ejecutar_integracion_numerica
from Metodos.metodo_ajuste_curvas import ejecutar_ajuste_curvas
from Metodos.metodo_sistemas_lineales import ejecutar_sistemas_lineales
from Metodos.metodo_edo import ejecutar_edo
from Metodos.metodo_red_neuronal_descenso_gradiente import ejecutar_red_neuronal_descenso_gradiente


def comparativa_metodos():
    """Ejecuta los 4 métodos con la misma función y compara resultados."""
    print("\n" + "=" * 100)
    print("COMPARATIVA DE LOS 4 MÉTODOS".center(100))
    print("=" * 100)

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
        x0 = parse_real(input("3. Aproximación inicial x₀: "), "x₀")
    except ValueError:
        print("Error: x₀ debe ser un número")
        return

    try:
        a = parse_real(input("4. Límite inferior a (para Bisección): "), "a")
        b = parse_real(input("5. Límite superior b (para Bisección): "), "b")
    except ValueError:
        print("Error: Los límites deben ser números")
        return

    try:
        tol_input = input("6. Tolerancia de error (default 1e-6): ").strip()
        tolerancia = parse_real_or_default(tol_input, 1e-6, "tolerancia")
    except ValueError:
        print("Error: Tolerancia debe ser un número")
        return

    try:
        iter_input = input("7. Máximo de iteraciones (default 100): ").strip()
        max_iteraciones = parse_int_or_default(iter_input, 100, "máximo de iteraciones")
    except ValueError:
        print("Error: Máximo de iteraciones debe ser un entero")
        return

    resultados = {}

    print("\n" + "-" * 100)
    print("Ejecutando Método de Newton-Raphson...")
    print("-" * 100)
    try:
        raiz_nr, iter_nr, conv_nr = metodo_newton_raphson(funcion, x0, tolerancia, max_iteraciones)
        if raiz_nr is not None:
            resultados["Newton-Raphson"] = {
                "raiz": raiz_nr,
                "iteraciones": len(iter_nr),
                "error": iter_nr[-1]["Error |x_(n+1) - x_n|"] if iter_nr else 0,
                "convergencia": conv_nr,
            }
    except Exception as e:
        print(f"Error en Newton-Raphson: {e}")

    print("\n" + "-" * 100)
    print("Ejecutando Método de Aceleración de Aitken...")
    print("-" * 100)
    try:
        raiz_aitken, iter_aitken, conv_aitken = metodo_aitken(g_iteracion, x0, tolerancia, max_iteraciones)
        if raiz_aitken is not None:
            resultados["Aitken"] = {
                "raiz": raiz_aitken,
                "iteraciones": len(iter_aitken),
                "error": iter_aitken[-1]["Error"] if iter_aitken else 0,
                "convergencia": conv_aitken,
            }
    except Exception as e:
        print(f"Error en Aitken: {e}")

    print("\n" + "-" * 100)
    print("Ejecutando Método de Bisección...")
    print("-" * 100)
    try:
        raiz_bis, iter_bis = metodo_biseccion(funcion, a, b, tolerancia, max_iteraciones)
        if raiz_bis is not None:
            resultados["Bisección"] = {
                "raiz": raiz_bis,
                "iteraciones": len(iter_bis),
                "error": abs(iter_bis[-1][6]) if iter_bis else 0,
                "convergencia": True,
            }
    except Exception as e:
        print(f"Error en Bisección: {e}")

    print("\n" + "-" * 100)
    print("Ejecutando Método de Punto Fijo...")
    print("-" * 100)
    try:
        raiz_pf, iter_pf = metodo_punto_fijo(g_iteracion, x0, tolerancia, max_iteraciones)
        if raiz_pf is not None:
            resultados["Punto Fijo"] = {
                "raiz": raiz_pf,
                "iteraciones": len(iter_pf),
                "error": iter_pf[-1][3] if iter_pf else 0,
                "convergencia": True,
            }
    except Exception as e:
        print(f"Error en Punto Fijo: {e}")

    print("\n\n" + "=" * 130)
    print("TABLA COMPARATIVA DE RESULTADOS".center(130))
    print("=" * 130)
    print(f"{'Método':^25} {'Raíz encontrada':^30} {'Iteraciones':^15} {'Error final':^25} {'Convergencia':^15}")
    print("-" * 130)

    for metodo, datos in resultados.items():
        raiz = datos["raiz"]
        iteraciones = datos["iteraciones"]
        error = datos["error"]
        convergencia = "✓ Sí" if datos["convergencia"] else "✗ No"
        print(f"{metodo:^25} {raiz:^30.15e} {iteraciones:^15} {error:^25.10e} {convergencia:^15}")

    print("=" * 130)

    if resultados:
        print("\n" + "=" * 130)
        print("ANÁLISIS COMPARATIVO".center(130))
        print("=" * 130)

        raices = [datos["raiz"] for datos in resultados.values()]
        raiz_promedio = np.mean(raices)
        print(f"\nRaíz promedio: {raiz_promedio:.15e}")

        metodo_rapido = min(resultados.items(), key=lambda x: x[1]["iteraciones"])
        print(f"Método más rápido (menos iteraciones): {metodo_rapido[0]} con {metodo_rapido[1]['iteraciones']} iteraciones")

        metodo_preciso = min(resultados.items(), key=lambda x: x[1]["error"])
        print(f"Método más preciso (menor error): {metodo_preciso[0]} con error de {metodo_preciso[1]['error']:.10e}")

        if len(raices) > 1:
            divergencia_max = max(raices) - min(raices)
            print(f"Divergencia máxima entre raíces: {divergencia_max:.10e}")

        print("=" * 130 + "\n")


def sistema_lineal_2d(fx=None, fy=None, A=None, eq_point=(0.0, 0.0), x0=(1.0, 0.0), t_max=10.0, num_points=400, grid_limits=((-5, 5), (-5, 5)), show_plot=True, symbolic=False):
    """Analiza y grafica un sistema dinámico lineal 2D.

    Parámetros:
    - fx, fy: funciones f(x,y) y g(x,y) o None si se proporciona A
    - A: matriz 2x2 (lista/array) del sistema lineal; si se pasa se usa directamente
    - eq_point: punto de equilibrio donde evaluar la jacobiana (por defecto (0,0))
    - x0: condición inicial (x0, y0) para mostrar la solución analítica
    - t_max, num_points: rango temporal para las trayectorias
    - grid_limits: ((xmin,xmax),(ymin,ymax)) para el quiver y líneas
    - show_plot: si True muestra las gráficas

    Retorna un diccionario con la matriz A, autovalores (formateados), autovectores, tipo y la solución analítica.
    """
    if A is None:
        if fx is None or fy is None:
            raise ValueError("Debe pasar A o las funciones fx y fy")
        h = 1e-6
        x_eq, y_eq = eq_point

        def pdx(f):
            return (f(x_eq + h, y_eq) - f(x_eq - h, y_eq)) / (2 * h)

        def pdy(f):
            return (f(x_eq, y_eq + h) - f(x_eq, y_eq - h)) / (2 * h)

        a = pdx(fx)
        b = pdy(fx)
        c = pdx(fy)
        d = pdy(fy)
        A = np.array([[a, b], [c, d]], dtype=float)
    else:
        A = np.array(A, dtype=float).reshape(2, 2)

    eigvals, eigvecs = np.linalg.eig(A)
    # Limpiar valores numéricos muy pequeños hacia cero para presentación
    tol_eig = 1e-12
    eigvals = np.array([complex(0, 0) if (abs(np.real(v)) < tol_eig and abs(np.imag(v)) < tol_eig) else v for v in eigvals])

    def fmt_lam(l):
        alpha = float(np.real(l))
        beta = float(np.imag(l))
        # Considerar como cero si son muy pequeños
        if abs(alpha) < tol_eig:
            alpha = 0.0
        if abs(beta) < tol_eig:
            beta = 0.0

        if beta == 0.0:
            return f"{alpha:.6g}"
        sign = "+" if beta >= 0 else "-"
        return f"{alpha:.6g} {sign} {abs(beta):.6g}i"

    eig_strs = [fmt_lam(l) for l in eigvals]

    re = np.real(eigvals)
    im = np.imag(eigvals)
    tipo = "Indeterminado"
    if np.all(np.isreal(eigvals)):
        if np.all(re < 0):
            tipo = "Nodo estable (atractor)"
        elif np.all(re > 0):
            tipo = "Nodo inestable (repulsor)"
        elif re[0] * re[1] < 0:
            tipo = "Punto silla (saddle)"
        else:
            tipo = "Nodo degenerado"
    else:
        if np.all(re < 0):
            tipo = "Espiral estable (atractor espiral)"
        elif np.all(re > 0):
            tipo = "Espiral inestable (repulsor espiral)"
        elif np.allclose(re, 0):
            tipo = "Centro (neutro, oscilatorio)"
        else:
            tipo = "Espiral (mixto)"

    try:
        P = eigvecs
        P_inv = np.linalg.inv(P)
    except Exception:
        P = None
        P_inv = None

    t = np.linspace(0, t_max, num_points)

    def analytic_solution(x0_vec):
        x0_vec = np.asarray(x0_vec, dtype=float)
        if P_inv is None:
            return None
        c = P_inv.dot(x0_vec)
        sol = np.empty((len(t), 2), dtype=float)
        for i, ti in enumerate(t):
            sol[i, :] = (P.dot(np.diag(np.exp(eigvals * ti))).dot(c)).real
        return sol

    sol_x0 = analytic_solution(x0)

    (xmin, xmax), (ymin, ymax) = grid_limits
    nx, ny = 20, 20
    X, Y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y

    nulls = []
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    if abs(b) > 1e-12:
        ys1 = -a / b * np.linspace(xmin, xmax, 2)
        xs1 = np.linspace(xmin, xmax, 2)
        nulls.append((xs1, ys1, "dx/dt=0"))
    else:
        xs1 = np.zeros(2)
        ys1 = np.linspace(ymin, ymax, 2)
        nulls.append((xs1, ys1, "dx/dt=0"))

    if abs(d) > 1e-12:
        ys2 = -c / d * np.linspace(xmin, xmax, 2)
        xs2 = np.linspace(xmin, xmax, 2)
        nulls.append((xs2, ys2, "dy/dt=0"))
    else:
        xs2 = np.zeros(2)
        ys2 = np.linspace(ymin, ymax, 2)
        nulls.append((xs2, ys2, "dy/dt=0"))

    if show_plot:
        fig, ax = plt.subplots(figsize=(7, 7))
        # Magnitud y dirección
        speed = np.hypot(U, V)
        angles = np.arctan2(V, U)  # en radianes (-pi, pi)

        # Streamplot de fondo (suave)
        ax.streamplot(X, Y, U, V, color='lightgray', density=1.2, linewidth=1)

        # Quiver principal: flechas coloreadas por ángulo para mostrar dirección claramente
        # Normalizar vectores para que las flechas de fondo sean uniformes en tamaño
        with np.errstate(invalid='ignore', divide='ignore'):
            U_norm = np.where(speed == 0, 0, U / speed)
            V_norm = np.where(speed == 0, 0, V / speed)

        # Colormap por ángulo (hsv da un mapa de direcciones intuitivo)
        cmap = plt.get_cmap('hsv')
        angle_norm = (angles + np.pi) / (2 * np.pi)  # normalizar a [0,1]
        flat_color = cmap(angle_norm.flatten())
        flat_color = flat_color.reshape(X.shape + (4,))

        # Dibujar flechitas pequeñas de fondo con cabecitas visibles
        # Escala en "xy" para que la longitud sea proporcional al plot
        quiver = ax.quiver(
            X,
            Y,
            U_norm,
            V_norm,
            angle_norm,
            cmap='hsv',
            pivot='mid',
            scale_units='xy',
            scale=10,
            width=0.0035,
            headwidth=3,
            headlength=5,
            minlength=0.1,
        )
        # Señalizar que la figura contiene quiver para el renderer
        fig._has_quiver = True
        # Añadir una barra de colores que indica la dirección (ángulo)
        cbar = fig.colorbar(quiver, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Dirección (normalizada: -π→π)')

        for xs, ys, label in nulls:
            ax.plot(xs, ys)

        # No mostrar etiquetas de núclinas para mantener el diagrama limpio

        starts = []
        grid_init = np.array([[-4, -4], [-4, 0], [-4, 4], [0, -4], [0, 4], [4, -4], [4, 0], [4, 4]])
        for xi in grid_init:
            sol = analytic_solution(xi)
            if sol is not None:
                ax.plot(sol[:, 0], sol[:, 1], '-', linewidth=1)
                ax.plot(sol[0, 0], sol[0, 1], 'o', color='C1')

        # Marcar punto de equilibrio (solo marcador, sin texto)
        eq_x, eq_y = float(eq_point[0]), float(eq_point[1])
        ax.scatter([eq_x], [eq_y], color='red', s=60, zorder=6)

        # Dibujar autovectores en el punto de equilibrio (si son reales)
        try:
            for idx in range(eigvecs.shape[1]):
                v = np.real(eigvecs[:, idx])
                if np.linalg.norm(v) < 1e-12:
                    continue
                v = v / np.linalg.norm(v)
                arrow_len = max(xmax - xmin, ymax - ymin) * 0.2
                ax.arrow(eq_x, eq_y, v[0] * arrow_len, v[1] * arrow_len, head_width=0.12 * arrow_len, color='C3', linewidth=2, length_includes_head=True)
                ax.arrow(eq_x, eq_y, -v[0] * arrow_len, -v[1] * arrow_len, head_width=0.12 * arrow_len, color='C3', linewidth=2, length_includes_head=True)
        except Exception:
            pass

        if sol_x0 is not None:
            ax.plot(sol_x0[:, 0], sol_x0[:, 1], 'r-', linewidth=2, label='Trayectoria x0')
            ax.plot(sol_x0[0, 0], sol_x0[0, 1], 'ro')

        # Ajustar límites con padding para alejar un poco la vista (menos zoom)
        xpad = max(0.2 * (xmax - xmin), 0.5)
        ypad = max(0.2 * (ymax - ymin), 0.5)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Diagrama de fase y núclinas')
        ax.legend()
        # No llamar plt.show() — devolver la figura en el resultado para que el caller la renderice
        fig_phase = fig

    resultado = {
        'A': A,
        'autovalores': eig_strs,
        'autovectores': eigvecs,
        'tipo': tipo,
        'solucion_analitica_t': t,
        'solucion_analitica_x0': sol_x0,
        'fig_phase': fig_phase if show_plot else None,
    }

    # Si se solicita, construir la solución simbólica general X(T) = exp(A*T) * [C1, C2]^T
    if symbolic:
        try:
            T = sp.symbols('T')
            A_sym = sp.Matrix(A)

            # Rutin a robusta para exponencial matricial en 2x2 usando descomposición por valores propios / Jordan
            def symbolic_matrix_exp_2x2(M, t):
                I = sp.eye(2)
                eigs = list(M.eigenvals().items())
                if len(eigs) == 2:
                    # Dos autovalores (posiblemente iguales listed separately)
                    # Build list of (lambda, multiplicity)
                    lambdas = []
                    for lam, mult in eigs:
                        lambdas.append((sp.simplify(lam), int(mult)))
                    # If distinct
                    if lambdas[0][0] != lambdas[1][0]:
                        l1 = lambdas[0][0]
                        l2 = lambdas[1][0]
                        expAT = (sp.exp(l1 * t) * (M - l2 * I) / (l1 - l2) + sp.exp(l2 * t) * (M - l1 * I) / (l2 - l1))
                        return sp.simplify(expAT)
                    else:
                        # same eigenvalue with multiplicities
                        lam = lambdas[0][0]
                        N = M - lam * I
                        expAT = sp.exp(lam * t) * (I + N * t)
                        return sp.simplify(expAT)
                else:
                    # Fallback: use general matrix exponential
                    return sp.exp(M * t)

            expAT = symbolic_matrix_exp_2x2(A_sym, T)
            C1, C2 = sp.symbols('C1 C2')
            C = sp.Matrix([C1, C2])
            X_T = sp.simplify(expAT * C)
            # Proveer formas simplificadas y factorizadas para mejor lectura
            X_T_simpl = sp.simplify(sp.expand(X_T))
            X_T_fact = sp.factor(X_T_simpl)
            resultado['solucion_symbolica'] = X_T_simpl
            resultado['solucion_symbolica_factored'] = X_T_fact

            # También devolver la forma en términos de condiciones iniciales simbólicas x0,y0 (C1=Cx0, C2=Cy0)
            x0_sym, y0_sym = sp.symbols('x0 y0')
            X_T_x0 = sp.simplify(X_T_simpl.subs({C1: x0_sym, C2: y0_sym}))
            resultado['solucion_symbolica_x0'] = X_T_x0
        except Exception:
            resultado['solucion_symbolica'] = None
            resultado['solucion_symbolica_factored'] = None
            resultado['solucion_symbolica_x0'] = None

    print('\nMatriz A:')
    print(A)
    print('\nAutovalores:')
    for s in eig_strs:
        print(' -', s)
    print('\nTipo de punto fijo:', tipo)

    return resultado


def main():
    while True:
        print("\n" + "=" * 80)
        print("MENÚ PRINCIPAL - MÉTODOS NUMÉRICOS PARA RAÍCES")
        print("=" * 80)
        print("1. Método de Newton-Raphson")
        print("2. Método de Aceleración de Aitken")
        print("3. Método de Bisección")
        print("4. Método de Punto Fijo")
        print("5. Comparativa de los 4 métodos")
        print("6. Interpolación Lagrange + Derivación + Diferencias Divididas")
        print("7. Integración numérica (Trapecio, Simpson 1/3 y 3/8)")
        print("8. Ajuste de curvas (Regresión lineal y polinomial)")
        print("9. Sistemas lineales (Gauss-Jordan y Gauss-Seidel)")
        print("10. EDO (Euler, Heun y RK4)")
        print("11. Red neuronal base (Descenso de gradiente)")
        print("12. Salir")
        print("=" * 80)

        try:
            opcion = int(input("Selecciona una opción (1-12): ").strip())
        except ValueError:
            print("Error: Debe ingresar un número entero.")
            continue

        if opcion == 1:
            print("\nIngrese la función como expresión Python:")
            print("Ejemplos: x**3 - 2*x - 5, sin(x) - x/2, exp(x) - 3*x")
            funcion = input("\nf(x) = ").strip()

            if not funcion:
                print("Error: Debe ingresar una función")
                continue

            try:
                x0 = parse_real(input("\nAproximación inicial (x₀): ").strip(), "x₀")
            except ValueError:
                print("Error: x₀ debe ser un número")
                continue

            try:
                tol_input = input("\nTolerancia de error (default 1e-6): ").strip()
                tolerancia = parse_real_or_default(tol_input, 1e-6, "tolerancia")
            except ValueError:
                print("Error: Tolerancia debe ser un número")
                continue

            try:
                iter_input = input("\nMáximo de iteraciones (default 100): ").strip()
                max_iter = parse_int_or_default(iter_input, 100, "máximo de iteraciones")
            except ValueError:
                print("Error: Máximo de iteraciones debe ser un entero")
                continue

            raiz, iteraciones, convergencia = metodo_newton_raphson(funcion, x0, tolerancia, max_iter)

            if iteraciones:
                mostrar_tabla_newton(iteraciones)

                print("\n" + "=" * 80)
                print("RESULTADO FINAL")
                print("=" * 80)
                print(f"Raíz encontrada: x = {raiz:.15e}")
                print(f"Convergencia: {' Sí' if convergencia else ' No'}")
                print(f"Número de iteraciones: {len(iteraciones)}")
                print("=" * 80 + "\n")

                graficar = input("¿Desea graficar la función? (s/n): ").strip().lower()
                if graficar in ["s", "si"]:
                    try:
                        graficar_newton(funcion, raiz)
                    except Exception as e:
                        print(f"Error al graficar: {e}")

        elif opcion == 2:
            try:
                g_str = input("\nIngrese la función de iteración g(x) (ej: cos(x), x**2/10 + 1, sqrt(x)): ").strip()

                if not g_str:
                    print("Error: Debe ingresar una función")
                    continue

                x0 = parse_real(input("Ingrese la aproximación inicial x₀: "), "x₀")

                tolerancia_input = input("Ingrese la tolerancia de error (Enter para 1e-6): ").strip()
                tolerancia = parse_real_or_default(tolerancia_input, 1e-6, "tolerancia")

                max_iter_input = input("Ingrese el máximo de iteraciones (Enter para 100): ").strip()
                max_iteraciones = parse_int_or_default(max_iter_input, 100, "máximo de iteraciones")

                raiz, iteraciones, convergencia = metodo_aitken(g_str, x0, tolerancia, max_iteraciones)

                if raiz is not None:
                    mostrar_tabla_aitken(iteraciones)

                    print(f"{'=' * 100}")
                    print(f"RESUMEN DE RESULTADOS".center(100))
                    print(f"{'=' * 100}")
                    print(f"{'Raíz encontrada:':.<50} {raiz:.15f}")
                    print(f"{'Número de iteraciones:':.<50} {len(iteraciones)}")
                    print(f"{'Convergencia:':.<50} {' SÍ' if convergencia else '✗ NO'}")
                    print(f"{'=' * 100}\n")

                    mostrar_grafico = input("¿Desea visualizar el gráfico de convergencia? (s/n): ").strip().lower()
                    if mostrar_grafico in ["s", "si", "sí", "yes", "y"]:
                        graficar_aitken(g_str, x0, raiz, iteraciones)

            except ValueError as e:
                print(f"Error de entrada: {e}")

        elif opcion == 3:
            func_str = input("Ingresa la función en términos de x (ej. x**2 - 2): ")
            a = parse_real(input("Ingresa el límite inferior a: "), "a")
            b = parse_real(input("Ingresa el límite superior b: "), "b")
            tol_input = input("Ingresa la tolerancia (por defecto 1e-6): ")
            tol = parse_real_or_default(tol_input, 1e-6, "tolerancia")
            max_iter_input = input("Ingresa el máximo de iteraciones (por defecto 100): ")
            max_iter = parse_int_or_default(max_iter_input, 100, "máximo de iteraciones")

            try:
                raiz, _ = metodo_biseccion(func_str, a, b, tol, max_iter)
                print(f"\nRaíz aproximada encontrada: {raiz}")

                ver_grafico = input("\n¿Deseas ver el gráfico de la función? (s/n): ").lower()
                if ver_grafico == "s":
                    graficar_biseccion(func_str, a, b, raiz)
            except ValueError as e:
                print(f"Error: {e}")

        elif opcion == 4:
            func_str = input("Ingresa la función f(x) = 0 (ej. x**2 - 2): ")
            g_str = input("Ingresa la función g(x) para el punto fijo (ej. x - (x**2 - 2)/(2*x)): ")
            x0 = parse_real(input("Ingresa el valor inicial x0: "), "x0")
            tol_input = input("Ingresa la tolerancia (por defecto 1e-6): ")
            tol = parse_real_or_default(tol_input, 1e-6, "tolerancia")
            max_iter_input = input("Ingresa el máximo de iteraciones (por defecto 100): ")
            max_iter = parse_int_or_default(max_iter_input, 100, "máximo de iteraciones")

            try:
                raiz, _ = metodo_punto_fijo(g_str, x0, tol, max_iter)
                print(f"\nRaíz aproximada encontrada: {raiz}")

                graficar = input("¿Quieres ver el gráfico de la función? (s/n): ").lower()
                if graficar == "s":
                    a = parse_real(input("Ingresa el límite inferior para el gráfico: "), "a")
                    b = parse_real(input("Ingresa el límite superior para el gráfico: "), "b")
                    graficar_punto_fijo(func_str, a, b, raiz)
            except ValueError as e:
                print(f"Error: {e}")

        elif opcion == 5:
            comparativa_metodos()

        elif opcion == 6:
            ejecutar_metodo_lagrange()

        elif opcion == 7:
            ejecutar_integracion_numerica()

        elif opcion == 8:
            ejecutar_ajuste_curvas()

        elif opcion == 9:
            ejecutar_sistemas_lineales()

        elif opcion == 10:
            ejecutar_edo()

        elif opcion == 11:
            ejecutar_red_neuronal_descenso_gradiente()

        elif opcion == 12:
            print("¡Hasta luego!")
            break

        else:
            print("Opción no válida. Intente de nuevo.")


if __name__ == "__main__":
    main()
