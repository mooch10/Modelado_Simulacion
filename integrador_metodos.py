import numpy as np
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
        print("10. EDO (Euler y RK4)")
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
