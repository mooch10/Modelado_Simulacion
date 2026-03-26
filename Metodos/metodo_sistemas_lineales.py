import numpy as np

from Metodos.input_parser import parse_real, parse_real_or_default, parse_int_or_default


def gauss_jordan(A, b, tol=1e-12):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A debe ser una matriz cuadrada")
    if b.shape[0] != A.shape[0]:
        raise ValueError("b debe tener la misma dimension que A")

    n = A.shape[0]
    M = np.hstack([A, b])
    pasos = []

    for col in range(n):
        pivote = col + int(np.argmax(np.abs(M[col:, col])))
        if abs(M[pivote, col]) < tol:
            raise ValueError("El sistema no tiene solucion unica (pivote cercano a cero)")

        if pivote != col:
            M[[col, pivote], :] = M[[pivote, col], :]
            pasos.append(
                {
                    "paso": f"Intercambio F{col + 1} <-> F{pivote + 1}",
                    "matriz": M.copy(),
                }
            )

        valor_pivote = M[col, col]
        M[col, :] = M[col, :] / valor_pivote
        pasos.append(
            {
                "paso": f"Normalizacion F{col + 1} / {valor_pivote:.7g}",
                "matriz": M.copy(),
            }
        )

        for fila in range(n):
            if fila == col:
                continue
            factor = M[fila, col]
            if abs(factor) < tol:
                continue
            M[fila, :] = M[fila, :] - factor * M[col, :]
            pasos.append(
                {
                    "paso": f"F{fila + 1} <- F{fila + 1} - ({factor:.7g})*F{col + 1}",
                    "matriz": M.copy(),
                }
            )

    solucion = M[:, -1]
    return solucion, pasos


def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A debe ser una matriz cuadrada")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("b debe ser un vector de la misma dimension que A")

    n = A.shape[0]

    if np.any(np.abs(np.diag(A)) < 1e-15):
        raise ValueError("Gauss-Seidel requiere diagonal no nula")

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float)
        if x.shape != (n,):
            raise ValueError("x0 debe tener la misma dimension que b")

    iteraciones = []
    convergio = False

    for k in range(1, max_iter + 1):
        x_anterior = x.copy()

        for i in range(n):
            suma1 = np.dot(A[i, :i], x[:i])
            suma2 = np.dot(A[i, i + 1:], x_anterior[i + 1:])
            x[i] = (b[i] - suma1 - suma2) / A[i, i]

        error = float(np.linalg.norm(x - x_anterior, ord=np.inf))
        fila = {"Iteracion": k, "Error_inf": error}
        for i in range(n):
            fila[f"x{i + 1}"] = float(x[i])
        iteraciones.append(fila)

        if error < tol:
            convergio = True
            break

    return x, iteraciones, convergio


def _leer_matriz_vector(n):
    A = []
    b = []

    print("Ingrese los coeficientes de la matriz A fila por fila (separados por coma):")
    for i in range(n):
        texto = input(f"Fila {i + 1}: ").strip()
        partes = [p.strip() for p in texto.split(",") if p.strip()]
        if len(partes) != n:
            raise ValueError(f"La fila {i + 1} debe tener exactamente {n} valores")
        fila = [parse_real(p, f"A[{i + 1},{j + 1}]") for j, p in enumerate(partes)]
        A.append(fila)

    print("Ingrese los terminos independientes b (uno por fila):")
    for i in range(n):
        b.append(parse_real(input(f"b[{i + 1}]: ").strip(), f"b[{i + 1}]"))

    return np.array(A, dtype=float), np.array(b, dtype=float)


def ejecutar_sistemas_lineales():
    print("\n" + "=" * 80)
    print("RESOLUCION DE SISTEMAS LINEALES")
    print("=" * 80)
    print("1. Gauss-Jordan")
    print("2. Gauss-Seidel")

    opcion = input("Seleccione metodo (1-2): ").strip()
    if opcion not in {"1", "2"}:
        print("Opcion no valida")
        return

    try:
        n = parse_int_or_default(input("Dimension del sistema n (default 3): "), 3, "n")
        if n <= 0:
            raise ValueError("n debe ser positivo")

        A, b = _leer_matriz_vector(n)

        if opcion == "1":
            solucion, pasos = gauss_jordan(A, b)
            print("\nSolucion:")
            for i, val in enumerate(solucion, start=1):
                print(f"x{i} = {val:.12g}")
            print(f"\nPasos registrados: {len(pasos)}")

        else:
            tol = parse_real_or_default(input("Tolerancia (default 1e-6): "), 1e-6, "tolerancia")
            max_iter = parse_int_or_default(input("Max iteraciones (default 100): "), 100, "max iteraciones")
            solucion, iteraciones, convergio = gauss_seidel(A, b, tol=tol, max_iter=max_iter)
            print("\nSolucion aproximada:")
            for i, val in enumerate(solucion, start=1):
                print(f"x{i} = {val:.12g}")
            print(f"Convergencia: {'Si' if convergio else 'No'}")
            print(f"Iteraciones realizadas: {len(iteraciones)}")

    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    ejecutar_sistemas_lineales()
