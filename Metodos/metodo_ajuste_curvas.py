import numpy as np
import matplotlib.pyplot as plt

from Metodos.input_parser import parse_real, parse_int_or_default


def normalizar_xy(x_vals, y_vals):
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x e y deben ser arreglos unidimensionales")
    if x.size != y.size:
        raise ValueError("x e y deben tener la misma cantidad de datos")
    if x.size < 2:
        raise ValueError("Se requieren al menos 2 puntos")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Los datos contienen valores no finitos")

    return x, y


def _r2(y_real, y_pred):
    ss_res = float(np.sum((y_real - y_pred) ** 2))
    ss_tot = float(np.sum((y_real - np.mean(y_real)) ** 2))
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    return 1.0 - ss_res / ss_tot


def regresion_lineal(x_vals, y_vals):
    x, y = normalizar_xy(x_vals, y_vals)

    A = np.column_stack((x, np.ones_like(x)))
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    m, b = coef

    y_pred = m * x + b
    r2 = _r2(y, y_pred)

    return {
        "tipo": "lineal",
        "coeficientes": [float(m), float(b)],
        "ecuacion": f"y = {m:.7g}x + {b:.7g}",
        "y_pred": y_pred,
        "r2": float(r2),
    }


def regresion_polinomial(x_vals, y_vals, grado):
    x, y = normalizar_xy(x_vals, y_vals)

    if grado < 1:
        raise ValueError("El grado debe ser mayor o igual a 1")
    if grado >= len(x):
        raise ValueError("El grado debe ser menor que la cantidad de puntos")

    V = np.vander(x, N=grado + 1, increasing=False)
    coef, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
    y_pred = np.polyval(coef, x)
    r2 = _r2(y, y_pred)

    partes = []
    for i, c in enumerate(coef):
        pot = grado - i
        if pot == 0:
            partes.append(f"{c:.7g}")
        elif pot == 1:
            partes.append(f"{c:.7g}x")
        else:
            partes.append(f"{c:.7g}x^{pot}")

    return {
        "tipo": "polinomial",
        "grado": int(grado),
        "coeficientes": [float(c) for c in coef],
        "ecuacion": "y = " + " + ".join(partes).replace("+ -", "- "),
        "y_pred": y_pred,
        "r2": float(r2),
    }


def graficar_ajuste(x_vals, y_vals, y_line, titulo):
    x, y = normalizar_xy(x_vals, y_vals)

    orden = np.argsort(x)
    x_ord = x[orden]
    y_line_ord = np.array(y_line, dtype=float)[orden]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(x, y, color="tab:red", label="Datos", zorder=3)
    ax.plot(x_ord, y_line_ord, color="tab:blue", linewidth=2, label="Ajuste")
    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def _leer_lista(mensaje):
    texto = input(mensaje).strip()
    partes = [p.strip() for p in texto.split(",") if p.strip()]
    if not partes:
        raise ValueError("Debe ingresar al menos un dato")
    return [parse_real(p, "dato") for p in partes]


def ejecutar_ajuste_curvas():
    print("\n" + "=" * 80)
    print("AJUSTE DE CURVAS - MINIMOS CUADRADOS")
    print("=" * 80)
    print("1. Regresion lineal")
    print("2. Regresion polinomial")

    opcion = input("Seleccione metodo (1-2): ").strip()
    if opcion not in {"1", "2"}:
        print("Opcion no valida")
        return

    try:
        x = _leer_lista("Ingrese x separados por coma: ")
        y = _leer_lista("Ingrese y separados por coma: ")

        if opcion == "1":
            resultado = regresion_lineal(x, y)
            titulo = "Ajuste lineal"
        else:
            grado = parse_int_or_default(input("Grado del polinomio (default 2): "), 2, "grado")
            resultado = regresion_polinomial(x, y, grado)
            titulo = f"Ajuste polinomial (grado {grado})"

        print(f"\nEcuacion: {resultado['ecuacion']}")
        print(f"R^2: {resultado['r2']:.7f}")

        ver_grafico = input("¿Desea ver el grafico del ajuste? (s/n): ").strip().lower()
        if ver_grafico in {"s", "si", "sí"}:
            graficar_ajuste(x, y, resultado["y_pred"], titulo)

    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    ejecutar_ajuste_curvas()
