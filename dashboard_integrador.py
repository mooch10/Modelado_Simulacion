import io
import contextlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp

from Metodos.input_parser import build_numeric_function
from Metodos.metodo_aitken import metodo_aitken
from Metodos.metodo_de_biseccion import metodo_biseccion
from Metodos.metodo_lagrange_derivacion import (
    aproximar_derivada_tres_formas,
    polinomio_lagrange,
    polinomio_newton_desde_dd,
    tabla_diferencias_divididas,
)
from Metodos.metodo_newton_raphson import metodo_newton_raphson
from Metodos.metodo_punto_fijo import metodo_punto_fijo


ALLOWED_LOCALS = {
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


def run_silent(func, *args, **kwargs):
    """Ejecuta funciones de metodos que imprimen en consola, silenciando su salida."""
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return func(*args, **kwargs)


def safe_eval_expr(expr_text, variable="x"):
    x = sp.Symbol(variable)
    expr = sp.sympify(expr_text, locals=ALLOWED_LOCALS)
    invalid = expr.free_symbols - {x}
    if invalid:
        raise ValueError(f"Variables no permitidas: {invalid}")
    return expr


def parse_expr_list(text):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("Debes ingresar al menos un valor")
    return [sp.sympify(p, locals=ALLOWED_LOCALS) for p in parts]


def to_float_array(values):
    return np.array([float(sp.N(v)) for v in values], dtype=float)


def _format_decimal(value, max_decimals=7):
    txt = f"{value:.{max_decimals}f}".rstrip("0").rstrip(".")
    return "0" if txt in {"", "-0"} else txt


def _is_terminating_denominator(den):
    """Un racional termina en decimal solo si su denominador tiene factores 2 y/o 5."""
    d = abs(int(den))
    if d == 0:
        return False
    for p in (2, 5):
        while d % p == 0:
            d //= p
    return d == 1


def _format_number_hybrid(value_expr, max_decimals=7, latex=False):
    """
    Formato mixto:
    - Fraccion para racionales pequenos o decimales periodicos/no terminantes.
    - Decimal para enteros y racionales con decimal terminante "comodo".
    """
    v = sp.nsimplify(value_expr, rational=True)

    if v.is_Rational:
        v_rat = sp.Rational(v)
        den = int(v_rat.q)
        num = int(v_rat.p)

        if den == 1:
            return str(num)

        use_fraction = (den <= 20) or (not _is_terminating_denominator(den))
        if use_fraction:
            if latex:
                return f"\\frac{{{num}}}{{{den}}}"
            return f"{num}/{den}"

        return _format_decimal(float(v_rat), max_decimals)

    return _format_decimal(float(sp.N(v)), max_decimals)


def polynomial_to_decimal_text(expr, variable="x", max_decimals=7):
    x = sp.Symbol(variable)
    expr = sp.expand(expr)

    try:
        poly = sp.Poly(expr, x)
    except sp.PolynomialError:
        return _format_number_hybrid(expr, max_decimals=max_decimals, latex=False)

    terms = []
    eps = 10 ** (-(max_decimals + 2))

    for (power,), coef in sorted(poly.terms(), key=lambda t: t[0][0], reverse=True):
        coef_expr = sp.nsimplify(coef, rational=True)
        coef_val = float(sp.N(coef_expr))
        if abs(coef_val) < eps:
            continue

        sign = "-" if coef_val < 0 else "+"
        coef_abs_expr = abs(coef_expr)
        coef_abs_txt = _format_number_hybrid(coef_abs_expr, max_decimals=max_decimals, latex=False)
        is_one = sp.simplify(coef_abs_expr - 1) == 0

        if power == 0:
            body = coef_abs_txt
        elif power == 1:
            body = variable if is_one else f"{coef_abs_txt}{variable}"
        else:
            body = f"{variable}^{power}" if is_one else f"{coef_abs_txt}{variable}^{power}"

        terms.append((sign, body))

    if not terms:
        return "0"

    first_sign, first_body = terms[0]
    out = f"-{first_body}" if first_sign == "-" else first_body

    for sign, body in terms[1:]:
        out += f" {sign} {body}"

    return out


def polynomial_to_decimal_latex(expr, variable="x", max_decimals=7):
    x = sp.Symbol(variable)
    expr = sp.expand(expr)

    try:
        poly = sp.Poly(expr, x)
    except sp.PolynomialError:
        return _format_number_hybrid(expr, max_decimals=max_decimals, latex=True)

    terms = []
    eps = 10 ** (-(max_decimals + 2))

    for (power,), coef in sorted(poly.terms(), key=lambda t: t[0][0], reverse=True):
        coef_expr = sp.nsimplify(coef, rational=True)
        coef_val = float(sp.N(coef_expr))
        if abs(coef_val) < eps:
            continue

        sign = "-" if coef_val < 0 else "+"
        coef_abs_expr = abs(coef_expr)
        coef_abs_txt = _format_number_hybrid(coef_abs_expr, max_decimals=max_decimals, latex=True)
        is_one = sp.simplify(coef_abs_expr - 1) == 0

        if power == 0:
            body = coef_abs_txt
        elif power == 1:
            body = variable if is_one else f"{coef_abs_txt}{variable}"
        else:
            body = f"{variable}^{{{power}}}" if is_one else f"{coef_abs_txt}{variable}^{{{power}}}"

        terms.append((sign, body))

    if not terms:
        return "0"

    first_sign, first_body = terms[0]
    out = f"-{first_body}" if first_sign == "-" else first_body

    for sign, body in terms[1:]:
        out += f" {sign} {body}"

    return out


def build_func_plot(expr_text, x_values):
    _, f_num = build_numeric_function(expr_text)
    y = np.array(f_num(x_values), dtype=float)
    return y


def plot_function_with_root(func_text, root, xmin, xmax, title):
    x = np.linspace(xmin, xmax, 800)
    y = build_func_plot(func_text, x)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, y, linewidth=2, label=f"f(x) = {func_text}")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.plot([root], [0], "ro", markersize=8, label=f"Raiz aprox: {root:.8g}")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_error_curve(errors, title, y_label="Error absoluto"):
    idx = np.arange(1, len(errors) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.2))

    errors = np.array(errors, dtype=float)
    valid = np.all(errors > 0)

    if valid:
        ax.semilogy(idx, errors, "o-", linewidth=2)
    else:
        ax.plot(idx, errors, "o-", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Iteracion")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3, which="both")
    return fig


def section_newton():
    st.subheader("Metodo de Newton-Raphson")

    with st.form("form_newton"):
        c1, c2 = st.columns(2)
        with c1:
            func = st.text_input("f(x)", value="x**3 - 2*x - 5")
            x0 = st.number_input("x0", value=2.0)
        with c2:
            tol = st.number_input("Tolerancia", value=1e-6, format="%.1e")
            max_iter = st.number_input("Max iteraciones", value=100, min_value=1, step=1)
        run_btn = st.form_submit_button("Ejecutar Newton")

    if run_btn:
        try:
            root, iterations, converged = run_silent(
                metodo_newton_raphson, func, float(x0), float(tol), int(max_iter)
            )
            if not iterations:
                st.error("No se generaron iteraciones.")
                return

            df = pd.DataFrame(iterations)
            st.dataframe(df, use_container_width=True)

            st.metric("Raiz aproximada", f"{root:.12g}")
            st.metric("Convergencia", "Si" if converged else "No")
            st.metric("Iteraciones", len(iterations))

            err_col = "Error |x_(n+1) - x_n|"
            errors = df[err_col].astype(float).to_numpy()

            margin = max(2.0, abs(float(root) - float(x0)) + 1.0)
            fig_f = plot_function_with_root(
                func,
                float(root),
                float(root) - margin,
                float(root) + margin,
                "Funcion y raiz aproximada (Newton)",
            )
            st.pyplot(fig_f)
            plt.close(fig_f)

            fig_e = plot_error_curve(errors, "Error por iteracion (Newton)")
            st.pyplot(fig_e)
            plt.close(fig_e)

        except Exception as exc:
            st.error(f"Error al ejecutar Newton: {exc}")


def section_aitken():
    st.subheader("Metodo de Aitken")

    with st.form("form_aitken"):
        c1, c2 = st.columns(2)
        with c1:
            g = st.text_input("g(x)", value="cos(x)")
            x0 = st.number_input("x0", value=0.5, key="aitken_x0")
        with c2:
            tol = st.number_input("Tolerancia", value=1e-6, format="%.1e", key="aitken_tol")
            max_iter = st.number_input("Max iteraciones", value=100, min_value=1, step=1, key="aitken_max")
        run_btn = st.form_submit_button("Ejecutar Aitken")

    if run_btn:
        try:
            root, iterations, converged = run_silent(
                metodo_aitken, g, float(x0), float(tol), int(max_iter)
            )
            if root is None or not iterations:
                st.error("No se pudo obtener resultado para Aitken.")
                return

            df = pd.DataFrame(iterations)
            st.dataframe(df, use_container_width=True)

            st.metric("Raiz aproximada", f"{root:.12g}")
            st.metric("Convergencia", "Si" if converged else "No")
            st.metric("Iteraciones", len(iterations))

            errors = df["Error"].astype(float).to_numpy()
            fig_e = plot_error_curve(errors, "Error por iteracion (Aitken)")
            st.pyplot(fig_e)
            plt.close(fig_e)

            margin = max(2.0, abs(float(root) - float(x0)) + 1.0)
            x = np.linspace(float(root) - margin, float(root) + margin, 800)
            y_g = build_func_plot(g, x)

            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.plot(x, y_g, linewidth=2, label=f"g(x) = {g}")
            ax.plot(x, x, "--", linewidth=1.5, label="y = x")
            ax.plot([root], [root], "ro", label=f"Punto fijo: {root:.8g}")
            ax.set_title("Diagrama de punto fijo (Aitken)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as exc:
            st.error(f"Error al ejecutar Aitken: {exc}")


def section_biseccion():
    st.subheader("Metodo de Biseccion")

    with st.form("form_biseccion"):
        c1, c2 = st.columns(2)
        with c1:
            func = st.text_input("f(x)", value="x**3 - x - 2", key="bis_f")
            a = st.number_input("a", value=1.0, key="bis_a")
        with c2:
            b = st.number_input("b", value=2.0, key="bis_b")
            tol = st.number_input("Tolerancia", value=1e-6, format="%.1e", key="bis_tol")
            max_iter = st.number_input("Max iteraciones", value=100, min_value=1, step=1, key="bis_max")
        run_btn = st.form_submit_button("Ejecutar Biseccion")

    if run_btn:
        try:
            root, rows = run_silent(
                metodo_biseccion,
                func,
                float(a),
                float(b),
                float(tol),
                int(max_iter),
            )
            if not rows:
                st.error("No se generaron iteraciones para Biseccion.")
                return

            cols = ["Iteracion", "a", "b", "c", "f(a)", "f(b)", "f(c)"]
            df = pd.DataFrame(rows, columns=cols)
            df["Error_f(c)"] = df["f(c)"].abs()
            df["Semiancho_intervalo"] = (df["b"] - df["a"]).abs() / 2.0
            st.dataframe(df, use_container_width=True)

            st.metric("Raiz aproximada", f"{root:.12g}")
            st.metric("Iteraciones", len(df))
            st.metric("Error final |f(c)|", f"{float(df['Error_f(c)'].iloc[-1]):.3e}")

            fig_f = plot_function_with_root(func, float(root), float(a), float(b), "Funcion y raiz aproximada (Biseccion)")
            st.pyplot(fig_f)
            plt.close(fig_f)

            fig_e = plot_error_curve(df["Error_f(c)"].to_numpy(), "Error por iteracion (Biseccion)", "|f(c)|")
            st.pyplot(fig_e)
            plt.close(fig_e)

        except Exception as exc:
            st.error(f"Error al ejecutar Biseccion: {exc}")


def section_punto_fijo():
    st.subheader("Metodo de Punto Fijo")

    with st.form("form_punto_fijo"):
        c1, c2 = st.columns(2)
        with c1:
            f_plot = st.text_input("f(x) para graficar", value="x**2 - 2")
            g = st.text_input("g(x)", value="(x + 2/x)/2")
            x0 = st.number_input("x0", value=1.0, key="pf_x0")
        with c2:
            tol = st.number_input("Tolerancia", value=1e-6, format="%.1e", key="pf_tol")
            max_iter = st.number_input("Max iteraciones", value=100, min_value=1, step=1, key="pf_max")
            xmin = st.number_input("x min grafico", value=-2.0, key="pf_xmin")
            xmax = st.number_input("x max grafico", value=2.0, key="pf_xmax")
        run_btn = st.form_submit_button("Ejecutar Punto Fijo")

    if run_btn:
        try:
            root, rows = run_silent(
                metodo_punto_fijo,
                g,
                float(x0),
                float(tol),
                int(max_iter),
            )
            if not rows:
                st.error("No se generaron iteraciones para Punto Fijo.")
                return

            cols = ["Iteracion", "x_n", "x_n1", "Error"]
            df = pd.DataFrame(rows, columns=cols)
            st.dataframe(df, use_container_width=True)

            st.metric("Raiz aproximada", f"{root:.12g}")
            st.metric("Iteraciones", len(df))
            st.metric("Error final", f"{float(df['Error'].iloc[-1]):.3e}")

            if xmax <= xmin:
                st.warning("Para graficar f(x), se requiere x max > x min.")
            else:
                fig_f = plot_function_with_root(
                    f_plot,
                    float(root),
                    float(xmin),
                    float(xmax),
                    "Funcion y raiz aproximada (Punto Fijo)",
                )
                st.pyplot(fig_f)
                plt.close(fig_f)

            fig_e = plot_error_curve(df["Error"].to_numpy(), "Error por iteracion (Punto Fijo)")
            st.pyplot(fig_e)
            plt.close(fig_e)

            margin = max(2.0, abs(float(root) - float(x0)) + 1.0)
            x = np.linspace(float(root) - margin, float(root) + margin, 800)
            y_g = build_func_plot(g, x)

            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.plot(x, y_g, linewidth=2, label=f"g(x) = {g}")
            ax.plot(x, x, "--", linewidth=1.5, label="y = x")
            ax.plot([root], [root], "ro", label=f"Punto fijo: {root:.8g}")
            ax.set_title("Diagrama de punto fijo")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as exc:
            st.error(f"Error al ejecutar Punto Fijo: {exc}")


def section_comparativa():
    st.subheader("Comparativa de los 4 metodos")

    with st.form("form_comparativa"):
        c1, c2 = st.columns(2)
        with c1:
            func = st.text_input("f(x)", value="x**3 - 2*x - 5", key="cmp_f")
            g = st.text_input("g(x)", value="cos(x)", key="cmp_g")
            x0 = st.number_input("x0", value=1.0, key="cmp_x0")
        with c2:
            a = st.number_input("a", value=1.0, key="cmp_a")
            b = st.number_input("b", value=2.0, key="cmp_b")
            tol = st.number_input("Tolerancia", value=1e-6, format="%.1e", key="cmp_tol")
            max_iter = st.number_input("Max iteraciones", value=100, min_value=1, step=1, key="cmp_max")
        run_btn = st.form_submit_button("Ejecutar comparativa")

    if run_btn:
        results = []

        try:
            root, iters, conv = run_silent(metodo_newton_raphson, func, float(x0), float(tol), int(max_iter))
            if iters:
                results.append(
                    {
                        "Metodo": "Newton-Raphson",
                        "Raiz": float(root),
                        "Iteraciones": len(iters),
                        "Error_final": float(iters[-1]["Error |x_(n+1) - x_n|"]),
                        "Convergencia": bool(conv),
                    }
                )
        except Exception as exc:
            st.warning(f"Newton no se pudo ejecutar: {exc}")

        try:
            root, iters, conv = run_silent(metodo_aitken, g, float(x0), float(tol), int(max_iter))
            if root is not None and iters:
                results.append(
                    {
                        "Metodo": "Aitken",
                        "Raiz": float(root),
                        "Iteraciones": len(iters),
                        "Error_final": float(iters[-1]["Error"]),
                        "Convergencia": bool(conv),
                    }
                )
        except Exception as exc:
            st.warning(f"Aitken no se pudo ejecutar: {exc}")

        try:
            root, rows = run_silent(metodo_biseccion, func, float(a), float(b), float(tol), int(max_iter))
            if rows:
                results.append(
                    {
                        "Metodo": "Biseccion",
                        "Raiz": float(root),
                        "Iteraciones": len(rows),
                        "Error_final": abs(float(rows[-1][6])),
                        "Convergencia": True,
                    }
                )
        except Exception as exc:
            st.warning(f"Biseccion no se pudo ejecutar: {exc}")

        try:
            root, rows = run_silent(metodo_punto_fijo, g, float(x0), float(tol), int(max_iter))
            if rows:
                results.append(
                    {
                        "Metodo": "Punto Fijo",
                        "Raiz": float(root),
                        "Iteraciones": len(rows),
                        "Error_final": float(rows[-1][3]),
                        "Convergencia": True,
                    }
                )
        except Exception as exc:
            st.warning(f"Punto Fijo no se pudo ejecutar: {exc}")

        if not results:
            st.error("No hubo resultados para mostrar en la comparativa.")
            return

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        fig1, ax1 = plt.subplots(figsize=(8, 4.2))
        ax1.bar(df["Metodo"], df["Iteraciones"])
        ax1.set_title("Iteraciones por metodo")
        ax1.set_ylabel("Cantidad de iteraciones")
        ax1.grid(axis="y", alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(8, 4.2))
        values = np.clip(df["Error_final"].to_numpy(dtype=float), 1e-18, None)
        ax2.bar(df["Metodo"], values)
        ax2.set_yscale("log")
        ax2.set_title("Error final por metodo (escala log)")
        ax2.set_ylabel("Error final")
        ax2.grid(axis="y", alpha=0.3, which="both")
        st.pyplot(fig2)
        plt.close(fig2)


def section_lagrange():
    st.subheader("Lagrange, derivacion y error")

    st.markdown("Ingresa datos para interpolacion. Puedes cargar y manualmente o desde f(x).")

    mode = st.radio(
        "Modo de carga",
        ["y manual", "y desde f(x)"],
        horizontal=True,
        key="lag_mode",
    )
    x_text = st.text_input("x (separados por coma)", value="0, 1, 2, 3", key="lag_x_text")

    if mode == "y manual":
        y_text = st.text_input("y (separados por coma)", value="1, 2, 0, 5", key="lag_y_text")
        f_exact_text = st.text_input("f(x) exacta opcional", value="", key="lag_f_exact_text")
    else:
        y_text = ""
        f_exact_text = st.text_input("f(x) exacta", value="sin(x)", key="lag_f_exact_text")

    build_btn = st.button("Construir interpolacion")

    if build_btn:
        try:
            x_vals = parse_expr_list(x_text)
            if mode == "y manual":
                y_vals = parse_expr_list(y_text)
            else:
                x_sym = sp.Symbol("x")
                f_expr = safe_eval_expr(f_exact_text, "x")
                y_vals = [sp.simplify(f_expr.subs(x_sym, xv)) for xv in x_vals]

            f_exact_expr = None
            if f_exact_text.strip():
                f_exact_expr = safe_eval_expr(f_exact_text, "x")

            st.session_state["lag_x"] = x_vals
            st.session_state["lag_y"] = y_vals
            st.session_state["lag_f_exact"] = f_exact_expr
            st.success("Datos cargados correctamente.")

        except Exception as exc:
            st.error(f"Error en los datos de interpolacion: {exc}")
            return

    if "lag_x" not in st.session_state:
        return

    x_vals = st.session_state["lag_x"]
    y_vals = st.session_state["lag_y"]
    f_exact_expr = st.session_state.get("lag_f_exact")

    try:
        p_lagr, bases_lagr = polinomio_lagrange(x_vals, y_vals)
        p_lagr = sp.expand(p_lagr)
        st.write("Polinomio de Lagrange:")
        st.latex(f"P(x) = {polynomial_to_decimal_latex(p_lagr, max_decimals=7)}")

        st.write("Bases de Lagrange:")
        with st.expander("Ver bases L_i(x)", expanded=True):
            for i, li in enumerate(bases_lagr):
                li_expandida = sp.expand(li)
                st.latex(f"L_{{{i}}}(x) = {sp.latex(li_expandida)}")

        c_eval_1, c_eval_2 = st.columns([2, 1])
        x_eval_text = c_eval_1.text_input(
            "Evaluar P(x) en x*",
            value="1",
            key="lag_eval_xstar",
        )
        eval_btn = c_eval_2.button("Evaluar polinomio", key="lag_eval_btn")

        if eval_btn:
            try:
                x_eval = float(sp.N(sp.sympify(x_eval_text, locals=ALLOWED_LOCALS)))
                x = sp.Symbol("x")
                y_eval = float(sp.N(p_lagr.subs(x, x_eval)))
                st.write(f"P({x_eval:.7f}) = {y_eval:.7f}")
            except Exception as exc:
                st.error(f"No se pudo evaluar el polinomio en x*: {exc}")

        x_num = to_float_array(x_vals)
        y_num = to_float_array(y_vals)

        x_min = float(np.min(x_num))
        x_max = float(np.max(x_num))
        margin = max(1.0, 0.2 * (x_max - x_min + 1e-12))
        x_plot = np.linspace(x_min - margin, x_max + margin, 900)

        x = sp.Symbol("x")
        p_fun = sp.lambdify(x, p_lagr, "numpy")
        y_plot = np.array(p_fun(x_plot), dtype=float)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(x_plot, y_plot, linewidth=2, label="P(x) Lagrange")
        ax.scatter(x_num, y_num, color="red", zorder=3, label="Datos")

        if f_exact_expr is not None:
            f_fun = sp.lambdify(x, f_exact_expr, "numpy")
            y_real = np.array(f_fun(x_plot), dtype=float)
            ax.plot(x_plot, y_real, "--", linewidth=1.8, label="f(x) exacta")

        ax.set_title("Interpolacion de Lagrange")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        if f_exact_expr is not None:
            f_fun = sp.lambdify(x, f_exact_expr, "numpy")
            y_real = np.array(f_fun(x_plot), dtype=float)
            err = np.abs(y_real - y_plot)

            fig_e = plot_error_curve(err, "Error |f(x) - P(x)| en el intervalo")
            st.pyplot(fig_e)
            plt.close(fig_e)

            # Error global maximo en el intervalo de interpolacion [min(x_i), max(x_i)].
            x_global = np.linspace(x_min, x_max, 2000)
            y_global_real = np.array(f_fun(x_global), dtype=float)
            y_global_interp = np.array(p_fun(x_global), dtype=float)
            err_global_arr = np.abs(y_global_real - y_global_interp)
            idx_global_max = int(np.argmax(err_global_arr))
            err_global_max = float(err_global_arr[idx_global_max])
            x_global_max = float(x_global[idx_global_max])

            st.markdown("---")
            st.markdown("Error local en punto especifico y comparacion con error global")

            c_local_1, c_local_2 = st.columns([2, 1])
            x_local_text = c_local_1.text_input(
                "Punto x* para error local",
                value=str(round((x_min + x_max) / 2.0, 6)),
                key="lag_local_xstar",
            )
            calc_local_btn = c_local_2.button("Calcular error local", key="lag_local_btn")

            st.write(
                f"Error global maximo en [{x_min:.7g}, {x_max:.7g}] = {err_global_max:.7f} "
                f"(en x = {x_global_max:.7f})"
            )

            if calc_local_btn:
                try:
                    x_star = float(sp.N(sp.sympify(x_local_text, locals=ALLOWED_LOCALS)))
                    y_star_real = float(sp.N(f_exact_expr.subs(x, x_star)))
                    y_star_interp = float(sp.N(p_lagr.subs(x, x_star)))
                    err_local = abs(y_star_real - y_star_interp)

                    razon = err_local / err_global_max if err_global_max > 1e-15 else np.nan
                    diferencia = err_global_max - err_local

                    st.write(f"f(x*) real = {y_star_real:.7f}")
                    st.write(f"P(x*) = {y_star_interp:.7f}")
                    st.write(f"Error local |f(x*) - P(x*)| = {err_local:.7f}")
                    st.write(f"Error global maximo = {err_global_max:.7f}")

                    if np.isnan(razon):
                        st.write("Comparacion local/global: no definida (error global ~ 0)")
                    else:
                        st.write(f"Relacion local/global = {razon:.7f}")
                        st.write(f"Diferencia (global - local) = {diferencia:.7f}")

                except Exception as exc:
                    st.error(f"No se pudo calcular el error local en x*: {exc}")
        else:
            st.markdown("---")
            st.info("Para calcular y comparar error local vs global, debes ingresar f(x) exacta.")

    except Exception as exc:
        st.error(f"Error al construir la interpolacion: {exc}")
        return

    st.markdown("---")
    st.markdown("Derivacion por aproximacion")

    c1, c2, c3 = st.columns(3)
    forma = c1.selectbox("Forma", ["adelante", "atras", "centrada"])
    x_obj_text = c2.text_input("x objetivo (opcional)", value="")
    deriv_btn = c3.button("Calcular derivada")

    if deriv_btn:
        try:
            x_obj = None
            if x_obj_text.strip():
                x_obj = float(sp.N(sp.sympify(x_obj_text, locals=ALLOWED_LOCALS)))

            result = aproximar_derivada_tres_formas(x_vals, y_vals, forma, x_obj)
            x_eval = float(result["x_evaluacion"])
            d_aprox = float(result["derivada"])

            # Si el usuario ingresa x objetivo, evaluamos la derivada del polinomio local
            # en ese punto para que el valor responda al input de la interfaz.
            if x_obj is not None:
                x = sp.Symbol("x")
                d_poly_local = result["derivada_polinomio_local"]
                d_aprox = float(sp.N(d_poly_local.subs(x, x_obj)))
                x_eval = float(x_obj)

            st.write(f"Derivada aproximada en x = {x_eval:.12g}")
            st.write(f"f'(x) aprox = {d_aprox:.12g}")
            st.write(f"Puntos usados para aproximar: {list(result['x_sub'])}")

            if f_exact_expr is not None:
                x = sp.Symbol("x")
                d_real_expr = sp.diff(f_exact_expr, x)
                d_real = float(sp.N(d_real_expr.subs(x, x_eval)))
                err_abs = abs(d_aprox - d_real)
                err_rel = err_abs / abs(d_real) if abs(d_real) > 1e-15 else np.nan

                st.write(f"f'(x) real = {d_real:.12g}")
                st.write(f"Error absoluto = {err_abs:.12g}")
                if np.isnan(err_rel):
                    st.write("Error relativo = no definido (derivada real cercana a 0)")
                else:
                    st.write(f"Error relativo = {err_rel:.12g}")
            else:
                st.info("No hay f(x) exacta, por eso no se puede calcular el error con la derivada real.")

        except Exception as exc:
            st.error(f"Error al aproximar derivada: {exc}")

    st.markdown("---")
    st.markdown("Diferencias divididas y polinomio de Newton")

    if st.button("Construir tabla DD"):
        try:
            x_dd, table = tabla_diferencias_divididas(x_vals, y_vals)

            n = len(x_dd)
            columns = ["x_i", "f[x_i]"] + [f"DD orden {k}" for k in range(1, n)]
            rows = []
            for i in range(n):
                row = [x_dd[i]]
                for j in range(n):
                    if i <= n - j - 1:
                        row.append(table[i, j])
                    else:
                        row.append(np.nan)
                rows.append(row)

            df = pd.DataFrame(rows, columns=columns)
            st.dataframe(df, use_container_width=True)

            p_newton = sp.expand(polinomio_newton_desde_dd(x_dd, table))
            st.code(f"P_N(x) = {polynomial_to_decimal_text(p_newton, max_decimals=7)}")

        except Exception as exc:
            st.error(f"Error en diferencias divididas: {exc}")


def main():
    st.set_page_config(page_title="Dashboard Integrador de Metodos", layout="wide")

    st.title("Dashboard Integrador de Metodos Numericos")
    st.caption("Interfaz grafica con formularios, botones, tablas y graficos de funcion/error")

    tabs = st.tabs(
        [
            "Newton-Raphson",
            "Aitken",
            "Biseccion",
            "Punto Fijo",
            "Comparativa",
            "Lagrange + Derivacion",
        ]
    )

    with tabs[0]:
        section_newton()
    with tabs[1]:
        section_aitken()
    with tabs[2]:
        section_biseccion()
    with tabs[3]:
        section_punto_fijo()
    with tabs[4]:
        section_comparativa()
    with tabs[5]:
        section_lagrange()


def _is_streamlit_context():
    """Detecta si el script se ejecuta con `streamlit run`."""
    try:
        return st.runtime.exists()
    except Exception:
        return False


if __name__ == "__main__":
    if _is_streamlit_context():
        main()
    else:
        print("Este dashboard debe ejecutarse con Streamlit para evitar errores de contexto.")
        print("Usa este comando desde la carpeta del proyecto:")
        print("streamlit run dashboard_integrador.py")
        sys.exit(0)
