import math
import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    standard_transformations,
    parse_expr,
)


_LOCALS = {
    "e": sp.E,
    "pi": sp.pi,
    "sin": sp.sin,
    "sen": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "cbrt": lambda x: sp.real_root(x, 3),
    "abs": sp.Abs,
}


def parse_real(texto: str, nombre: str = "valor") -> float:
    texto = texto.strip()
    if not texto:
        raise ValueError(f"{nombre} no puede estar vacío")

    expr = sp.sympify(texto, locals=_LOCALS)
    if expr.free_symbols:
        raise ValueError(f"{nombre} no debe contener variables")

    valor = float(sp.N(expr))
    if not math.isfinite(valor):
        raise ValueError(f"{nombre} no es finito")
    return valor


def parse_real_or_default(texto: str, default: float, nombre: str = "valor") -> float:
    texto = texto.strip()
    if not texto:
        return default
    return parse_real(texto, nombre)


def parse_int_or_default(texto: str, default: int, nombre: str = "valor") -> int:
    texto = texto.strip()
    if not texto:
        return default

    valor = parse_real(texto, nombre)
    entero = round(valor)
    if abs(valor - entero) > 1e-12:
        raise ValueError(f"{nombre} debe ser un entero")
    return int(entero)


def parse_function_expression(funcion: str, variable: str = "x") -> sp.Expr:
    """Parsea una función f(x) con soporte de pi/e y funciones matemáticas."""
    texto = funcion.strip()
    if not texto:
        raise ValueError("La función no puede estar vacía")

    # Corrige casos como 0.4e**x para evitar errores de sintaxis.
    texto = re.sub(r"(?P<num>\d)(?P<e>e\*\*)", r"\g<num>*\g<e>", texto)

    x = sp.Symbol(variable)
    transformations = standard_transformations + (convert_xor, implicit_multiplication_application)
    expr = parse_expr(texto, local_dict={**_LOCALS, variable: x}, transformations=transformations)
    simbolos_invalidos = expr.free_symbols - {x}
    if simbolos_invalidos:
        raise ValueError(f"Se encontraron variables no permitidas: {simbolos_invalidos}")
    return expr


def build_numeric_function(funcion: str, variable: str = "x"):
    """Construye (expresion_simbolica, funcion_numerica_numpy) desde una función de usuario."""
    x = sp.Symbol(variable)
    expr = parse_function_expression(funcion, variable)
    return expr, sp.lambdify(x, expr, "numpy")
