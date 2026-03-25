# Modelado_Simulacion
Scripts de modelado y simulacion
(Desarrollo del TPO)

librerias a intalar:

pip install streamlit numpy matplotlib sympy
pip install pandas openpyxl

si te salta un error con pandas:
python -m pip install pandas openpyxl

Nueva funcionalidad agregada en el integrador:
- Interpolación con polinomio de base de Lagrange.
- Derivación por aproximación con 3 formas: adelante, atrás y centrada.
- Diferencias divididas (tabla) y construcción del polinomio de Newton.
- Menú interactivo para elegir cada operación.

Dashboard grafico (Streamlit):
- Archivo: dashboard_integrador.py
- Ejecutar desde la carpeta del proyecto:

streamlit run dashboard_integrador.py

Incluye:
- Formularios con botones para Newton-Raphson, Aitken, Biseccion y Punto Fijo.
- Comparativa de metodos con graficos de iteraciones y error final.
- Modulo de Lagrange con interpolacion, derivacion aproximada y error contra derivada real.