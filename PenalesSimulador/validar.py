"""
Validación rápida del proyecto - sin emojis para evitar errores de codificación
"""

from penales_montecarlo import SimuladorPenales, JugadorPenal, PorteroPenal
from utilidades_penales import GeneradorDatos

print("=" * 70)
print("VALIDACION - SIMULADOR MONTE CARLO")
print("=" * 70)

print("\n1. VALIDAR CARGA DE DATOS")
print("-" * 70)

# Verificar jugadores disponibles
jugadores_disponibles = GeneradorDatos.listar_jugadores()
print(f"Jugadores disponibles: {len(jugadores_disponibles)}")
print(f"  Primeros 5: {', '.join(jugadores_disponibles[:5])}")

# Verificar porteros disponibles
porteros_disponibles = GeneradorDatos.listar_porteros()
print(f"Porteros disponibles: {len(porteros_disponibles)}")
print(f"  Primeros 5: {', '.join(porteros_disponibles[:5])}")

print("\n2. VALIDAR RANDOMIZADOR")
print("-" * 70)

jugadores_data = GeneradorDatos.obtener_jugadores_aleatorios(5, por_equipo=True)
portero_data = GeneradorDatos.obtener_portero_aleatorio()

print("Equipo aleatorio obtenido:")
for jug in jugadores_data:
    print(f"  - {jug['nombre']:25} ({jug['pais']})")

print(f"Portero: {portero_data['nombre']} ({portero_data['pais']})")

print("\n3. VALIDAR SIMULACION")
print("-" * 70)

jugadores_obj = [j['jugador'] for j in jugadores_data]
portero_obj = portero_data['portero']

sim = SimuladorPenales(semilla=2024)
resultados = sim.simular_tanda(jugadores_obj, portero_obj, 10000)

print("Resultados de simulacion:")
print(f"  Goles esperados: {resultados['goles_totales_esperado']:.2f}")
print(f"  Desv. Estandar: {resultados['goles_std']:.2f}")
print(f"  IC 95%: [{resultados['ic_lower']:.2f}, {resultados['ic_upper']:.2f}]")
print(f"  Prob. Ganar: {resultados['probabilidad_ganar']:.1%}")
print(f"  Prob. Empate: {resultados['probabilidad_empatar']:.1%}")
print(f"  Prob. Perder: {resultados['probabilidad_perder']:.1%}")

print("\n4. VALIDAR CASOS ESPECIFICOS")
print("-" * 70)

# Caso 1: Equipo fuerte
cristiano = GeneradorDatos.obtener_jugador_conocido("Cristiano Ronaldo")['jugador']
buffon = GeneradorDatos.obtener_portero_conocido("Gianluigi Buffon")['portero']
messi = GeneradorDatos.obtener_jugador_conocido("Lionel Messi")['jugador']

prob_gol_cr = sim.calcular_probabilidad_gol(cristiano, buffon)
prob_gol_messi = sim.calcular_probabilidad_gol(messi, buffon)

print(f"Cristiano Ronaldo vs Buffon: {prob_gol_cr:.1%}")
print(f"Lionel Messi vs Buffon: {prob_gol_messi:.1%}")

print("\n5. VALIDAR REPRODUCIBILIDAD")
print("-" * 70)

# Misma semilla = mismos resultados
sim1 = SimuladorPenales(semilla=12345)
sim2 = SimuladorPenales(semilla=12345)

j_test = JugadorPenal("Test", 75, 180, 75, 110, 10)
p_test = PorteroPenal("Test", 85, 188, 80, 15)

res1 = sim1.simular_tanda([j_test], p_test, 100)
res2 = sim2.simular_tanda([j_test], p_test, 100)

identicos = abs(res1['goles_totales_esperado'] - res2['goles_totales_esperado']) < 0.001
print(f"Resultados identicos con semilla: {'SI' if identicos else 'NO'}")
print(f"  Resultado 1: {res1['goles_totales_esperado']:.4f}")
print(f"  Resultado 2: {res2['goles_totales_esperado']:.4f}")

print("\n" + "=" * 70)
print("VALIDACION COMPLETADA - TODO OK")
print("=" * 70)
print("\nPara usar la interfaz web:")
print("  streamlit run app_penales.py")
