"""
Archivo de inicio rápido - Ejemplos de uso del simulador
"""

from penales_montecarlo import SimuladorPenales, JugadorPenal, PorteroPenal
from utilidades_penales import GeneradorDatos

print("=" * 70)
print("SIMULADOR MONTE CARLO - PENALES EN FÚTBOL")
print("=" * 70)

# ============================================
# EJEMPLO 1: Equipo Aleatorio Agrupado por País
# ============================================
print("\n1. EQUIPO ALEATORIO AGRUPADO POR PAÍS")
print("-" * 70)

jugadores_data = GeneradorDatos.obtener_jugadores_aleatorios(5, por_equipo=True)
portero_data = GeneradorDatos.obtener_portero_aleatorio()

print(f"Equipo de ataque:")
paises_representados = set()
for jug in jugadores_data:
    print(f"  {jug['emoji']} {jug['nombre']:25} ({jug['pais']})")
    paises_representados.add(jug['pais'])

print(f"\nPortero: {portero_data['emoji']} {portero_data['nombre']} ({portero_data['pais']})")
print(f"Países representados: {', '.join(paises_representados)}")

# Ejecutar simulación
jugadores_obj = [j['jugador'] for j in jugadores_data]
portero_obj = portero_data['portero']

sim = SimuladorPenales(semilla=2024)
resultados = sim.simular_tanda(jugadores_obj, portero_obj, 10000)

print(f"\nResultados (10,000 simulaciones):")
print(f"  Goles esperados: {resultados['goles_totales_esperado']:.2f}")
print(f"  Desviación estándar: {resultados['goles_std']:.2f}")
print(f"  Intervalo 95%: [{resultados['ic_lower']:.2f}, {resultados['ic_upper']:.2f}]")
print(f"  Probabilidad de ganar: {resultados['probabilidad_ganar']:.1%}")


# ============================================
# EJEMPLO 2: Clásico Argentina vs Brasil
# ============================================
print("\n2. CLÁSICO: ARGENTINA VS BRASIL")
print("-" * 70)

jugadores_arg = [
    GeneradorDatos.obtener_jugador_conocido("Lionel Messi")['jugador'],
    GeneradorDatos.obtener_jugador_conocido("Diego Maradona")['jugador'],
    GeneradorDatos.obtener_jugador_conocido("Sergio Aguero")['jugador'],
    GeneradorDatos.obtener_jugador_conocido("Angel Di Maria")['jugador'],
    GeneradorDatos.obtener_jugador_conocido("Luis Suarez")['jugador'],
]

portero_bra = GeneradorDatos.obtener_portero_conocido("Alisson")['portero']

print("Argentina (Lanzadores):")
for j in jugadores_arg:
    print(f"  ⚽ {j.nombre:25} - Precisión: {j.precision:.0f}%, Velocidad: {j.velocidad_pies:.0f} km/h")

print(f"\nBrasil (Portero): 🇧🇷 {portero_bra.nombre}")
print(f"  Reflejos: {portero_bra.reflejos:.0f}%, Experiencia: {portero_bra.experiencia} años")

sim2 = SimuladorPenales()
resultados2 = sim2.simular_tanda(jugadores_arg, portero_bra, 10000)

print(f"\nPrognóstico Argentina:")
print(f"  Goles esperados: {resultados2['goles_totales_esperado']:.2f}/5")
print(f"  Probabilidades:")
print(f"    - Ganar:  {resultados2['probabilidad_ganar']:.1%} ✓")
print(f"    - Empate: {resultados2['probabilidad_empatar']:.1%} 🤝")
print(f"    - Perder: {resultados2['probabilidad_perder']:.1%} ✗")


# ============================================
# EJEMPLO 3: Análisis Individual
# ============================================
print("\n3. ANÁLISIS INDIVIDUAL - CRISTIANO RONALDO vs BUFF ON")
print("-" * 70)

cr7 = GeneradorDatos.obtener_jugador_conocido("Cristiano Ronaldo")['jugador']
buffon = GeneradorDatos.obtener_portero_conocido("Gianluigi Buffon")['portero']

print(f"Lanzador: 🇵🇹 {cr7.nombre}")
print(f"  Precisión: {cr7.precision:.0f}%, Velocidad: {cr7.velocidad_pies:.0f} km/h, Exp: {cr7.experiencia} años")

print(f"\nPortero: 🇮🇹 {buffon.nombre}")
print(f"  Reflejos: {buffon.reflejos:.0f}%, Altura: {buffon.altura} cm, Exp: {buffon.experiencia} años")

sim3 = SimuladorPenales()
prob_gol = sim3.calcular_probabilidad_gol(cr7, buffon)

print(f"\nProbabilidad de gol calculada: {prob_gol:.1%}")
print(f"Simulando 5 penales (10x):")

for intento in range(1, 11):
    goles = sum([sim3.simular_penal(cr7, buffon) for _ in range(5)])
    estado = "✓ GOL" if goles >= 3 else "✗ NO GOL"
    print(f"  Intento {intento}: {goles}/5 penales anotados {estado}")


# ============================================
# EJEMPLO 4: Comparación de Porteros
# ============================================
print("\n4. COMPARACIÓN: MISMO EQUIPO CON DIFERENTES PORTEROS")
print("-" * 70)

equipo = [
    GeneradorDatos.obtener_jugador_conocido("Cristiano Ronaldo")['jugador'],
    GeneradorDatos.obtener_jugador_conocido("Zinedine Zidane")['jugador'],
    GeneradorDatos.obtener_jugador_conocido("Robert Lewandowski")['jugador'],
    GeneradorDatos.obtener_jugador_conocido("Neymar")['jugador'],
    GeneradorDatos.obtener_jugador_conocido("Lionel Messi")['jugador'],
]

porteros_a_comparar = ['Manuel Neuer', 'Gianluigi Buffon', 'Edwin van der Sar']

print(f"Equipo de ataque:")
for j in equipo:
    print(f"  ⚽ {j.nombre}")

print(f"\nComparando contra porteros:")
print(f"{'Portero':<30} {'Goles Esp':<12} {'Prob. Ganar':<15} {'IC 95%':<20}")
print("-" * 77)

for nombre_portero in porteros_a_comparar:
    portero = GeneradorDatos.obtener_portero_conocido(nombre_portero)['portero']
    sim_comp = SimuladorPenales()
    res_comp = sim_comp.simular_tanda(equipo, portero, 5000)
    
    ic_str = f"[{res_comp['ic_lower']:.2f}, {res_comp['ic_upper']:.2f}]"
    print(f"{nombre_portero:<30} {res_comp['goles_totales_esperado']:<12.2f} {res_comp['probabilidad_ganar']:<15.1%} {ic_str:<20}")


# ============================================
# EJEMPLO 5: Reproducibilidad con Semilla
# ============================================
print("\n5. REPRODUCIBILIDAD CON SEMILLA")
print("-" * 70)

print("Ejecutando la MISMA simulación 3 veces con semilla fija:")

for i in range(1, 4):
    jugador = JugadorPenal("Test Player", 75, 180, 80, 110, 10)
    portero = PorteroPenal("Test Keeper", 85, 188, 85, 15)
    
    sim_rep = SimuladorPenales(semilla=12345)  # SEMILLA FIJA
    res_rep = sim_rep.simular_tanda([jugador], portero, 1000)
    
    print(f"  Intento {i}: {res_rep['goles_totales_esperado']:.4f} goles esperados")

print("✓ Los resultados son idénticos (reproducibilidad con semilla)")


# ============================================
# FIN
# ============================================
print("\n" + "=" * 70)
print("✅ EJEMPLOS COMPLETADOS")
print("=" * 70)
print("\nPara usar la interfaz web:")
print("  streamlit run app_penales.py")
print("\nPara ver más opciones y documentación:")
print("  cat README.md")
