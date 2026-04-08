"""Interfaz Streamlit para simulación de penales - Híbrida Manual/Predefinida."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from penales_montecarlo import SimuladorPenales, JugadorPenal, PorteroPenal
from utilidades_penales import GeneradorDatos

st.set_page_config(page_title="⚽ Simulador Penales", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&display=swap');
    * { font-family: 'Poppins', sans-serif; }
    [data-testid="stSidebar"] { min-width: 500px !important; width: 500px !important; }
    [data-testid="stSidebar"] > div:first-child { width: 500px !important; }
    .main { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
    h1, h2, h3 { color: #f39c12; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
    .metric-card {
        background: linear-gradient(135deg, #0f3460 0%, #533483 100%);
        padding: 20px; border-radius: 12px; border-left: 5px solid #f39c12;
        color: white; margin: 10px 0; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .stat-box {
        background: linear-gradient(135deg, #16213e, #0f3460);
        padding: 15px; border-radius: 8px; border: 1px solid #f39c12;
        color: white; margin: 8px 0;
    }
    .equipo-banner {
        background: linear-gradient(90deg, #f39c12 0%, #e67e22 100%);
        color: white; padding: 15px; border-radius: 10px;
        font-weight: 600; text-align: center; margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# INIT SESSION STATE
if 'resultados1' not in st.session_state:
    st.session_state.resultados1 = None
if 'resultados2' not in st.session_state:
    st.session_state.resultados2 = None
if 'team1' not in st.session_state:
    st.session_state.team1 = None
if 'team2' not in st.session_state:
    st.session_state.team2 = None
if 'portero1' not in st.session_state:
    st.session_state.portero1 = None
if 'portero2' not in st.session_state:
    st.session_state.portero2 = None
if 'modo_manual' not in st.session_state:
    st.session_state.modo_manual = False
if 'modo_mixto' not in st.session_state:
    st.session_state.modo_mixto = False

# SIDEBAR - CONFIGURACIÓN EQUIPO 1
with st.sidebar:
    st.markdown("# ⚙️ EQUIPO 1")
    st.markdown("---")
    
    modo1 = st.radio("Modo Team 1", ["Predefinido", "Manual"], key="modo1")
    
    if modo1 == "Predefinido":
        eq1 = st.selectbox("Selecciona Team 1", ["Argentina", "Francia"], key="eq1")
        if eq1 == "Argentina":
            st.session_state.team1 = GeneradorDatos.obtener_equipo_argentina()[:5]
            st.session_state.portero1 = GeneradorDatos.obtener_portero_argentina()
        else:
            st.session_state.team1 = GeneradorDatos.obtener_equipo_francia()[:5]
            st.session_state.portero1 = GeneradorDatos.obtener_portero_francia()
        st.success(f"✓ {eq1}: {len(st.session_state.team1)} jugadores")
    
    else:
        st.session_state.team1 = []
        for i in range(5):
            with st.expander(f"J{i+1}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    nom = st.text_input(f"Nombre J{i+1}", f"J{i+1}", key=f"n1_{i}")
                    pes = st.number_input(f"Peso J{i+1}", 60.0, 110.0, 75.0, key=f"p1_{i}")
                    alt = st.number_input(f"Alt J{i+1}", 160, 200, 180, key=f"a1_{i}")
                with col2:
                    prec = st.slider(f"Prec J{i+1} %", 0, 100, 75, key=f"pr1_{i}")
                    vel = st.slider(f"Vel J{i+1} km/h", 80, 130, 105, key=f"v1_{i}")
                    exp = st.number_input(f"Exp J{i+1}", 0, 30, 5, key=f"e1_{i}")
                st.session_state.team1.append(JugadorPenal(nom, pes, alt, prec, vel, exp))
        
        with st.expander("🥅 Portero 1", expanded=False):
            nom_p1 = st.text_input("Portero 1", "Portero1", key="nom_p1")
            pes_p1 = st.number_input("Peso P1", 80.0, 100.0, 85.0, key="pes_p1")
            alt_p1 = st.number_input("Alt P1", 180, 210, 190, key="alt_p1")
            ref_p1 = st.slider("Reflejos P1 %", 0, 100, 85, key="ref_p1")
            exp_p1 = st.number_input("Exp P1", 0, 30, 10, key="exp_p1")
            st.session_state.portero1 = PorteroPenal(nom_p1, pes_p1, alt_p1, ref_p1, exp_p1)

st.sidebar.markdown("---")

# SIDEBAR - CONFIGURACIÓN EQUIPO 2
with st.sidebar:
    st.markdown("# ⚙️ EQUIPO 2")
    
    modo2 = st.radio("Modo Team 2", ["Predefinido", "Manual"], key="modo2")
    
    if modo2 == "Predefinido":
        eq2 = st.selectbox("Selecciona Team 2", ["Argentina", "Francia"], key="eq2")
        if eq2 == "Argentina":
            st.session_state.team2 = GeneradorDatos.obtener_equipo_argentina()[:5]
            st.session_state.portero2 = GeneradorDatos.obtener_portero_argentina()
        else:
            st.session_state.team2 = GeneradorDatos.obtener_equipo_francia()[:5]
            st.session_state.portero2 = GeneradorDatos.obtener_portero_francia()
        st.success(f"✓ {eq2}: {len(st.session_state.team2)} jugadores")
    
    else:
        st.session_state.team2 = []
        for i in range(5):
            with st.expander(f"J{i+1}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    nom = st.text_input(f"Nombre J{i+1}", f"J{i+1}", key=f"n2_{i}")
                    pes = st.number_input(f"Peso J{i+1}", 60.0, 110.0, 75.0, key=f"p2_{i}")
                    alt = st.number_input(f"Alt J{i+1}", 160, 200, 180, key=f"a2_{i}")
                with col2:
                    prec = st.slider(f"Prec J{i+1} %", 0, 100, 75, key=f"pr2_{i}")
                    vel = st.slider(f"Vel J{i+1} km/h", 80, 130, 105, key=f"v2_{i}")
                    exp = st.number_input(f"Exp J{i+1}", 0, 30, 5, key=f"e2_{i}")
                st.session_state.team2.append(JugadorPenal(nom, pes, alt, prec, vel, exp))
        
        with st.expander("🥅 Portero 2", expanded=False):
            nom_p2 = st.text_input("Portero 2", "Portero2", key="nom_p2")
            pes_p2 = st.number_input("Peso P2", 80.0, 100.0, 85.0, key="pes_p2")
            alt_p2 = st.number_input("Alt P2", 180, 210, 190, key="alt_p2")
            ref_p2 = st.slider("Reflejos P2 %", 0, 100, 85, key="ref_p2")
            exp_p2 = st.number_input("Exp P2", 0, 30, 10, key="exp_p2")
            st.session_state.portero2 = PorteroPenal(nom_p2, pes_p2, alt_p2, ref_p2, exp_p2)

st.sidebar.markdown("---")

with st.sidebar:
    st.markdown("### ⚙️ SIMULACIÓN")
    num_sim = st.number_input("Simulaciones", 1000, 100000, 10000, 1000)
    usar_sem = st.checkbox("Semilla fija", False)
    semilla = st.number_input("Semilla", 42) if usar_sem else None

# TÍTULO
st.markdown("""
    <div style='text-align: center; padding: 25px; background: linear-gradient(135deg, #0f3460, #533483); 
                border-radius: 15px; margin-bottom: 20px;'>
        <h1 style='color: #f39c12; margin: 0; font-size: 48px;'>⚽ SIMULADOR PENALES</h1>
        <p style='color: white; font-size: 18px; margin: 10px 0;'>5 vs 5 - Argentina vs Francia</p>
    </div>
""", unsafe_allow_html=True)

# MODO INICIAL
st.markdown("### 🎮 SELECCIONA MODO DE JUEGO")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⚙️ Predefinidos\n(Argentina-Francia)", use_container_width=True):
        st.session_state.team1 = GeneradorDatos.obtener_equipo_argentina()[:5]
        st.session_state.portero1 = GeneradorDatos.obtener_portero_argentina()
        st.session_state.team2 = GeneradorDatos.obtener_equipo_francia()[:5]
        st.session_state.portero2 = GeneradorDatos.obtener_portero_francia()
        st.success("✓ Equipos predefinidos cargados")

with col2:
    if st.button("✏️ Manual\n(Crear equipos)", use_container_width=True):
        st.session_state.modo_manual = True

with col3:
    if st.button("🎯 Mixto\n(Predef + Manual)", use_container_width=True):
        st.session_state.modo_mixto = True

st.markdown("---")

# SIDEBAR - MANUAL O MIXTO
if st.session_state.get('modo_manual'):
    st.info("📋 Configura ambos equipos en el sidebar")
    
    with st.sidebar:
        st.markdown("# ✏️ EQUIPO 1 (MANUAL)")
        st.session_state.team1 = []
        for i in range(5):
            with st.expander(f"J{i+1}", expanded=(i==0)):
                col1, col2 = st.columns(2)
                with col1:
                    nom = st.text_input(f"Nombre J{i+1}", f"J{i+1}", key=f"nm1_{i}")
                    pes = st.number_input(f"Peso J{i+1}", 60.0, 110.0, 75.0, key=f"ps1_{i}")
                    alt = st.number_input(f"Alt J{i+1}", 160, 200, 180, key=f"al1_{i}")
                with col2:
                    prec = st.slider(f"Prec J{i+1} %", 0, 100, 75, key=f"pc1_{i}")
                    vel = st.slider(f"Vel J{i+1} km/h", 80, 130, 105, key=f"vl1_{i}")
                    exp = st.number_input(f"Exp J{i+1}", 0, 30, 5, key=f"ex1_{i}")
                st.session_state.team1.append(JugadorPenal(nom, pes, alt, prec, vel, exp))
        
        with st.expander("🥅 Portero 1", expanded=False):
            np1 = st.text_input("Nombre P1", "Portero1", key="np1")
            psp1 = st.number_input("Peso P1", 80.0, 100.0, 85.0, key="psp1")
            alp1 = st.number_input("Alt P1", 180, 210, 190, key="alp1")
            refp1 = st.slider("Reflejos P1 %", 0, 100, 85, key="refp1")
            expp1 = st.number_input("Exp P1", 0, 30, 10, key="expp1")
            st.session_state.portero1 = PorteroPenal(np1, psp1, alp1, refp1, expp1)
        
        st.markdown("---")
        st.markdown("# ✏️ EQUIPO 2 (MANUAL)")
        st.session_state.team2 = []
        for i in range(5):
            with st.expander(f"J{i+1}", expanded=(i==0)):
                col1, col2 = st.columns(2)
                with col1:
                    nom = st.text_input(f"Nombre J{i+1}", f"J{i+1}", key=f"nm2_{i}")
                    pes = st.number_input(f"Peso J{i+1}", 60.0, 110.0, 75.0, key=f"ps2_{i}")
                    alt = st.number_input(f"Alt J{i+1}", 160, 200, 180, key=f"al2_{i}")
                with col2:
                    prec = st.slider(f"Prec J{i+1} %", 0, 100, 75, key=f"pc2_{i}")
                    vel = st.slider(f"Vel J{i+1} km/h", 80, 130, 105, key=f"vl2_{i}")
                    exp = st.number_input(f"Exp J{i+1}", 0, 30, 5, key=f"ex2_{i}")
                st.session_state.team2.append(JugadorPenal(nom, pes, alt, prec, vel, exp))
        
        with st.expander("🥅 Portero 2", expanded=False):
            np2 = st.text_input("Nombre P2", "Portero2", key="np2")
            psp2 = st.number_input("Peso P2", 80.0, 100.0, 85.0, key="psp2")
            alp2 = st.number_input("Alt P2", 180, 210, 190, key="alp2")
            refp2 = st.slider("Reflejos P2 %", 0, 100, 85, key="refp2")
            expp2 = st.number_input("Exp P2", 0, 30, 10, key="expp2")
            st.session_state.portero2 = PorteroPenal(np2, psp2, alp2, refp2, expp2)

elif st.session_state.get('modo_mixto'):
    st.info("📋 Equipo 1: Predefinido | Equipo 2: Manual")
    
    with st.sidebar:
        st.markdown("# ⚙️ EQUIPO 1 (PREDEFINIDO)")
        eq1 = st.selectbox("Equipo 1", ["Argentina", "Francia"], key="eq1m")
        if eq1 == "Argentina":
            st.session_state.team1 = GeneradorDatos.obtener_equipo_argentina()[:5]
            st.session_state.portero1 = GeneradorDatos.obtener_portero_argentina()
        else:
            st.session_state.team1 = GeneradorDatos.obtener_equipo_francia()[:5]
            st.session_state.portero1 = GeneradorDatos.obtener_portero_francia()
        st.success(f"✓ {eq1}")
        
        st.markdown("---")
        st.markdown("# ✏️ EQUIPO 2 (MANUAL)")
        st.session_state.team2 = []
        for i in range(5):
            with st.expander(f"J{i+1}", expanded=(i==0)):
                col1, col2 = st.columns(2)
                with col1:
                    nom = st.text_input(f"Nombre J{i+1}", f"J{i+1}", key=f"nm_{i}")
                    pes = st.number_input(f"Peso J{i+1}", 60.0, 110.0, 75.0, key=f"ps_{i}")
                    alt = st.number_input(f"Alt J{i+1}", 160, 200, 180, key=f"al_{i}")
                with col2:
                    prec = st.slider(f"Prec J{i+1} %", 0, 100, 75, key=f"pc_{i}")
                    vel = st.slider(f"Vel J{i+1} km/h", 80, 130, 105, key=f"vl_{i}")
                    exp = st.number_input(f"Exp J{i+1}", 0, 30, 5, key=f"ex_{i}")
                st.session_state.team2.append(JugadorPenal(nom, pes, alt, prec, vel, exp))
        
        with st.expander("🥅 Portero 2", expanded=False):
            np2m = st.text_input("Nombre P2", "Portero2", key="np2m")
            psp2m = st.number_input("Peso P2", 80.0, 100.0, 85.0, key="psp2m")
            alp2m = st.number_input("Alt P2", 180, 210, 190, key="alp2m")
            refp2m = st.slider("Reflejos P2 %", 0, 100, 85, key="refp2m")
            expp2m = st.number_input("Exp P2", 0, 30, 10, key="expp2m")
            st.session_state.portero2 = PorteroPenal(np2m, psp2m, alp2m, refp2m, expp2m)

else:
    with st.sidebar:
        st.markdown("# ⚙️ SIMULACIÓN")
        num_sim = st.number_input("Simulaciones", 1000, 100000, 10000, 1000, key="numsim0")
        usar_sem = st.checkbox("Semilla fija", False, key="usesem0")
        semilla = st.number_input("Semilla", 42, key="sem0") if usar_sem else None

# CONFIG COMÚN
if st.session_state.get('modo_manual') or st.session_state.get('modo_mixto'):
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ SIMULACIÓN")
        num_sim = st.number_input("Simulaciones", 1000, 100000, 10000, 1000, key="numsim1")
        usar_sem = st.checkbox("Semilla fija", False, key="usesem1")
        semilla = st.number_input("Semilla", 42, key="sem1") if usar_sem else None
else:
    num_sim = st.session_state.get('num_sim', 10000)
    semilla = None

# EJECUTAR SIMULACIÓN
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    ejecutar = st.button("▶️ SIMULAR PENALES", use_container_width=True)

if ejecutar and st.session_state.team1 and st.session_state.team2 and st.session_state.portero1 and st.session_state.portero2:
    with st.spinner("Simulando 10,000+ tandas..."):
        sim = SimuladorPenales(semilla=semilla)
        st.session_state.resultados1 = sim.simular_tanda(st.session_state.team1, st.session_state.portero2, num_sim)
        st.session_state.resultados2 = sim.simular_tanda(st.session_state.team2, st.session_state.portero1, num_sim)
    st.balloons()
    st.success("✅ ¡Simulación completada!")

# RESULTADOS
if st.session_state.resultados1 and st.session_state.resultados2:
    res1 = st.session_state.resultados1
    res2 = st.session_state.resultados2
    
    st.markdown("### 📊 COMPARATIVA: EQUIPO 1 vs EQUIPO 2")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #0f3460, #533483); padding: 20px; border-radius: 10px;'>
        <h3 style='color: #f39c12; margin-top: 0;'>⚽ EQUIPO 1</h3>
        <p><strong>Goles esperados:</strong> {res1['goles_totales_esperado']:.2f}</p>
        <p><strong>Desv. Estándar:</strong> {res1['goles_std']:.2f}</p>
        <p><strong>Probabilidad Ganar:</strong> <span style='color: #27ae60; font-weight: bold;'>{res1['probabilidad_ganar']:.1f}%</span></p>
        <p><strong>IC 95%:</strong> [{res1['ic_lower']:.2f}, {res1['ic_upper']:.2f}]</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #533483, #0f3460); padding: 20px; border-radius: 10px;'>
        <h3 style='color: #f39c12; margin-top: 0;'>⚽ EQUIPO 2</h3>
        <p><strong>Goles esperados:</strong> {res2['goles_totales_esperado']:.2f}</p>
        <p><strong>Desv. Estándar:</strong> {res2['goles_std']:.2f}</p>
        <p><strong>Probabilidad Ganar:</strong> <span style='color: #27ae60; font-weight: bold;'>{res2['probabilidad_ganar']:.1f}%</span></p>
        <p><strong>IC 95%:</strong> [{res2['ic_lower']:.2f}, {res2['ic_upper']:.2f}]</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # GRÁFICOS COMPARATIVOS
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = go.Figure()
        goles_u1 = sorted(set(res1['distribucion_goles']))
        freq1 = [res1['distribucion_goles'].count(g) for g in goles_u1]
        fig1.add_trace(go.Bar(x=goles_u1, y=freq1, marker=dict(color='#f39c12')))
        fig1.update_layout(title="Dist. Equipo 1", xaxis_title="Goles", yaxis_title="Frecuencia", template="plotly_dark", height=350)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = go.Figure()
        goles_u2 = sorted(set(res2['distribucion_goles']))
        freq2 = [res2['distribucion_goles'].count(g) for g in goles_u2]
        fig2.add_trace(go.Bar(x=goles_u2, y=freq2, marker=dict(color='#e67e22')))
        fig2.update_layout(title="Dist. Equipo 2", xaxis_title="Goles", yaxis_title="Frecuencia", template="plotly_dark", height=350)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # PROBABILIDADES
    st.markdown("### 🏆 PROBABILIDADES DE RESULTADO")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if res1['probabilidad_ganar'] > res2['probabilidad_ganar']:
            color = "#27ae60"
            txt = "FAVORITO"
        elif res1['probabilidad_ganar'] < res2['probabilidad_ganar']:
            color = "#e74c3c"
            txt = "DESFAVORITO"
        else:
            color = "#f39c12"
            txt = "IGUAL"
        st.markdown(f"""<div style='background: {color}; padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0;'>{res1['probabilidad_ganar']:.1f}%</h3>
            <p style='margin: 5px 0;'>SE QUEDA AL EQUIPO 1</p>
            <small>{txt}</small></div>""", unsafe_allow_html=True)
    
    with col2:
        prob_empate1 = (np.array(res1['distribucion_goles']) == np.array(res2['distribucion_goles'])).sum()/len(res1['distribucion_goles'])*100
        st.markdown(f"""<div style='background: #f39c12; padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0;'>{prob_empate1:.1f}%</h3>
            <p style='margin: 5px 0;'>EMPATE</p></div>""", unsafe_allow_html=True)
    
    with col3:
        if res2['probabilidad_ganar'] > res1['probabilidad_ganar']:
            color = "#27ae60"
            txt = "FAVORITO"
        elif res2['probabilidad_ganar'] < res1['probabilidad_ganar']:
            color = "#e74c3c"
            txt = "DESFAVORITO"
        else:
            color = "#f39c12"
            txt = "IGUAL"
        st.markdown(f"""<div style='background: {color}; padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0;'>{res2['probabilidad_ganar']:.1f}%</h3>
            <p style='margin: 5px 0;'>SE QUEDA AL EQUIPO 2</p>
            <small>{txt}</small></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # TABLA JUGADORES
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Probs. Equipo 1")
        df1 = pd.DataFrame([{"Jugador": j.nombre, "Prob. Gol": f"{p:.1%}"} 
                           for j, p in zip(st.session_state.team1, [res1['prob_por_jugador'].get(j.nombre, 0) for j in st.session_state.team1])])
        st.dataframe(df1, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 🎯 Probs. Equipo 2")
        df2 = pd.DataFrame([{"Jugador": j.nombre, "Prob. Gol": f"{p:.1%}"} 
                           for j, p in zip(st.session_state.team2, [res2['prob_por_jugador'].get(j.nombre, 0) for j in st.session_state.team2])])
        st.dataframe(df2, use_container_width=True, hide_index=True)

else:
    st.info("⬆️ Configura ambos equipos y presiona SIMULAR")
