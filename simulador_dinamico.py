"""
Simulador Interactivo de Sistemas Dinámicos
Materia: Modelado y Simulación
"""
import streamlit as st
import numpy as np
from sim_engine import (find_equilibria_1d, classify_equilibrium_1d, find_equilibria_2d,
    classify_equilibrium_2d, solve_1d, solve_2d, solve_3d, compute_bifurcation_diagram)
from sim_presets import PRESETS_1D, PRESETS_2D, PRESETS_3D
from sim_viz import (plot_1d_phase, plot_time_series_1d, plot_bifurcation,
    plot_2d_phase, plot_time_series_2d, plot_3d_trajectory, plot_3d_time_series)

st.set_page_config(page_title="Simulador Dinámico", page_icon="🌀", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(135deg, #0E1117 0%, #1a1f2e 50%, #0E1117 100%); }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(108,99,255,0.1); border-radius: 8px; padding: 8px 20px;
    border: 1px solid rgba(108,99,255,0.2); color: #FAFAFA;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6C63FF, #00D2FF); border: none; font-weight: 600;
}
div[data-testid="stExpander"] { background: rgba(30,36,51,0.6); border: 1px solid rgba(108,99,255,0.15); border-radius: 12px; }
.eq-card { background: linear-gradient(135deg, rgba(30,36,51,0.8), rgba(20,24,33,0.9));
    border: 1px solid rgba(108,99,255,0.2); border-radius: 12px; padding: 16px; margin: 8px 0; }
.hero-title { background: linear-gradient(135deg, #6C63FF, #00D2FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.2rem; font-weight: 700; text-align: center; margin-bottom: 0; }
.hero-sub { color: #8892a4; text-align: center; font-size: 1rem; margin-bottom: 2rem; }
</style>""", unsafe_allow_html=True)

st.markdown('<p class="hero-title">🌀 Simulador de Sistemas Dinámicos</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Modelado y Simulación — Análisis interactivo de sistemas dinámicos 1D, 2D y 3D</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Sistemas 1D", "🌊 Sistemas 2D", "🦋 Sistemas 3D / Caos"])

# ════════════════════════════ TAB 1D ════════════════════════════
with tab1:
    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.markdown("### ⚙️ Configuración")
        mode_1d = st.radio("Modo", ["Preset", "Personalizado"], horizontal=True, key="m1d")
        if mode_1d == "Preset":
            preset_name = st.selectbox("Preset", list(PRESETS_1D.keys()), key="p1d")
            p = PRESETS_1D[preset_name]
            eq_str = st.text_input("ẋ = f(x)", p["equation"], key="eq1d")
            params = {}
            for pname, pconf in p["params"].items():
                params[pname] = st.slider(pname, pconf["min"], pconf["max"], pconf["default"], pconf["step"], key=f"s1d_{pname}")
            x0 = st.number_input("x₀", value=p["x0"], key="x01d")
            t_end = st.slider("T final", 1.0, 100.0, float(p["t_span"][1]), 1.0, key="t1d")
            xr = p["x_range"]
        else:
            eq_str = st.text_input("ẋ = f(x)", "r*x - x**3", key="eq1dc")
            st.caption("Usa variables: `x`. Parámetros: cualquier letra.")
            n_params = st.number_input("Nº parámetros", 0, 5, 1, key="np1d")
            params = {}
            for i in range(int(n_params)):
                pc1, pc2 = st.columns(2)
                nm = pc1.text_input(f"Nombre {i+1}", chr(114+i), key=f"pn1d{i}")
                vl = pc2.number_input(f"Valor", value=1.0, key=f"pv1d{i}")
                params[nm] = vl
            x0 = st.number_input("x₀", value=0.5, key="x01dc")
            t_end = st.slider("T final", 1.0, 100.0, 20.0, 1.0, key="t1dc")
            xr = (st.number_input("x min", value=-5.0, key="xmin1d"), st.number_input("x max", value=5.0, key="xmax1d"))
            p = None

        run_1d = st.button("▶ Simular", key="run1d", use_container_width=True, type="primary")
        bif_1d = st.button("📊 Diagrama de Bifurcación", key="bif1d", use_container_width=True)

    with c2:
        if mode_1d == "Preset" and p and p.get("theory"):
            with st.expander("📖 Marco Teórico", expanded=False):
                st.markdown(p["theory"])

        if run_1d or "res1d" not in st.session_state:
            eqs = find_equilibria_1d(eq_str, params, xr)
            clss = [classify_equilibrium_1d(eq_str, params, eq) for eq in eqs]
            t, x = solve_1d(eq_str, params, x0, (0, t_end))
            st.session_state.res1d = (eqs, clss, t, x)

        if "res1d" in st.session_state:
            eqs, clss, t, x = st.session_state.res1d
            if eqs:
                st.markdown("#### 🎯 Puntos de Equilibrio")
                cols = st.columns(min(len(eqs), 4))
                for i, (eq, (cls, df)) in enumerate(zip(eqs, clss)):
                    with cols[i % len(cols)]:
                        st.markdown(f'<div class="eq-card"><b>x* = {eq:.4f}</b><br>{cls}<br>f\'(x*) = {df:.4f}</div>', unsafe_allow_html=True)
            fig1 = plot_1d_phase(eq_str, params, xr, eqs, clss)
            st.plotly_chart(fig1, use_container_width=True)
            fig2 = plot_time_series_1d(t, x)
            st.plotly_chart(fig2, use_container_width=True)

        if bif_1d:
            if not params:
                st.warning("⚠️ Necesitás al menos un parámetro para el diagrama de bifurcación.")
            else:
                bp = p.get("bif_param", list(params.keys())[0]) if p else list(params.keys())[0]
                br = p.get("bif_range", (-5, 5)) if p else (-5, 5)
                other = {k: v for k, v in params.items() if k != bp}
                with st.spinner("Calculando diagrama de bifurcación..."):
                    try:
                        sp_, sx, up, ux = compute_bifurcation_diagram(eq_str, bp, br, other, xr)
                        st.session_state.bif1d = (sp_, sx, up, ux, bp)
                    except Exception as e:
                        st.error(f"Error al calcular bifurcación: {e}")

        if "bif1d" in st.session_state:
            sp_, sx, up, ux, bp = st.session_state.bif1d
            if sp_ or up:
                fig = plot_bifurcation(sp_, sx, up, ux, bp)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No se encontraron puntos de equilibrio en el rango. Verificá que la ecuación use el parámetro correctamente.")

# ════════════════════════════ TAB 2D ════════════════════════════
with tab2:
    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.markdown("### ⚙️ Configuración")
        mode_2d = st.radio("Modo", ["Preset", "Personalizado"], horizontal=True, key="m2d")
        if mode_2d == "Preset":
            pname2 = st.selectbox("Preset", list(PRESETS_2D.keys()), key="p2d")
            p2 = PRESETS_2D[pname2]
            fx_str = st.text_input("ẋ = f(x,y)", p2["fx"], key="fx2d")
            fy_str = st.text_input("ẏ = g(x,y)", p2["fy"], key="fy2d")
            params2 = {}
            for pn, pc in p2["params"].items():
                params2[pn] = st.slider(pn, pc["min"], pc["max"], pc["default"], pc["step"], key=f"s2d_{pn}")
            x0_2 = st.number_input("x₀", value=p2["x0"], key="x02d")
            y0_2 = st.number_input("y₀", value=p2["y0"], key="y02d")
            t_end2 = st.slider("T final", 1.0, 100.0, float(p2["t_span"][1]), 1.0, key="t2d")
            xr2, yr2 = p2["x_range"], p2["y_range"]
            labels2 = p2.get("labels", ("x", "y"))
        else:
            fx_str = st.text_input("ẋ = f(x,y)", "y", key="fx2dc")
            fy_str = st.text_input("ẏ = g(x,y)", "-x - 0.5*y", key="fy2dc")
            n_p2 = st.number_input("Nº parámetros", 0, 5, 0, key="np2d")
            params2 = {}
            for i in range(int(n_p2)):
                pc1, pc2 = st.columns(2)
                nm = pc1.text_input(f"Nombre {i+1}", key=f"pn2d{i}")
                vl = pc2.number_input(f"Valor", value=1.0, key=f"pv2d{i}")
                if nm: params2[nm] = vl
            x0_2 = st.number_input("x₀", value=2.0, key="x02dc")
            y0_2 = st.number_input("y₀", value=0.0, key="y02dc")
            t_end2 = st.slider("T final", 1.0, 100.0, 20.0, 1.0, key="t2dc")
            xr2 = (st.number_input("x min", value=-5.0, key="xmin2d"), st.number_input("x max", value=5.0, key="xmax2d"))
            yr2 = (st.number_input("y min", value=-5.0, key="ymin2d"), st.number_input("y max", value=5.0, key="ymax2d"))
            labels2 = ("x", "y")
            p2 = None

        n_traj = st.number_input("Trayectorias extra", 0, 5, 0, key="nt2d")
        extra_ics = []
        for i in range(int(n_traj)):
            ec1, ec2 = st.columns(2)
            ex = ec1.number_input(f"x₀ #{i+2}", value=float(np.random.uniform(*xr2)), key=f"ex2d{i}")
            ey = ec2.number_input(f"y₀ #{i+2}", value=float(np.random.uniform(*yr2)), key=f"ey2d{i}")
            extra_ics.append((ex, ey))

        run_2d = st.button("▶ Simular", key="run2d", use_container_width=True, type="primary")

    with c2:
        if mode_2d == "Preset" and p2 and p2.get("theory"):
            with st.expander("📖 Marco Teórico", expanded=False):
                st.markdown(p2["theory"])

        if run_2d:
            eqs2 = find_equilibria_2d(fx_str, fy_str, params2,
                (min(xr2[0], yr2[0]), max(xr2[1], yr2[1])))
            clss2 = [classify_equilibrium_2d(fx_str, fy_str, params2, eq) for eq in eqs2]
            trajectories = []
            all_ics = [(x0_2, y0_2)] + extra_ics
            for ix, iy in all_ics:
                t, x, y = solve_2d(fx_str, fy_str, params2, ix, iy, (0, t_end2))
                trajectories.append((t, x, y))
            st.session_state.res2d = (eqs2, clss2, trajectories)

        if "res2d" in st.session_state:
            eqs2, clss2, trajectories = st.session_state.res2d
            if eqs2:
                st.markdown("#### 🎯 Puntos de Equilibrio")
                for (ex, ey), (cls, eigs, J, tr, det) in zip(eqs2, clss2):
                    with st.expander(f"({ex:.3f}, {ey:.3f}) — {cls}"):
                        c_a, c_b = st.columns(2)
                        c_a.latex(rf"\lambda_1 = {eigs[0]:.4f}")
                        c_b.latex(rf"\lambda_2 = {eigs[1]:.4f}")
                        st.latex(rf"\text{{tr}}(J) = {tr:.4f}, \quad \det(J) = {det:.4f}")
                        st.latex(rf"J = \begin{{pmatrix}} {J[0,0]:.3f} & {J[0,1]:.3f} \\ {J[1,0]:.3f} & {J[1,1]:.3f} \end{{pmatrix}}")

            fig_phase = plot_2d_phase(fx_str, fy_str, params2, xr2, yr2, trajectories, eqs2, clss2, labels2)
            st.plotly_chart(fig_phase, use_container_width=True)
            t0, x0s, y0s = trajectories[0]
            fig_ts = plot_time_series_2d(t0, x0s, y0s, labels2)
            st.plotly_chart(fig_ts, use_container_width=True)

# ════════════════════════════ TAB 3D ════════════════════════════
with tab3:
    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.markdown("### ⚙️ Configuración")
        mode_3d = st.radio("Modo", ["Preset", "Personalizado"], horizontal=True, key="m3d")
        if mode_3d == "Preset":
            pname3 = st.selectbox("Preset", list(PRESETS_3D.keys()), key="p3d")
            p3 = PRESETS_3D[pname3]
            fx3 = st.text_input("ẋ", p3["fx"], key="fx3d")
            fy3 = st.text_input("ẏ", p3["fy"], key="fy3d")
            fz3 = st.text_input("ż", p3["fz"], key="fz3d")
            params3 = {}
            for pn, pc in p3["params"].items():
                params3[pn] = st.slider(pn, pc["min"], pc["max"], pc["default"], pc["step"], key=f"s3d_{pn}")
            x0_3 = st.number_input("x₀", value=p3["x0"], key="x03d")
            y0_3 = st.number_input("y₀", value=p3["y0"], key="y03d")
            z0_3 = st.number_input("z₀", value=p3["z0"], key="z03d")
            t_end3 = st.slider("T final", 1.0, 100.0, float(p3["t_span"][1]), 1.0, key="t3d")
        else:
            fx3 = st.text_input("ẋ", "sigma*(y-x)", key="fx3dc")
            fy3 = st.text_input("ẏ", "x*(rho-z)-y", key="fy3dc")
            fz3 = st.text_input("ż", "x*y-beta*z", key="fz3dc")
            n_p3 = st.number_input("Nº parámetros", 0, 6, 3, key="np3d")
            params3 = {}
            defaults3 = [("sigma",10),("rho",28),("beta",2.667)]
            for i in range(int(n_p3)):
                pc1, pc2 = st.columns(2)
                dn, dv = defaults3[i] if i < len(defaults3) else (f"p{i}", 1.0)
                nm = pc1.text_input(f"Nombre {i+1}", dn, key=f"pn3d{i}")
                vl = pc2.number_input(f"Valor", value=dv, key=f"pv3d{i}")
                params3[nm] = vl
            x0_3 = st.number_input("x₀", value=1.0, key="x03dc")
            y0_3 = st.number_input("y₀", value=1.0, key="y03dc")
            z0_3 = st.number_input("z₀", value=1.0, key="z03dc")
            t_end3 = st.slider("T final", 1.0, 200.0, 50.0, 1.0, key="t3dc")
            p3 = None

        butterfly = st.checkbox("🦋 Comparar efecto mariposa", key="but3d")
        if butterfly:
            eps = st.number_input("Perturbación ε", value=1e-6, format="%.1e", key="eps3d")
        run_3d = st.button("▶ Simular", key="run3d", use_container_width=True, type="primary")

    with c2:
        if mode_3d == "Preset" and p3 and p3.get("theory"):
            with st.expander("📖 Marco Teórico", expanded=False):
                st.markdown(p3["theory"])

        if run_3d:
            with st.spinner("Integrando sistema 3D con RK45..."):
                t, x, y, z = solve_3d(fx3, fy3, fz3, params3, x0_3, y0_3, z0_3, (0, t_end3))
            fig3d = plot_3d_trajectory(t, x, y, z, "Trayectoria 3D")
            st.plotly_chart(fig3d, use_container_width=True)
            fig_ts3 = plot_3d_time_series(t, x, y, z)
            st.plotly_chart(fig_ts3, use_container_width=True)

            if butterfly:
                t2, x2, y2, z2 = solve_3d(fx3, fy3, fz3, params3, x0_3+eps, y0_3, z0_3, (0, t_end3))
                import plotly.graph_objects as go
                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(x=t, y=x, mode="lines", name="Original", line=dict(color="#6C63FF", width=2)))
                fig_b.add_trace(go.Scatter(x=t2, y=x2, mode="lines", name=f"Perturbada (ε={eps})", line=dict(color="#FF6584", width=2)))
                diff = np.abs(np.interp(t, t2, x2) - x)
                fig_b.add_trace(go.Scatter(x=t, y=diff, mode="lines", name="|Δx(t)|", line=dict(color="#FFD740", width=2, dash="dot"), yaxis="y2"))
                fig_b.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#141821",
                    title="🦋 Efecto Mariposa: Sensibilidad a Condiciones Iniciales",
                    xaxis_title="t", yaxis_title="x(t)",
                    yaxis2=dict(title="|Δx|", overlaying="y", side="right", showgrid=False),
                    height=450, font=dict(family="Inter", color="#FAFAFA"))
                st.plotly_chart(fig_b, use_container_width=True)
                st.info(f"Las trayectorias divergen exponencialmente a pesar de diferir solo en ε = {eps}. Esto es el **caos determinista**.")

# Footer
st.markdown("---")
st.markdown('<p style="text-align:center;color:#8892a4;font-size:0.85rem;">Simulador de Sistemas Dinámicos — Modelado y Simulación © 2026</p>', unsafe_allow_html=True)
