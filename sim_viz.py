"""Módulo de visualización Plotly para el simulador."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "primary": "#6C63FF", "secondary": "#FF6584", "accent": "#00D2FF",
    "stable": "#00E676", "unstable": "#FF5252", "saddle": "#FFD740",
    "center": "#40C4FF", "bg": "#0E1117", "grid": "#1E2433",
    "text": "#FAFAFA", "semi": "#FF9800",
}
LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["bg"], plot_bgcolor="#141821",
    font=dict(family="Inter, sans-serif", color=COLORS["text"]),
    margin=dict(l=50, r=30, t=50, b=50),
)

def _stability_color(cls):
    c = cls.lower()
    if "estable" in c and "inestable" not in c: return COLORS["stable"]
    if "inestable" in c or "fuente" in c or "repulsor" in c: return COLORS["unstable"]
    if "silla" in c: return COLORS["saddle"]
    if "centro" in c: return COLORS["center"]
    if "semi" in c: return COLORS["semi"]
    return COLORS["text"]

def _icon(cls):
    c = cls.lower()
    if "estable" in c and "inestable" not in c: return "🟢"
    if "inestable" in c or "fuente" in c: return "🔴"
    if "silla" in c: return "🔶"
    if "centro" in c: return "🔵"
    if "semi" in c: return "🟠"
    return "⚪"

def plot_1d_phase(expr_str, params, x_range, equilibria, classifications):
    from sim_engine import parse_expr
    f = parse_expr(expr_str, ["x"], params)
    x = np.linspace(x_range[0], x_range[1], 500)
    y = np.array([f(xi) for xi in x])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="ẋ = f(x)",
        line=dict(color=COLORS["primary"], width=3)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    for eq, (cls, df_val) in zip(equilibria, classifications):
        col = _stability_color(cls)
        icon = _icon(cls)
        fig.add_trace(go.Scatter(x=[eq], y=[0], mode="markers+text",
            marker=dict(size=14, color=col, line=dict(width=2, color="white")),
            text=[f"{icon} x*={eq:.2f}"], textposition="top center",
            name=f"{cls}: x*={eq:.2f}", textfont=dict(size=11)))
    # Arrows
    for i in range(0, len(x)-1, 20):
        if abs(y[i]) > 0.01:
            color = COLORS["stable"] if y[i] < 0 else COLORS["unstable"]
            fig.add_annotation(x=x[i], y=0, ax=x[i]+np.sign(y[i])*0.3, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor=color, opacity=0.6)
    fig.update_layout(**LAYOUT, title="Diagrama de Fase 1D: ẋ vs x",
        xaxis_title="x", yaxis_title="ẋ = f(x)", height=450)
    return fig

def plot_time_series_1d(t, x):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, mode="lines", name="x(t)",
        line=dict(color=COLORS["accent"], width=2.5)))
    fig.update_layout(**LAYOUT, title="Solución Temporal x(t)",
        xaxis_title="Tiempo t", yaxis_title="x(t)", height=400)
    return fig

def plot_bifurcation(s_p, s_x, u_p, u_x, param_name):
    fig = go.Figure()
    if s_p:
        fig.add_trace(go.Scatter(x=s_p, y=s_x, mode="markers",
            marker=dict(size=3, color=COLORS["stable"]), name="Estable"))
    if u_p:
        fig.add_trace(go.Scatter(x=u_p, y=u_x, mode="markers",
            marker=dict(size=3, color=COLORS["unstable"]), name="Inestable"))
    fig.update_layout(**LAYOUT, title=f"Diagrama de Bifurcación (param: {param_name})",
        xaxis_title=param_name, yaxis_title="x*", height=450)
    return fig

def plot_2d_phase(fx_str, fy_str, params, x_range, y_range, trajectories, equilibria, classifications, labels=("x","y")):
    from sim_engine import compute_vector_field
    fig = go.Figure()
    X, Y, U, V = compute_vector_field(fx_str, fy_str, params, x_range, y_range, 20)
    mag = np.sqrt(U**2 + V**2)
    mag[mag == 0] = 1
    Un, Vn = U/mag, V/mag
    scale = (x_range[1]-x_range[0]) / 25
    for i in range(0, X.shape[0], 1):
        for j in range(0, X.shape[1], 1):
            fig.add_annotation(x=X[i,j]+Un[i,j]*scale, y=Y[i,j]+Vn[i,j]*scale,
                ax=X[i,j], ay=Y[i,j], xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2,
                arrowcolor=f"rgba(108,99,255,{min(0.15+float(mag[i,j])/(mag.max()+1e-9)*0.5, 0.7):.2f})")
    colors_traj = [COLORS["accent"], COLORS["secondary"], "#FFD740", "#00E676", "#E040FB"]
    for idx, (t, x, y) in enumerate(trajectories):
        c = colors_traj[idx % len(colors_traj)]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"Trayectoria {idx+1}",
            line=dict(color=c, width=2), opacity=0.9))
        fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers",
            marker=dict(size=8, color=c, symbol="diamond"), showlegend=False))
    for (ex, ey), (cls, eigs, J, tr, det) in zip(equilibria, classifications):
        col = _stability_color(cls)
        icon = _icon(cls)
        fig.add_trace(go.Scatter(x=[ex], y=[ey], mode="markers+text",
            marker=dict(size=14, color=col, line=dict(width=2, color="white")),
            text=[f"{icon} ({ex:.1f},{ey:.1f})"], textposition="top right",
            name=f"{cls}", textfont=dict(size=10, color=col)))
    fig.update_layout(**LAYOUT, title="Retrato de Fase 2D",
        xaxis_title=labels[0], yaxis_title=labels[1], height=550,
        xaxis=dict(range=x_range), yaxis=dict(range=y_range, scaleanchor="x"))
    return fig

def plot_time_series_2d(t, x, y, labels=("x","y")):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, mode="lines", name=labels[0],
        line=dict(color=COLORS["primary"], width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=labels[1],
        line=dict(color=COLORS["secondary"], width=2.5)))
    fig.update_layout(**LAYOUT, title="Series Temporales",
        xaxis_title="t", yaxis_title="Valor", height=400)
    return fig

def plot_3d_trajectory(t, x, y, z, title="Trayectoria 3D"):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines",
        line=dict(color=t, colorscale="Plasma", width=3),
        name="Trayectoria"))
    fig.add_trace(go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode="markers",
        marker=dict(size=5, color=COLORS["stable"]), name="Inicio"))
    fig.update_layout(**LAYOUT, title=title, height=600,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            bgcolor="#141821"))
    return fig

def plot_3d_time_series(t, x, y, z):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("x(t)", "y(t)", "z(t)"))
    for i, (data, color, name) in enumerate([(x, COLORS["primary"], "x"),
            (y, COLORS["secondary"], "y"), (z, COLORS["accent"], "z")]):
        fig.add_trace(go.Scatter(x=t, y=data, mode="lines", name=name,
            line=dict(color=color, width=2)), row=i+1, col=1)
    fig.update_layout(**LAYOUT, height=600, title="Series Temporales 3D")
    return fig
