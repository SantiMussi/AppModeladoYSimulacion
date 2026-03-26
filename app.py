import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import re

# Configuracion de la pagina
st.set_page_config(page_title="Santiago Mussi | Numeric Solver", layout="wide")

st.title("Analizador de Metodos Numericos")
st.markdown("---")

# --- FUNCIONES DE EVALUACION Y DERIVADA ---

def evaluar_f(f_str, x_val):
    try:
        f_proc = f_str.replace("^", "**")
        f_proc = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', f_proc)
        contexto = {
            "np": np, "x": x_val, "sin": np.sin, "cos": np.cos, 
            "tan": np.tan, "exp": np.exp, "log": np.log, 
            "sqrt": np.sqrt, "pi": np.pi, "e": np.e
        }
        return eval(f_proc, {"__builtins__": None}, contexto)
    except:
        return None

def calcular_derivada_robusta(f_str, x_val, h=1e-5):
    f = lambda x: evaluar_f(f_str, x)
    num = -f(x_val + 2*h) + 8*f(x_val + h) - 8*f(x_val - h) + f(x_val - 2*h)
    den = 12 * h
    return num / den

def calcular_error_relativo(x_nuevo, x_anterior):
    if abs(x_nuevo) < 1e-18: return 100.0
    return (abs(x_nuevo - x_anterior) / abs(x_nuevo)) * 100

# --- LOGICA DE LOS NUEVOS METODOS (LAGRANGE Y DIFERENCIAS) ---

def metodo_lagrange(x_points, y_points, x_eval):
    n = len(x_points)
    resultado = 0
    for i in range(n):
        li = 1
        for j in range(n):
            if i != j:
                li *= (x_eval - x_points[j]) / (x_points[i] - x_points[j])
        resultado += y_points[i] * li
    return resultado

def metodo_diferencias_centrales(x_points, y_points):
    # Asumimos h constante basado en los dos primeros puntos
    h = x_points[1] - x_points[0]
    derivadas = []
    # Solo podemos calcular diferencias centrales para puntos que tengan vecinos
    for i in range(1, len(x_points) - 1):
        d1 = (y_points[i+1] - y_points[i-1]) / (2 * h)
        d2 = (y_points[i+1] - 2*y_points[i] + y_points[i-1]) / (h**2)
        derivadas.append({"Punto x": x_points[i], "f'(x) (Vel)": d1, "f''(x) (Ace)": d2})
    return pd.DataFrame(derivadas)

# --- LOGICA DE LOS METODOS ANTERIORES ---

def metodo_biseccion(f_str, a, b, tol, max_iter):
    history = []
    fa, fb = evaluar_f(f_str, a), evaluar_f(f_str, b)
    if fa is None or fb is None or fa * fb >= 0:
        return None, "error_signos", 0, 100
    x_ant = a
    for i in range(max_iter):
        c = (a + b) / 2
        fc = evaluar_f(f_str, c)
        error_p = calcular_error_relativo(c, x_ant)
        history.append({"Iter": i+1, "a": a, "b": b, "x_n (c)": c, "f(c)": fc, "Error (%)": error_p})
        if abs(fc) < 1e-15 or error_p < tol:
            return pd.DataFrame(history).set_index("Iter"), "convergencia", c, error_p
        if fa * fc < 0: b = c
        else: a, fa = c, fc
        x_ant = c
    return pd.DataFrame(history).set_index("Iter"), "limite", x_ant, error_p

def metodo_newton_raphson(f_str, x0, tol, max_iter):
    history = []
    x_n = x0
    for i in range(max_iter):
        fx = evaluar_f(f_str, x_n)
        dfx = calcular_derivada_robusta(f_str, x_n)
        if dfx is None or abs(dfx) < 1e-15: break
        x_next = x_n - fx / dfx
        error_p = calcular_error_relativo(x_next, x_n)
        if i > 2 and error_p > history[-1]["Error (%)"] * 2:
            return pd.DataFrame(history).set_index("Iter"), "divergencia", x_next, error_p
        history.append({"Iter": i+1, "x_n": x_n, "f(x_n)": fx, "f'(x_n)": dfx, "x_n+1": x_next, "Error (%)": error_p})
        if error_p < tol:
            return pd.DataFrame(history).set_index("Iter"), "convergencia", x_next, error_p
        x_n = x_next
    return pd.DataFrame(history).set_index("Iter"), "limite", x_n, error_p

# --- INTERFAZ ---

st.sidebar.header("Configuracion")
metodo_sel = st.sidebar.selectbox("Selecciona Metodo", 
    ["Bisección", "Newton-Raphson", "Interpolación Lagrange", "Diferencias Centrales"])

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Entrada")
    
    if "Interpolación" in metodo_sel or "Diferencias" in metodo_sel:
        st.info("Ingresa los puntos como 'x, y' (uno por linea)")
        puntos_input = st.text_area("Puntos (x, y):", value="0, 1\n1, 3\n2, 2")
        try:
            puntos = [list(map(float, line.split(','))) for line in puntos_input.strip().split('\n')]
            x_pts, y_pts = zip(*puntos)
            x_pts, y_pts = np.array(x_pts), np.array(y_pts)
        except:
            st.error("Formato de puntos incorrecto.")
        
        if metodo_sel == "Interpolación Lagrange":
            x_eval_target = st.number_input("Valor x a evaluar:", value=0.5)
    else:
        func_input = st.text_input("Funcion:", value="x**2 - 2")
        if metodo_sel == "Bisección":
            a_in = st.number_input("Extremo a", value=0.0)
            b_in = st.number_input("Extremo b", value=2.0)
        else:
            x0_in = st.number_input("Valor inicial x0", value=1.0)
        
        tol_in = st.number_input("Tolerancia (%)", value=0.001, format="%.6f")
        iter_in = st.slider("Max Iteraciones", 5, 100, 20)

    ejecutar = st.button("Calcular")

with col2:
    if ejecutar:
        fig = go.Figure()
        
        if metodo_sel == "Interpolación Lagrange":
            y_res = metodo_lagrange(x_pts, y_pts, x_eval_target)
            st.success(f"Valor interpolado en x={x_eval_target}: {y_res:.6f}")
            
            # Graficar curva de Lagrange
            x_range = np.linspace(min(x_pts)-0.5, max(x_pts)+0.5, 100)
            y_range = [metodo_lagrange(x_pts, y_pts, x) for x in x_range]
            fig.add_trace(go.Scatter(x=x_range, y=y_range, name="Polinomio Lagrange", line=dict(color='#00cfcc')))
            fig.add_trace(go.Scatter(x=x_pts, y=y_pts, mode='markers', name="Puntos Originales", marker=dict(size=10, color='white')))
            fig.add_trace(go.Scatter(x=[x_eval_target], y=[y_res], mode='markers', name="Punto Evaluado", marker=dict(size=12, color='red', symbol='star')))
            st.plotly_chart(fig, use_container_width=True)

        elif metodo_sel == "Diferencias Centrales":
            df_derivs = metodo_diferencias_centrales(x_pts, y_pts)
            st.subheader("Cálculo de Derivadas (Velocidad y Aceleración)")
            st.dataframe(df_derivs.style.format(precision=6), use_container_width=True)
            
            fig.add_trace(go.Scatter(x=x_pts, y=y_pts, name="Datos Discretos", line=dict(dash='dash', color='gray')))
            fig.add_trace(go.Scatter(x=df_derivs["Punto x"], y=df_derivs["f'(x) (Vel)"], name="Primera Derivada", mode='lines+markers'))
            st.plotly_chart(fig, use_container_width=True)

        else: # Metodos de Raices (Biseccion/Newton)
            if metodo_sel == "Bisección":
                df, estado, raiz, err_f = metodo_biseccion(func_input, a_in, b_in, tol_in, iter_in)
            else:
                df, estado, raiz, err_f = metodo_newton_raphson(func_input, x0_in, tol_in, iter_in)

            if df is not None:
                if estado == "convergencia": st.success(f"Raiz: {raiz:.8f}")
                elif estado == "error_signos": st.error("f(a) y f(b) deben tener signos opuestos.")
                
                if estado != "error_signos":
                    m1, m2 = st.columns(2)
                    m1.metric("Raiz Aproximada", f"{raiz:.8f}")
                    m2.metric("Error Final", f"{err_f:.2e}%")
                    
                    x_plot = np.linspace(raiz-2, raiz+2, 200)
                    y_plot = [evaluar_f(func_input, v) for v in x_plot]
                    fig.add_trace(go.Scatter(x=x_plot, y=y_plot, name="f(x)"))
                    fig.add_hline(y=0, line_dash="dash")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df.style.format(precision=6), use_container_width=True)