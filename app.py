import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import re

# Configuración de la página
st.set_page_config(page_title="Santiago Mussi | Numeric Solver", layout="wide")

st.title("Analizador de Métodos Numéricos Pro")
st.markdown("---")

# --- FUNCIONES DE EVALUACIÓN Y DERIVADA ---

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
    # Diferencia central de 4to orden (Precisión Python 3.13)
    num = -f(x_val + 2*h) + 8*f(x_val + h) - 8*f(x_val - h) + f(x_val - 2*h)
    den = 12 * h
    return num / den

def calcular_error_relativo(x_nuevo, x_anterior):
    if abs(x_nuevo) < 1e-18: return 100.0
    return (abs(x_nuevo - x_anterior) / abs(x_nuevo)) * 100

# --- LÓGICA DE LOS MÉTODOS ---

def metodo_biseccion(f_str, a, b, tol, max_iter):
    history = []
    fa, fb = evaluar_f(f_str, a), evaluar_f(f_str, b)
    if fa * fb >= 0:
        st.error("Error: f(a) y f(b) deben tener signos opuestos.")
        return None, False, 0, 100
    
    x_ant = a
    for i in range(max_iter):
        c = (a + b) / 2
        fc = evaluar_f(f_str, c)
        error_p = calcular_error_relativo(c, x_ant)
        
        history.append({"Iter": i+1, "a": a, "b": b, "x_n (c)": c, "f(c)": fc, "Error (%)": error_p})
        
        if abs(fc) < 1e-15 or error_p < tol:
            return pd.DataFrame(history).set_index("Iter"), True, c, error_p
        
        if fa * fc < 0: b = c
        else: a, fa = c, fc
        x_ant = c
    return pd.DataFrame(history).set_index("Iter"), False, x_ant, error_p

def metodo_newton_raphson(f_str, x0, tol, max_iter):
    history = []
    x_n = x0
    for i in range(max_iter):
        fx = evaluar_f(f_str, x_n)
        dfx = calcular_derivada_robusta(f_str, x_n)
        if dfx is None or abs(dfx) < 1e-15: break
        x_next = x_n - fx / dfx
        error_p = calcular_error_relativo(x_next, x_n)
        history.append({"Iter": i+1, "x_n": x_n, "f(x_n)": fx, "f'(x_n)": dfx, "x_n+1": x_next, "Error (%)": error_p})
        if error_p < tol:
            return pd.DataFrame(history).set_index("Iter"), True, x_next, error_p
        x_n = x_next
    return pd.DataFrame(history).set_index("Iter"), False, x_n, error_p

def metodo_punto_fijo(g_str, x0, tol, max_iter):
    history = []
    x_ant = x0
    for i in range(max_iter):
        x_nuevo = evaluar_f(g_str, x_ant)
        error_p = calcular_error_relativo(x_nuevo, x_ant)
        history.append({"Iter": i+1, "x_n": x_ant, "x_n+1": x_nuevo, "Error (%)": error_p})
        if error_p < tol:
            return pd.DataFrame(history).set_index("Iter"), True, x_nuevo, error_p
        x_ant = x_nuevo
    return pd.DataFrame(history).set_index("Iter"), False, x_ant, error_p

def metodo_aitken(g_str, x0, tol, max_iter):
    history = []
    x_n = x0
    for i in range(max_iter):
        x1 = evaluar_f(g_str, x_n)
        x2 = evaluar_f(g_str, x1)
        den = x2 - 2*x1 + x_n
        if abs(den) < 1e-15: break
        x_hat = x_n - ((x1 - x_n)**2) / den
        error_p = calcular_error_relativo(x_hat, x_n)
        history.append({"Iter": i+1, "x_n": x_n, "x_hat": x_hat, "Error (%)": error_p})
        if error_p < tol:
            return pd.DataFrame(history).set_index("Iter"), True, x_hat, error_p
        x_n = x_hat
    return pd.DataFrame(history).set_index("Iter"), False, x_n, error_p

# --- INTERFAZ ---

st.sidebar.header("Configuración")
metodo_sel = st.sidebar.selectbox("Selecciona Método", ["Bisección", "Newton-Raphson", "Punto Fijo", "Aceleración Aitken"])

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Entrada")
    
    # Teclado Científico
    if 'input_val' not in st.session_state: st.session_state.input_val = "x**2 - 2"
    btns = st.columns(4)
    ops = ["sin(x)", "cos(x)", "exp(x)", "log(x)", "sqrt(x)", "x^2", "pi", "("]
    for i, op in enumerate(ops):
        if btns[i % 4].button(op): st.session_state.input_val += op.replace("x^2", "**2")
    
    func_input = st.text_input("Función:", value=st.session_state.input_val)
    st.session_state.input_val = func_input

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
        if metodo_sel == "Bisección":
            df, conv, raiz, err_f = metodo_biseccion(func_input, a_in, b_in, tol_in, iter_in)
        elif metodo_sel == "Newton-Raphson":
            df, conv, raiz, err_f = metodo_newton_raphson(func_input, x0_in, tol_in, iter_in)
        elif metodo_sel == "Punto Fijo":
            df, conv, raiz, err_f = metodo_punto_fijo(func_input, x0_in, tol_in, iter_in)
        else:
            df, conv, raiz, err_f = metodo_aitken(func_input, x0_in, tol_in, iter_in)

        if df is not None:
            m1, m2 = st.columns(2)
            m1.metric("Raíz Aproximada", f"{raiz:.8f}")
            m2.metric("Error Final", f"{err_f:.2e}%")

            # Gráfico
            x_range = np.linspace(raiz - 2, raiz + 2, 200)
            y_range = [evaluar_f(func_input, v) for v in x_range]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_range, y=y_range, name="f(x)", line=dict(color='#00cfcc')))
            fig.add_trace(go.Scatter(x=[raiz], y=[0], mode='markers', marker=dict(color='red', size=12, symbol='star'), name="Raíz"))
            fig.add_hline(y=0, line_dash="dash", line_color="white")
            fig.update_layout(template="plotly_dark", title=f"Gráfico: {metodo_sel}")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df.style.format(precision=6), use_container_width=True)