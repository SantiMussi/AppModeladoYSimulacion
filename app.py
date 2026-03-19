import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Numerical Solver Pro", layout="wide")

st.title("Analizador de Métodos Numéricos")
st.markdown("---")

# --- BARRA LATERAL ---
st.sidebar.header("Configuración General")
metodo = st.sidebar.selectbox("Selecciona el Método", 
    ["Punto Fijo", "Bisección", "Newton-Raphson", "Aceleración Aitken"])

# --- FUNCIONES AUXILIARES ---

def evaluar_f(f_str, x):
    f_str_limpia = f_str.replace("^", "**")
    # Intento de corregir multiplicaciones implícitas como 2x -> 2*x
    import re
    f_str_limpia = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', f_str_limpia)
    return eval(f_str_limpia, {"np": np, "x": x, "math": np})

def derivada_num(f_str, x, h=1e-7):
    """Calcula la derivada numérica usando diferencia central."""
    return (evaluar_f(f_str, x + h) - evaluar_f(f_str, x - h)) / (2 * h)

# --- MÉTODOS ---

def metodo_punto_fijo(g_str, x0, tol, max_iter):
    try:
        history = []
        x_ant = x0
        for i in range(max_iter):
            x_nuevo = evaluar_f(g_str, x_ant)
            error = abs(x_nuevo - x_ant)
            history.append({"Iter": i+1, "x_n": x_ant, "g(x_n)": x_nuevo, "Error": error})
            if error < tol:
                return pd.DataFrame(history).set_index("Iter"), True, x_nuevo
            x_ant = x_nuevo
        return pd.DataFrame(history).set_index("Iter"), False, x_ant
    except Exception as e:
        st.error(f"Error: {e}")
        return None, False, None

def metodo_newton_raphson(f_str, x0, tol, max_iter):
    try:
        history = []
        x_n = x0
        for i in range(max_iter):
            fx = evaluar_f(f_str, x_n)
            dfx = derivada_num(f_str, x_n) 
            
            if abs(dfx) < 1e-15:
                st.error("La derivada es demasiado pequeña.")
                break
                
            x_next = x_n - fx / dfx
            error = abs(x_next - x_n)
            history.append({"Iter": i+1, "x_n": x_n, "f(x_n)": fx, "f'(x_n)": dfx, "Error": error})
            
            if error < tol:
                return pd.DataFrame(history).set_index("Iter"), True, x_next
            x_n = x_next
        return pd.DataFrame(history).set_index("Iter"), False, x_n
    except Exception as e:
        st.error(f"Error: {e}")
        return None, False, None

def metodo_aitken(g_str, x0, tol, max_iter):
    try:
        history = []
        x_n = x0
        for i in range(max_iter):
            x1 = evaluar_f(g_str, x_n)
            x2 = evaluar_f(g_str, x1)
            den = x2 - 2*x1 + x_n
            if abs(den) < 1e-15: break
            x_hat = x_n - ((x1 - x_n)**2) / den
            error = abs(x_hat - x_n)
            history.append({"Iter": i+1, "x_n": x_n, "x_n+1": x1, "x_n+2": x2, "x_hat": x_hat, "Error": error})
            if error < tol:
                return pd.DataFrame(history).set_index("Iter"), True, x_hat
            x_n = x_hat
        return pd.DataFrame(history).set_index("Iter"), False, x_n
    except Exception as e:
        st.error(f"Error: {e}")
        return None, False, None

def metodo_biseccion(f_str, a, b, tol, max_iter):
    try:
        if evaluar_f(f_str, a) * evaluar_f(f_str, b) >= 0:
            st.error("f(a) y f(b) deben tener signos opuestos.")
            return None, False, None
        history = []
        for i in range(1, max_iter + 1):
            c = (a + b) / 2
            fc = evaluar_f(f_str, c)
            history.append({"Iter": i, "a": a, "b": b, "c": c, "f(c)": fc})
            if abs(fc) < tol or (b-a)/2 < tol:
                return pd.DataFrame(history).set_index("Iter"), True, c
            if evaluar_f(f_str, a) * fc < 0: b = c
            else: a = c
        return pd.DataFrame(history).set_index("Iter"), False, c
    except Exception as e:
        st.error(f"Error: {e}")
        return None, False, None

# --- UI ---

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Entrada de Datos")
    if metodo in ["Punto Fijo", "Aceleración Aitken"]:
        func_input = st.text_input("Define g(x):", "np.cos(x)")
        x0 = st.number_input("x0 (Valor inicial)", value=0.5)
    elif metodo == "Newton-Raphson":
        func_input = st.text_input("Define f(x):", "x**2 - 2")
        x0 = st.number_input("x0 (Valor inicial)", value=1.0)
    else:
        func_input = st.text_input("Define f(x):", "x**2 - 2")
        a_in = st.number_input("a", value=1.0)
        b_in = st.number_input("b", value=2.0)
    
    tol = st.number_input("Tolerancia", value=1e-7, format="%.8f")
    iters = st.slider("Máximo de iteraciones", 5, 100, 20)
    boton = st.button("Calcular")

with col2:
    if boton:
        if metodo == "Punto Fijo":
            df, conv, res = metodo_punto_fijo(func_input, x0, tol, iters)
        elif metodo == "Newton-Raphson":
            df, conv, res = metodo_newton_raphson(func_input, x0, tol, iters)
        elif metodo == "Aceleración Aitken":
            df, conv, res = metodo_aitken(func_input, x0, tol, iters)
        else:
            df, conv, res = metodo_biseccion(func_input, a_in, b_in, tol, iters)
            
        if df is not None:
            if conv:
                st.success(f"¡Convergencia lograda! Raíz: **{res:.8f}**")
            else:
                st.warning("No se alcanzó la tolerancia.")
            
            fig = go.Figure()
            x_plot = np.linspace(res-2, res+2, 100)
            y_plot = [evaluar_f(func_input, val) for val in x_plot]
            fig.add_trace(go.Scatter(x=x_plot, y=y_plot, name="f(x)"))
            fig.add_hline(y=0, line_dash="dash", line_color="white")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Tabla de Iteraciones")
            st.dataframe(df.style.format(precision=8), use_container_width=True)