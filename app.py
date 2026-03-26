import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import re
import sympy as sp

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

def calcular_error_absoluto(x_nuevo, x_anterior):
    return abs(x_nuevo - x_anterior)

def calcular_error_relativo(x_nuevo, x_anterior):
    if abs(x_nuevo) < 1e-18: return 100.0
    return (abs(x_nuevo - x_anterior) / abs(x_nuevo)) * 100

# --- LOGICA DE LAGRANGE (SIMBOLICO Y NUMERICO) ---

def calcular_lagrange_completo(x_points, y_points):
    x_sym = sp.symbols('x')
    n = len(x_points)
    listado_L = []
    polinomio_total = 0
    
    for i in range(n):
        li = 1
        for j in range(n):
            if i != j:
                li *= (x_sym - x_points[j]) / (x_points[i] - x_points[j])
        
        # sp.nsimplify convierte decimales (0.5) a fracciones exactas (1/2)
        # sp.expand hace la distributiva para eliminar los paréntesis
        li_limpio = sp.expand(sp.nsimplify(li))
        listado_L.append(li_limpio)
        
        polinomio_total += y_points[i] * li
    
    # Hacemos lo mismo para el polinomio final
    poly_total_limpio = sp.expand(polinomio_total).evalf(5)
    
    return poly_total_limpio, listado_L

def calcular_termino_error_lagrange(x_points, x_eval):
    # Calcula el productorio (x - x0)*(x - x1)... que compone la cota de error
    producto = 1.0
    for xi in x_points:
        producto *= (x_eval - xi)
    return abs(producto)

# --- LOGICA DE DIFERENCIAS CENTRALES ---

def metodo_diferencias_centrales(x_points, y_points):
    # Asumimos h constante basado en los dos primeros puntos
    h = x_points[1] - x_points[0]
    derivadas = []
    # Solo podemos calcular diferencias centrales para puntos que tengan vecinos (ignoramos extremos)
    for i in range(1, len(x_points) - 1):
        d1 = (y_points[i+1] - y_points[i-1]) / (2 * h)
        d2 = (y_points[i+1] - 2*y_points[i] + y_points[i-1]) / (h**2)
        
        # Error local de truncamiento teorico de dif centrales es O(h^2)
        error_truncamiento_estimado = h**2 
        
        derivadas.append({
            "Punto x": x_points[i], 
            "f'(x) (Vel)": d1, 
            "f''(x) (Ace)": d2,
            "Error Local O(h^2)": error_truncamiento_estimado
        })
    return pd.DataFrame(derivadas)

# --- LOGICA DE LOS METODOS ANTERIORES (RAICES) ---

def metodo_biseccion(f_str, a, b, tol, max_iter):
    history = []
    fa, fb = evaluar_f(f_str, a), evaluar_f(f_str, b)
    if fa is None or fb is None or fa * fb >= 0:
        return None, "error_signos", 0, 100
    x_ant = a
    for i in range(max_iter):
        c = (a + b) / 2
        fc = evaluar_f(f_str, c)
        error_abs = calcular_error_absoluto(c, x_ant)
        error_rel = calcular_error_relativo(c, x_ant)
        
        history.append({
            "Iter": i+1, "a": a, "b": b, "x_n (c)": c, 
            "Residual f(c)": fc, 
            "Error Local (Abs)": error_abs, 
            "Error Relativo (%)": error_rel
        })
        
        if abs(fc) < 1e-15 or error_rel < tol:
            return pd.DataFrame(history).set_index("Iter"), "convergencia", c, error_rel
        if fa * fc < 0: b = c
        else: a, fa = c, fc
        x_ant = c
    return pd.DataFrame(history).set_index("Iter"), "limite", x_ant, error_rel

def metodo_newton_raphson(f_str, x0, tol, max_iter):
    history = []
    x_n = x0
    for i in range(max_iter):
        fx = evaluar_f(f_str, x_n)
        dfx = calcular_derivada_robusta(f_str, x_n)
        if dfx is None or abs(dfx) < 1e-15: break
        
        x_next = x_n - fx / dfx
        error_abs = calcular_error_absoluto(x_next, x_n)
        error_rel = calcular_error_relativo(x_next, x_n)
        
        if i > 2 and error_rel > history[-1]["Error Relativo (%)"] * 2:
            return pd.DataFrame(history).set_index("Iter"), "divergencia", x_next, error_rel
            
        history.append({
            "Iter": i+1, "x_n": x_n, "f(x_n)": fx, "f'(x_n)": dfx, "x_n+1": x_next, 
            "Error Local (Abs)": error_abs, 
            "Error Relativo (%)": error_rel
        })
        
        if error_rel < tol:
            return pd.DataFrame(history).set_index("Iter"), "convergencia", x_next, error_rel
        x_n = x_next
    return pd.DataFrame(history).set_index("Iter"), "limite", x_n, error_rel

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
            puntos = []
            for line in puntos_input.strip().split('\n'):
                if line.strip():
                    x_str, y_str = line.split(',')
                    # Usamos nuestra propia función evaluar_f para que entienda 'e', 'pi', etc.
                    # Pasamos 0 como valor de x de relleno, ya que son puntos fijos
                    x_val = evaluar_f(x_str.strip(), 0) 
                    y_val = evaluar_f(y_str.strip(), 0)
                    
                    if x_val is None or y_val is None:
                        raise ValueError("Error al evaluar expresión")
                        
                    puntos.append([x_val, y_val])
                    
            x_pts, y_pts = zip(*puntos)
            x_pts, y_pts = np.array(x_pts), np.array(y_pts)
        except Exception as e:
            st.error("Formato de puntos incorrecto. Asegúrate de usar 'x, y' (ej: 1, e**1)")
        
        if metodo_sel == "Interpolación Lagrange":
            x_eval_target = st.number_input("Valor x a evaluar (opcional):", value=0.5)
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
            poly_simplificado, lista_Li = calcular_lagrange_completo(x_pts, y_pts)
            
            st.subheader("Resultado de Lagrange")
            st.markdown("### Polinomio Interpolante $P(x)$:")
            st.latex(sp.latex(poly_simplificado))
            
            with st.expander("Ver Polinomios Base L_i(x)"):
                for idx, li in enumerate(lista_Li):
                    st.latex(f"L_{{{idx}}}(x) = {sp.latex(li)}")
            
            # Evaluación y Análisis de Error
            x_sym = sp.symbols('x')
            y_res = float(poly_simplificado.subs(x_sym, x_eval_target))
            termino_err = calcular_termino_error_lagrange(x_pts, x_eval_target)
            
            c1, c2 = st.columns(2)
            c1.info(f"Evaluado en x={x_eval_target}: **{y_res:.6f}**")
            c2.warning(f"Término de Error $\prod(x-x_i)$: **{termino_err:.6f}**")
            
            # Gráfico
            x_range = np.linspace(min(x_pts)-0.5, max(x_pts)+0.5, 100)
            f_lamb = sp.lambdify(x_sym, poly_simplificado, "numpy")
            y_range = f_lamb(x_range) if isinstance(f_lamb(x_range), np.ndarray) else np.full_like(x_range, f_lamb(x_range))
            
            fig.add_trace(go.Scatter(x=x_range, y=y_range, name="P(x)", line=dict(color='#00cfcc')))
            fig.add_trace(go.Scatter(x=x_pts, y=y_pts, mode='markers', name="Puntos Originales", marker=dict(size=10, color='white')))
            fig.add_trace(go.Scatter(x=[x_eval_target], y=[y_res], mode='markers', name="Punto Evaluado", marker=dict(size=12, color='red', symbol='star')))
            fig.update_layout(template="plotly_dark", title="Interpolación de Lagrange y Análisis de Error")
            st.plotly_chart(fig, use_container_width=True)

        elif metodo_sel == "Diferencias Centrales":
            df_derivs = metodo_diferencias_centrales(x_pts, y_pts)
            st.subheader("Cálculo de Derivadas y Error de Truncamiento")
            st.dataframe(df_derivs.style.format(precision=6), use_container_width=True)
            
            fig.add_trace(go.Scatter(x=x_pts, y=y_pts, name="Datos Discretos", line=dict(dash='dash', color='gray')))
            fig.add_trace(go.Scatter(x=df_derivs["Punto x"], y=df_derivs["f'(x) (Vel)"], name="Primera Derivada", mode='lines+markers', line=dict(color='#00cfcc')))
            fig.update_layout(template="plotly_dark", title="Diferencias Centrales")
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
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Raiz Aproximada", f"{raiz:.8f}")
                    m2.metric("Error Relativo Final", f"{err_f:.2e}%")
                    m3.metric("Error Local Final", f"{df.iloc[-1]['Error Local (Abs)']:.2e}")
                    
                    x_plot = np.linspace(raiz-2, raiz+2, 200)
                    y_plot = [evaluar_f(func_input, v) for v in x_plot]
                    fig.add_trace(go.Scatter(x=x_plot, y=y_plot, name="f(x)"))
                    fig.add_hline(y=0, line_dash="dash")
                    fig.update_layout(template="plotly_dark", title=f"Método de {metodo_sel} - Análisis de Error")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df.style.format(precision=6), use_container_width=True)