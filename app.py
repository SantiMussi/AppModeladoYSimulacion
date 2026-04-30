import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import re
import sympy as sp
import scipy.stats as st_stats
import scipy.integrate as spi
import streamlit.components.v1 as components

# Configuración de la página
st.set_page_config(page_title="Santiago Mussi | Numeric Solver", layout="wide")

st.title("Analizador de Métodos Numéricos")
st.markdown("---")

# --- INYECCIÓN DE SHORTCUT OCULTO ---
components.html(
    """
    <script>
    const parent = window.parent.document;
    if (!parent.getElementById('shortcut-listener')) {
        parent.addEventListener('keydown', function(event) {
            if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === 'f') {
                event.preventDefault();
                let labels = parent.querySelectorAll('div[data-testid="stCheckbox"] label p');
                for (let p of labels) {
                    if (p.textContent.includes('Visor Fórmulas (Ctrl+Shift+F)')) {
                        p.closest('label').click();
                        break;
                    }
                }
            }
        });
        let div = parent.createElement('div');
        div.id = 'shortcut-listener';
        div.style.display = 'none';
        parent.body.appendChild(div);
    }
    
    setTimeout(() => {
        let labels = parent.querySelectorAll('div[data-testid="stCheckbox"] label p');
        for (let p of labels) {
            if (p.textContent.includes('Visor Fórmulas (Ctrl+Shift+F)')) {
                let container = p.closest('div[data-testid="stCheckbox"]');
                if (container) {
                    container.style.opacity = '0';
                    container.style.height = '0px';
                    container.style.overflow = 'hidden';
                    container.style.position = 'absolute';
                }
                break;
            }
        }
    }, 50);
    </script>
    """,
    height=0,
    width=0
)

# --- FUNCIONES DE EVALUACIÓN Y DERIVADA ---

def evaluar_f(f_str, x_val):
    """Evalúa funciones para cálculos numéricos rápidos (gráficos y raíces)"""
    try:
        f_proc = f_str.replace("^", "**").replace("sen", "sin").replace("ln", "log")
        f_proc = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', f_proc)
        contexto = {
            "np": np, "x": x_val, "sin": np.sin, "cos": np.cos, 
            "tan": np.tan, "exp": np.exp, "log": np.log, "log10": np.log10,
            "sqrt": np.sqrt, "pi": np.pi, "e": np.e, "sp": sp
        }
        res = eval(f_proc, {"__builtins__": None}, contexto)
        if np.isnan(float(res)) or np.isinf(float(res)):
            raise ValueError("Valor indefinido")
        return float(res)
    except Exception:
        # Intento de aproximación por límite si hay singularidad matemática (ej. x=0 en sin(x)/x)
        try:
            contexto["x"] = x_val + 1e-12
            res_lim = eval(f_proc, {"__builtins__": None}, contexto)
            if np.isnan(float(res_lim)) or np.isinf(float(res_lim)):
                return None
            return float(res_lim)
        except:
            return None

def evaluar_f_array(f_str, x_arr, y_arr=None):
    """Evalúa funciones con arrays de numpy de forma segura"""
    try:
        f_proc = f_str.replace("^", "**").replace("sen", "sin").replace("ln", "log")
        f_proc = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', f_proc)
        contexto = {
            "np": np, "x": x_arr, "y": y_arr, "sin": np.sin, "cos": np.cos, 
            "tan": np.tan, "exp": np.exp, "log": np.log, "log10": np.log10,
            "sqrt": np.sqrt, "pi": np.pi, "e": np.e, "sp": sp
        }
        res = eval(f_proc, {"__builtins__": None}, contexto)
        if isinstance(res, (int, float)):
            res = np.full(np.shape(x_arr), res)
        return np.asarray(res, dtype=float)
    except Exception:
        # Fallback iterativo
        res_list = []
        if y_arr is None:
            for xv in np.atleast_1d(x_arr):
                val = evaluar_f(f_str, xv)
                res_list.append(val if val is not None else np.nan)
        else:
            for xv, yv in zip(np.atleast_1d(x_arr), np.atleast_1d(y_arr)):
                contexto_scalar = {
                    "np": np, "x": xv, "y": yv, "sin": np.sin, "cos": np.cos, 
                    "tan": np.tan, "exp": np.exp, "log": np.log, "log10": np.log10,
                    "sqrt": np.sqrt, "pi": np.pi, "e": np.e
                }
                try:
                    val = eval(f_proc, {"__builtins__": None}, contexto_scalar)
                    res_list.append(float(val) if not (np.isnan(float(val)) or np.isinf(float(val))) else np.nan)
                except:
                    res_list.append(np.nan)
        return np.array(res_list)

def evaluar_f_con_indeterminacion(f_str, x_val):
    """Evalúa la función, detectando indeterminaciones y resolviendo por límite."""
    f_proc = f_str.replace("^", "**").replace("sen", "sin").replace("ln", "log")
    f_proc = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', f_proc)
    contexto = {
        "np": np, "x": x_val, "sin": np.sin, "cos": np.cos, 
        "tan": np.tan, "exp": np.exp, "log": np.log, "log10": np.log10,
        "sqrt": np.sqrt, "pi": np.pi, "e": np.e
    }
    es_indet = False
    try:
        with np.errstate(divide='raise', invalid='raise'):
            res = eval(f_proc, {"__builtins__": None}, contexto)
        if np.isnan(float(res)) or np.isinf(float(res)):
            es_indet = True
    except Exception:
        es_indet = True

    if es_indet:
        try:
            x_sym = sp.Symbol('x')
            f_sym = sp.sympify(f_str.replace("^", "**").replace("sen", "sin").replace("ln", "log"))
            limite = sp.limit(f_sym, x_sym, x_val)
            if limite.is_real:
                return float(limite), True
        except:
            pass
        try:
            contexto["x"] = x_val + 1e-12
            res_lim = eval(f_proc, {"__builtins__": None}, contexto)
            if not (np.isnan(float(res_lim)) or np.isinf(float(res_lim))):
                return float(res_lim), True
        except:
            pass
        return None, True
    return float(res), False

def calcular_derivada_robusta(f_str, x_val, h=1e-5):
    f = lambda x: evaluar_f(f_str, x)
    val_plus2 = f(x_val + 2*h)
    val_plus1 = f(x_val + h)
    val_minus1 = f(x_val - h)
    val_minus2 = f(x_val - 2*h)
    if any(v is None for v in [val_plus2, val_plus1, val_minus1, val_minus2]): return 0
    num = -val_plus2 + 8*val_plus1 - 8*val_minus1 + val_minus2
    den = 12 * h
    return num / den

# --- LÓGICA DE LAGRANGE (SIMBÓLICO CON VARIABLES) ---

def calcular_lagrange_avanzado(x_strs, y_strs):
    x_sym = sp.symbols('x')
    n = len(x_strs)
    listado_L = []
    polinomio_total = 0
    
    # Convertimos strings a objetos simbólicos (acepta 'b', 'pi', etc.)
    x_exact = [sp.nsimplify(sp.sympify(x), constants=[sp.pi, sp.E]) for x in x_strs]
    y_exact = [sp.nsimplify(sp.sympify(y), constants=[sp.pi, sp.E]) for y in y_strs]
    
    for i in range(n):
        li = 1
        for j in range(n):
            if i != j:
                li *= (x_sym - x_exact[j]) / (x_exact[i] - x_exact[j])
        
        li_limpio = sp.simplify(li)
        listado_L.append(li_limpio)
        polinomio_total += y_exact[i] * li
    
    poly_final = sp.simplify(sp.expand(polinomio_total))
    return poly_final, listado_L, x_exact, y_exact

# --- LÓGICA DE DIFERENCIAS CENTRALES ---

def metodo_diferencias_centrales(x_pts, y_pts):
    if len(x_pts) < 3: return None
    h = x_pts[1] - x_pts[0]
    derivadas = []
    for i in range(1, len(x_pts) - 1):
        d1 = (y_pts[i+1] - y_pts[i-1]) / (2 * h)
        d2 = (y_pts[i+1] - 2*y_pts[i] + y_pts[i-1]) / (h**2)
        derivadas.append({
            "Punto x": x_pts[i], "f'(x)": d1, "f''(x)": d2, "Error O(h^2)": h**2
        })
    return pd.DataFrame(derivadas)

# --- INTEGRACIÓN: SIMPSON 1/3 ---

def metodo_simpson_13(f_str, a, b, n, xi_punto=None):
    """Regla de Simpson 1/3 compuesta. n debe ser par."""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x_pts = np.linspace(a, b, n + 1)
    y_pts_list = []
    indets = []
    for xi in x_pts:
        val, is_indet = evaluar_f_con_indeterminacion(f_str, xi)
        if val is None:
            return None, None, None, None, None, None
        if is_indet:
            indets.append((xi, val))
        y_pts_list.append(val)
    y_pts = np.array(y_pts_list)

    # Suma compuesta Simpson 1/3
    suma = y_pts[0] + y_pts[-1]
    for i in range(1, n):
        suma += (4 if i % 2 != 0 else 2) * y_pts[i]
    integral = (h / 3) * suma

    # Error de truncamiento: |E_T| <= (b-a)*h^4/180 * |f''''(xi)|
    h_num = max(h * 0.1, 1e-4)
    if xi_punto is not None:
        # Evaluar f⁴ en el punto exacto ξ dado por el usuario
        vals = [evaluar_f(f_str, xi_punto + k * h_num) for k in [-2, -1, 0, 1, 2]]
        if all(v is not None for v in vals):
            f4_max = abs((vals[0] - 4*vals[1] + 6*vals[2] - 4*vals[3] + vals[4]) / h_num**4)
        else:
            f4_max = 0.0
    else:
        f4_vals = []
        for xi in x_pts[2:-2]:
            vals = [evaluar_f(f_str, xi + k * h_num) for k in [-2, -1, 0, 1, 2]]
            if all(v is not None for v in vals):
                f4 = (vals[0] - 4*vals[1] + 6*vals[2] - 4*vals[3] + vals[4]) / h_num**4
                f4_vals.append(abs(f4))
        f4_max = max(f4_vals) if f4_vals else 0.0
    error_trunc = abs((b - a) * h**4 * f4_max / 180)

    # Tabla por nodo: N, Xn, F(Xn), coeficiente Simpson, término ponderado
    tabla = []
    for k in range(n + 1):
        if k == 0 or k == n:
            coef = 1
        elif k % 2 != 0:
            coef = 4
        else:
            coef = 2
        fk = float(y_pts[k])
        tabla.append({
            "N": k,
            "Xₙ": round(float(x_pts[k]), precision),
            "F(Xₙ)": round(fk, precision),
            "Coef.": coef,
            "Coef. × F(Xₙ)": round(coef * fk, precision),
        })
    return integral, error_trunc, h, pd.DataFrame(tabla), indets, y_pts_list


# --- INTEGRACIÓN: SIMPSON 3/8 ---

def metodo_simpson_38(f_str, a, b, n, a_str="", b_str="", xi_punto=None):
    """Regla de Simpson 3/8 compuesta. n debe ser múltiplo de 3."""
    if n % 3 != 0:
        n += (3 - (n % 3))
    h = (b - a) / n
    x_pts = np.linspace(a, b, n + 1)
    y_pts_list = []
    indets = []
    for xi in x_pts:
        val, is_indet = evaluar_f_con_indeterminacion(f_str, xi)
        if val is None:
            return None, None, None, None, None, None
        if is_indet:
            indets.append((xi, val))
        y_pts_list.append(val)
    y_pts = np.array(y_pts_list)

    # Suma compuesta Simpson 3/8
    suma = y_pts[0] + y_pts[-1]
    for i in range(1, n):
        if i % 3 == 0:
            suma += 2 * y_pts[i]
        else:
            suma += 3 * y_pts[i]
    integral = (3 * h / 8) * suma

    # Error de truncamiento: |E_T| <= (b-a)*h^4/80 * |f''''(xi)|
    h_num = max(h * 0.1, 1e-4)
    if xi_punto is not None:
        vals = [evaluar_f(f_str, xi_punto + k * h_num) for k in [-2, -1, 0, 1, 2]]
        if all(v is not None for v in vals):
            f4_max = abs((vals[0] - 4*vals[1] + 6*vals[2] - 4*vals[3] + vals[4]) / h_num**4)
        else:
            f4_max = 0.0
    else:
        f4_vals = []
        for xi in x_pts[2:-2]:
            vals = [evaluar_f(f_str, xi + k * h_num) for k in [-2, -1, 0, 1, 2]]
            if all(v is not None for v in vals):
                f4 = (vals[0] - 4*vals[1] + 6*vals[2] - 4*vals[3] + vals[4]) / h_num**4
                f4_vals.append(abs(f4))
        f4_max = max(f4_vals) if f4_vals else 0.0
    error_trunc = abs((b - a) * h**4 * f4_max / 80)

    def format_xn(val_float, k):
        try:
            a_s = str(a_str).replace('π', 'pi').lower()
            b_s = str(b_str).replace('π', 'pi').lower()
            if 'pi' in a_s or 'pi' in b_s:
                a_sym = sp.sympify(a_s)
                b_sym = sp.sympify(b_s)
                a_c = sp.simplify(a_sym / sp.pi)
                b_c = sp.simplify(b_sym / sp.pi)
                if a_c.is_Rational and b_c.is_Rational:
                    h_c = (b_c - a_c) / n
                    D = int(sp.lcm(a_c.q, h_c.q))
                    num = int((a_c + k * h_c) * D)
                    if num == 0:
                        return "0"
                    
                    if D == 1:
                        if num == 1: return "π"
                        elif num == -1: return "-π"
                        return f"{num}π"
                    
                    if num == 1: return f"π/{D}"
                    elif num == -1: return f"-π/{D}"
                    return f"{num}π/{D}"
        except:
            pass
        return str(round(float(val_float), precision))

    # Tabla por nodo
    tabla = []
    for k in range(n + 1):
        if k == 0 or k == n:
            coef = 1
        elif k % 3 == 0:
            coef = 2
        else:
            coef = 3
        fk = float(y_pts[k])
        tabla.append({
            "N": k,
            "Xₙ": format_xn(x_pts[k], k),
            "F(Xₙ)": round(fk, precision),
            "Coef.": coef,
            "Coef. × F(Xₙ)": round(coef * fk, precision),
        })
    return integral, error_trunc, h, pd.DataFrame(tabla), indets, y_pts_list


# --- INTEGRACIÓN: TRAPECIOS ---

def metodo_trapecios(f_str, a, b, n, xi_punto=None):
    """Regla de Trapecios compuesta con n subintervalos."""
    h = (b - a) / n
    x_pts = np.linspace(a, b, n + 1)
    y_pts_list = []
    indets = []
    for xi in x_pts:
        val, is_indet = evaluar_f_con_indeterminacion(f_str, xi)
        if val is None:
            return None, None, None, None, None, None
        if is_indet:
            indets.append((xi, val))
        y_pts_list.append(val)
    y_pts = np.array(y_pts_list)

    # Suma compuesta: extremos x1, interiores x2
    suma = y_pts[0] + y_pts[-1] + 2 * np.sum(y_pts[1:-1])
    integral = (h / 2) * suma

    # Error de truncamiento: |E_T| <= (b-a)*h^2/12 * max|f''(xi)|
    h_num = max(h * 0.1, 1e-4)
    if xi_punto is not None:
        vals = [evaluar_f(f_str, xi_punto + k * h_num) for k in [-1, 0, 1]]
        if all(v is not None for v in vals):
            f2_max = abs((vals[0] - 2*vals[1] + vals[2]) / h_num**2)
        else:
            f2_max = 0.0
    else:
        f2_vals = []
        for xi in x_pts[1:-1]:
            vals = [evaluar_f(f_str, xi + k * h_num) for k in [-1, 0, 1]]
            if all(v is not None for v in vals):
                f2 = (vals[0] - 2*vals[1] + vals[2]) / h_num**2
                f2_vals.append(abs(f2))
        f2_max = max(f2_vals) if f2_vals else 0.0
    error_trunc = abs((b - a) * h**2 * f2_max / 12)

    # Tabla por nodo
    tabla = []
    for k in range(n + 1):
        coef = 1 if (k == 0 or k == n) else 2
        fk = float(y_pts[k])
        tabla.append({
            "N": k,
            "Xₙ": round(float(x_pts[k]), precision),
            "F(Xₙ)": round(fk, precision),
            "Coef.": coef,
            "Coef. × F(Xₙ)": round(coef * fk, precision),
        })
    return integral, error_trunc, h, pd.DataFrame(tabla), indets, y_pts_list


# --- INTEGRACIÓN: RECTÁNGULO MEDIO ---

def metodo_rectangulo_medio(f_str, a, b, n):
    """Regla de Rectángulo Medio (Punto Medio) compuesta con n subintervalos."""
    h = (b - a) / n
    x_pts = np.linspace(a, b, n + 1)
    x_mid = (x_pts[:-1] + x_pts[1:]) / 2
    y_mid = np.array([evaluar_f(f_str, xi) for xi in x_mid])
    if any(v is None for v in y_mid):
        return None, None, None, None

    # Suma compuesta
    suma = np.sum(y_mid)
    integral = h * suma

    # Error de truncamiento: |E_T| <= (b-a)*h^2/24 * max|f''(xi)|
    h_num = max(h * 0.1, 1e-4)
    f2_vals = []
    for xi in x_mid:
        vals = [evaluar_f(f_str, xi + k * h_num) for k in [-1, 0, 1]]
        if all(v is not None for v in vals):
            f2 = (vals[0] - 2*vals[1] + vals[2]) / h_num**2
            f2_vals.append(abs(f2))
    f2_max = max(f2_vals) if f2_vals else 0.0
    error_trunc = abs((b - a) * h**2 * f2_max / 24)

    # Tabla por nodo (puntos medios e intervalos)
    tabla = []
    for k in range(n):
        fk = float(y_mid[k])
        tabla.append({
            "N": k+1,
            "Xₙ_inf": round(float(x_pts[k]), precision),
            "Xₙ_sup": round(float(x_pts[k+1]), precision),
            "Xₘ (Pto. Medio)": round(float(x_mid[k]), precision),
            "F(Xₘ)": round(fk, precision),
            "Coef.": 1,
            "Coef. × F(Xₘ)": round(1 * fk, precision),
        })
    return integral, error_trunc, h, pd.DataFrame(tabla)


# --- INTEGRACIÓN: MONTECARLO ---

def metodo_montecarlo(f_str, a, b, n, conf_level=95.0, seed=None, antithetic=False):
    if seed is not None:
        np.random.seed(seed)
        
    if antithetic:
        half_n = n // 2
        x_half = np.random.uniform(a, b, half_n)
        x_rand = np.concatenate([x_half, a + b - x_half])
        if n % 2 != 0:
            x_rand = np.append(x_rand, np.random.uniform(a, b, 1))
    else:
        x_rand = np.random.uniform(a, b, n)
        
    y_eval = evaluar_f_array(f_str, x_rand)
    
    vol = (b - a)
    valid_mask = ~np.isnan(y_eval)
    valid_n = np.sum(valid_mask)
    if valid_n == 0:
        return 0.0, 0.0, vol, 0.0, 0.0, pd.DataFrame(), x_rand, y_eval, n
        
    y_valid = y_eval[valid_mask]
    integral = vol * np.mean(y_valid)
    
    var = np.var(y_valid, ddof=1) if valid_n > 1 else 0.0
    std_dev = np.sqrt(var)
    std_error = vol * np.sqrt(var / valid_n)
    
    alpha = 1 - (conf_level / 100.0)
    z = st_stats.norm.ppf(1 - alpha/2)
    margin = z * std_error
    ic_lower = integral - margin
    ic_upper = integral + margin
    
    tabla = []
    for i in range(min(10, n)):
        tabla.append({
            "Punto i": i+1,
            "x_i": round(x_rand[i], precision),
            "f(x_i)": round(y_eval[i], precision) if not np.isnan(y_eval[i]) else "Indefinido",
        })
    df_tabla = pd.DataFrame(tabla)
    
    invalid_count = n - valid_n
    return integral, std_error, std_dev, vol, ic_lower, ic_upper, df_tabla, x_rand, y_eval, invalid_count

def metodo_montecarlo_doble(f_str, a_x, b_x, a_y, b_y, n, conf_level=95.0, seed=None, antithetic=False):
    if seed is not None:
        np.random.seed(seed)
        
    if antithetic:
        half_n = n // 2
        x_half = np.random.uniform(a_x, b_x, half_n)
        y_half = np.random.uniform(a_y, b_y, half_n)
        x_rand = np.concatenate([x_half, a_x + b_x - x_half])
        y_rand = np.concatenate([y_half, a_y + b_y - y_half])
        if n % 2 != 0:
            x_rand = np.append(x_rand, np.random.uniform(a_x, b_x, 1))
            y_rand = np.append(y_rand, np.random.uniform(a_y, b_y, 1))
    else:
        x_rand = np.random.uniform(a_x, b_x, n)
        y_rand = np.random.uniform(a_y, b_y, n)
        
    z_eval = evaluar_f_array(f_str, x_rand, y_rand)
    
    area = (b_x - a_x) * (b_y - a_y)
    valid_mask = ~np.isnan(z_eval)
    valid_n = np.sum(valid_mask)
    if valid_n == 0:
        return 0.0, 0.0, area, 0.0, 0.0, pd.DataFrame(), x_rand, y_rand, z_eval, n
        
    z_valid = z_eval[valid_mask]
    integral = area * np.mean(z_valid)
    
    var = np.var(z_valid, ddof=1) if valid_n > 1 else 0.0
    std_dev = np.sqrt(var)
    std_error = area * np.sqrt(var / valid_n)
    
    alpha = 1 - (conf_level / 100.0)
    z = st_stats.norm.ppf(1 - alpha/2)
    margin = z * std_error
    ic_lower = integral - margin
    ic_upper = integral + margin
    
    tabla = []
    for i in range(min(10, n)):
        tabla.append({
            "Punto i": i+1,
            "x_i": round(x_rand[i], precision),
            "y_i": round(y_rand[i], precision),
            "f(x_i, y_i)": round(z_eval[i], precision) if not np.isnan(z_eval[i]) else "Indefinido",
        })
    df_tabla = pd.DataFrame(tabla)
    
    invalid_count = n - valid_n
    return integral, std_error, std_dev, area, ic_lower, ic_upper, df_tabla, x_rand, y_rand, z_eval, invalid_count


# --- MÉTODOS DE RAÍCES ---

def formatear_error(valor):
    if valor == 0:
        return f"0.{"" .zfill(precision)}"
    if abs(valor) < 0.0001:
        s = f"{valor:.4e}"
        base, exp = s.split('e')
        exp_int = int(exp)  # Convierte '-05' a -5
        superscripts = {'-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}
        exp_sup = ''.join(superscripts.get(c, c) for c in str(exp_int))
        return f"{base} × 10{exp_sup}"
    return f"{valor:{fmt}}"


def metodo_biseccion(f_str, a, b, tol, max_iter):
    history = []
    fa, fb = evaluar_f(f_str, a), evaluar_f(f_str, b)
    if fa is None or fb is None or fa * fb >= 0: return None, "error_signos", 0, 100
    x_ant = a
    for i in range(max_iter):
        c = (a + b) / 2
        fc = evaluar_f(f_str, c)
        err = abs(c - x_ant) / abs(c) * 100 if abs(c) > 1e-12 else 0
        history.append({"Iter": i+1, "a": a, "b": b, "x_n": c, "f(x_n)": fc, "Error (%)": err})
        if abs(fc) < 1e-12 or err < tol: return pd.DataFrame(history), "convergencia", c, err
        if fa * fc < 0: b = c
        else: a, fa = c, fc
        x_ant = c
    return pd.DataFrame(history), "limite", x_ant, err

def metodo_newton_raphson(f_str, x0, tol, max_iter):
    history = []
    x_n = x0
    for i in range(max_iter):
        fx = evaluar_f(f_str, x_n)
        dfx = calcular_derivada_robusta(f_str, x_n)
        if dfx == 0: break
        x_next = x_n - fx / dfx
        err = abs(x_next - x_n) / abs(x_next) * 100 if abs(x_next) > 1e-12 else 0
        history.append({"Iter": i+1, "x_n": x_n, "f(x_n)": fx, "f'(x_n)": dfx, "Error (%)": "" if i == 0 else err})
        if i > 0 and err < tol: return pd.DataFrame(history), "convergencia", x_next, err
        x_n = x_next
    return pd.DataFrame(history), "limite", x_n, err

# --- PUNTO FIJO Y AITKEN ---

def metodo_punto_fijo(g_str, x0, tol, max_iter):
    """Iteración de Punto Fijo clásico."""
    history = []
    xn = x0
    
    for i in range(max_iter):
        xn1 = evaluar_f(g_str, xn)
        if xn1 is None:
            return pd.DataFrame(history), "error_eval", xn, 100.0
            
        error_pct = abs((xn1 - xn) / xn1) * 100 if abs(xn1) > 1e-12 else 0.0
        
        row = {
            "n": i,
            "Xn": xn,
            "Xn+1": xn1,
            "Error %": "" if i == 0 else error_pct
        }
        history.append(row)
        
        if i > 0 and error_pct < tol:
            return pd.DataFrame(history), "convergencia", xn1, error_pct
            
        xn = xn1
        
    return pd.DataFrame(history), "limite", xn, error_pct if i > 0 else 100.0



def metodo_punto_fijo_aitken(g_str, x0, tol, max_iter):
    """Iteración de Punto Fijo con aceleración de Aitken (Δ²)."""
    history = []
    xn = x0

    for i in range(max_iter):
        # Xn+1 = g(Xn)
        xn1 = evaluar_f(g_str, xn)
        if xn1 is None:
            return pd.DataFrame(history), "error_eval", xn, 100.0

        # Xn+2 = g(Xn+1)
        xn2 = evaluar_f(g_str, xn1)
        if xn2 is None:
            return pd.DataFrame(history), "error_eval", xn1, 100.0

        # Aitken: Xn* = Xn - (Xn+1 - Xn)^2 / (Xn+2 - 2*Xn+1 + Xn)
        denom = xn2 - 2 * xn1 + xn
        x_hat = None
        error_pct = None
        if abs(denom) > 1e-15:
            x_hat = xn - (xn1 - xn) ** 2 / denom
            if abs(x_hat) > 1e-12:
                error_pct = abs((x_hat - xn) / x_hat) * 100
            else:
                error_pct = 0.0

        row = {
            "n": i,
            "Xn": xn,
            "Xn+1": xn1,
            "Xn+2": xn2,
            "Xn*": x_hat if x_hat is not None else "",
            "Error %": error_pct if error_pct is not None else "",
        }
        history.append(row)

        # Criterio de convergencia sobre x_hat
        if error_pct is not None and error_pct < tol:
            return pd.DataFrame(history), "convergencia", x_hat, error_pct

        # Siguiente iteración: Xn = Xn+2 (avanzar secuencia punto fijo)
        xn = xn2

    ultimo_xhat = x_hat if x_hat is not None else xn
    ultimo_err  = error_pct if error_pct is not None else 100.0
    return pd.DataFrame(history), "limite", ultimo_xhat, ultimo_err


# --- RUNGE-KUTTA: EVALUADOR DE EDO f(x, y) ---

def evaluar_edo(f_str, x_val, y_val):
    """Evalúa f(x, y) para EDOs dy/dx = f(x, y)"""
    try:
        f_proc = f_str.replace("^", "**").replace("sen", "sin").replace("ln", "log")
        f_proc = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', f_proc)
        contexto = {
            "np": np, "x": x_val, "y": y_val, "t": x_val,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
            "pi": np.pi, "e": np.e, "abs": abs
        }
        res = eval(f_proc, {"__builtins__": None}, contexto)
        if np.isnan(float(res)) or np.isinf(float(res)):
            return None
        return float(res)
    except Exception:
        return None


def metodo_euler(f_str, x0, y0, h, n_steps):
    """Método de Euler (RK1)"""
    tabla = []
    x, y = x0, y0
    for i in range(n_steps + 1):
        f_val = evaluar_edo(f_str, x, y) if i < n_steps else None
        row = {"i": i, "xᵢ": x, "yᵢ": y}
        if f_val is not None:
            row["f(xᵢ,yᵢ)"] = f_val
            row["yᵢ₊₁"] = y + h * f_val
        tabla.append(row)
        if i < n_steps and f_val is not None:
            y = y + h * f_val
            x = x + h
        elif i < n_steps:
            break
    xs = [r["xᵢ"] for r in tabla]
    ys = [r["yᵢ"] for r in tabla]
    return pd.DataFrame(tabla), xs, ys


def metodo_rk2(f_str, x0, y0, h, n_steps, variante="Heun"):
    """Método de Runge-Kutta Orden 2 (Heun, Punto Medio, Ralston)"""
    if variante == "Heun":
        a2, b1, b2, p1 = 1.0, 0.5, 0.5, 1.0
    elif variante == "Punto Medio":
        a2, b1, b2, p1 = 0.5, 0.0, 1.0, 0.5
    else:  # Ralston
        a2, b1, b2, p1 = 2/3, 0.25, 0.75, 2/3

    tabla = []
    x, y = x0, y0
    for i in range(n_steps + 1):
        row = {"i": i, "xᵢ": x, "yᵢ": y}
        if i < n_steps:
            k1 = evaluar_edo(f_str, x, y)
            if k1 is None: break
            k2 = evaluar_edo(f_str, x + p1 * h, y + p1 * h * k1)
            if k2 is None: break
            row["k₁"] = k1
            row["k₂"] = k2
            y_next = y + h * (b1 * k1 + b2 * k2)
            row["yᵢ₊₁"] = y_next
            y = y_next
            x = x + h
        tabla.append(row)
    if len(tabla) <= n_steps:
        tabla.append({"i": len(tabla), "xᵢ": x, "yᵢ": y})
    xs = [r["xᵢ"] for r in tabla]
    ys = [r["yᵢ"] for r in tabla]
    return pd.DataFrame(tabla), xs, ys


def metodo_rk4(f_str, x0, y0, h, n_steps):
    """Método de Runge-Kutta Orden 4 (Clásico)"""
    tabla = []
    x, y = x0, y0
    for i in range(n_steps + 1):
        row = {"i": i, "xᵢ": x, "yᵢ": y}
        if i < n_steps:
            k1 = evaluar_edo(f_str, x, y)
            if k1 is None: break
            k2 = evaluar_edo(f_str, x + h/2, y + h/2 * k1)
            if k2 is None: break
            k3 = evaluar_edo(f_str, x + h/2, y + h/2 * k2)
            if k3 is None: break
            k4 = evaluar_edo(f_str, x + h, y + h * k3)
            if k4 is None: break
            row["k₁"] = k1
            row["k₂"] = k2
            row["k₃"] = k3
            row["k₄"] = k4
            y_next = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
            row["yᵢ₊₁"] = y_next
            y = y_next
            x = x + h
        tabla.append(row)
    if len(tabla) <= n_steps:
        tabla.append({"i": len(tabla), "xᵢ": x, "yᵢ": y})
    xs = [r["xᵢ"] for r in tabla]
    ys = [r["yᵢ"] for r in tabla]
    return pd.DataFrame(tabla), xs, ys


def metodo_rk4_sistema(f1_str, f2_str, x0, y1_0, y2_0, h, n_steps):
    """RK4 para sistemas de 2 EDOs: dy1/dx = f1(x,y1,y2), dy2/dx = f2(x,y1,y2)"""
    def eval_sys(f_str, x_val, y1_val, y2_val):
        try:
            f_proc = f_str.replace("^", "**").replace("sen", "sin").replace("ln", "log")
            f_proc = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', f_proc)
            ctx = {
                "np": np, "x": x_val, "t": x_val,
                "y1": y1_val, "y2": y2_val, "y": y1_val, "z": y2_val,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
                "pi": np.pi, "e": np.e, "abs": abs
            }
            res = eval(f_proc, {"__builtins__": None}, ctx)
            return float(res)
        except:
            return None

    tabla = []
    x, y1, y2 = x0, y1_0, y2_0
    for i in range(n_steps + 1):
        row = {"i": i, "xᵢ": x, "y₁ᵢ": y1, "y₂ᵢ": y2}
        if i < n_steps:
            k1_1 = eval_sys(f1_str, x, y1, y2)
            k1_2 = eval_sys(f2_str, x, y1, y2)
            if k1_1 is None or k1_2 is None: break
            k2_1 = eval_sys(f1_str, x+h/2, y1+h/2*k1_1, y2+h/2*k1_2)
            k2_2 = eval_sys(f2_str, x+h/2, y1+h/2*k1_1, y2+h/2*k1_2)
            if k2_1 is None or k2_2 is None: break
            k3_1 = eval_sys(f1_str, x+h/2, y1+h/2*k2_1, y2+h/2*k2_2)
            k3_2 = eval_sys(f2_str, x+h/2, y1+h/2*k2_1, y2+h/2*k2_2)
            if k3_1 is None or k3_2 is None: break
            k4_1 = eval_sys(f1_str, x+h, y1+h*k3_1, y2+h*k3_2)
            k4_2 = eval_sys(f2_str, x+h, y1+h*k3_1, y2+h*k3_2)
            if k4_1 is None or k4_2 is None: break
            row["k₁⁽¹⁾"]=k1_1; row["k₂⁽¹⁾"]=k2_1; row["k₃⁽¹⁾"]=k3_1; row["k₄⁽¹⁾"]=k4_1
            row["k₁⁽²⁾"]=k1_2; row["k₂⁽²⁾"]=k2_2; row["k₃⁽²⁾"]=k3_2; row["k₄⁽²⁾"]=k4_2
            y1 = y1 + h/6*(k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
            y2 = y2 + h/6*(k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
            x = x + h
        tabla.append(row)
    if len(tabla) <= n_steps:
        tabla.append({"i": len(tabla), "xᵢ": x, "y₁ᵢ": y1, "y₂ᵢ": y2})
    xs = [r["xᵢ"] for r in tabla]
    y1s = [r["y₁ᵢ"] for r in tabla]
    y2s = [r["y₂ᵢ"] for r in tabla]
    return pd.DataFrame(tabla), xs, y1s, y2s


def obtener_solucion_exacta_edo(f_str, x0, y0, xs_eval):
    """Obtiene solución exacta usando scipy.integrate.solve_ivp"""
    def f_scipy(t, y):
        val = evaluar_edo(f_str, t, y[0])
        return [val if val is not None else 0.0]
    try:
        from scipy.integrate import solve_ivp
        x_max = max(xs_eval)
        sol = solve_ivp(f_scipy, [x0, x_max], [y0], t_eval=xs_eval, method='RK45', rtol=1e-12, atol=1e-14)
        if sol.success:
            return sol.y[0]
    except:
        pass
    return None


# --- INTERFAZ ---

st.sidebar.header("Configuración")
mostrar_formulas = st.sidebar.checkbox("Visor Fórmulas (Ctrl+Shift+F)", value=False)
precision = st.sidebar.slider("Precisión Decimal", 1, 12, 6)
fmt = f".{precision}f"

def format_df(df_in):
    return df_in.style.format(lambda x: f"{x:.{precision}f}" if isinstance(x, float) else x)


metodo_sel = st.sidebar.selectbox("Selecciona Método",
    ["Bisección", "Newton-Raphson", "Punto Fijo", "Punto Fijo y Aitken", "Interpolación Lagrange", "Diferencias Centrales", "Rectángulo Medio", "Trapecios", "Simpson 1/3", "Simpson 3/8", "Montecarlo", "Montecarlo Doble", "Runge-Kutta"])

# --- BOTONERA DE FUNCIONES MATEMÁTICAS ---

# Defaults por método para el campo f(x) principal
_FX_DEFAULTS = {
    "Bisección": "x**2 - 2",
    "Newton-Raphson": "x**2 - 2",
    "Punto Fijo": "cos(x)",
    "Punto Fijo y Aitken": "cos(x)",
    "Simpson 1/3": "x**3",
    "Simpson 3/8": "x**3",
    "Trapecios": "x**2",
    "Rectángulo Medio": "x**2",
    "Montecarlo": "sin(x)",
    "Montecarlo Doble": "x**2 + y**2",
    "Runge-Kutta": "x + y",
}

# Detectar cambio de método y resetear el input
if "prev_metodo" not in st.session_state:
    st.session_state.prev_metodo = metodo_sel
if st.session_state.prev_metodo != metodo_sel:
    st.session_state.prev_metodo = metodo_sel
    if "fx_input" in st.session_state:
        del st.session_state["fx_input"]

# Inicializar fx_input con el default del método actual
if "fx_input" not in st.session_state:
    st.session_state.fx_input = _FX_DEFAULTS.get(metodo_sel, "x**2 - 2")

# Botones: (etiqueta visible, texto insertado)
_MATH_BUTTONS = [
    ("√x",     "sqrt("),
    ("eˣ",     "exp("),
    ("ln",     "log("),
    ("log₁₀",  "log10("),
    ("sin",    "sin("),
    ("cos",    "cos("),
    ("tan",    "tan("),
    ("xⁿ",    "**"),
    ("π",      "pi"),
    ("e",      "e"),
    ("(",      "("),
    (")",      ")"),
]

def _insertar_texto(texto):
    """Callback: concatena texto al final del campo f(x)."""
    st.session_state.fx_input = st.session_state.fx_input + texto

def _render_botonera():
    """Dibuja la cuadrícula de botones matemáticos (4 por fila) con estilo compacto."""
    # CSS para botones compactos tipo calculadora
    st.markdown("""
    <style>
    /* Botones de la botonera matemática: compactos */
    div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) button {
        padding: 0.15rem 0.3rem !important;
        font-size: 0.78rem !important;
        min-height: 0 !important;
        line-height: 1.3 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) {
        gap: 0.25rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(
        "<p style='margin:0 0 2px 0;font-size:0.82em;color:#888;'>📐 Insertar función:</p>",
        unsafe_allow_html=True,
    )
    cols_per_row = 4
    for row_start in range(0, len(_MATH_BUTTONS), cols_per_row):
        row_btns = _MATH_BUTTONS[row_start : row_start + cols_per_row]
        cols = st.columns(len(row_btns))
        for col, (label, insert_text) in zip(cols, row_btns):
            col.button(
                label,
                key=f"mbtn_{label}",
                on_click=_insertar_texto,
                args=(insert_text,),
                use_container_width=True,
            )
    st.markdown("<hr style='margin:4px 0 8px 0;border:none;border-top:1px solid #333;'>", unsafe_allow_html=True)

# Métodos que NO usan un campo f(x) principal
_METODOS_SIN_FX = {"Interpolación Lagrange", "Diferencias Centrales"}

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Entrada")
    # Mostrar botonera solo para métodos con campo de función
    if metodo_sel not in _METODOS_SIN_FX:
        _render_botonera()
    if "Interpolación" in metodo_sel or "Diferencias" in metodo_sel:
        st.info("Formato: x, y (un punto por línea)")
        default_pts = "1, 1\n4, 2"
        puntos_input = st.text_area("Puntos:", value=default_pts, height=150)
        
        if metodo_sel == "Interpolación Lagrange":
            func_teorica = st.text_input("Función real f(x):", value="sqrt(x)")
            x_eval = st.text_input("Evaluar en x (Epsilon):", value="2")
        elif metodo_sel == "Diferencias Centrales":
            x_buscar_dc = st.text_input("Buscar valor de x (opcional):", value="")
            
        try:
            lineas = [l.strip() for l in puntos_input.strip().split('\n') if l.strip()]
            x_in_strs = [l.split(',')[0].strip() for l in lineas]
            y_in_strs = [l.split(',')[1].strip() for l in lineas]
            x_in_num = np.array([float(sp.sympify(x).evalf()) for x in x_in_strs])
            y_in_num = np.array([float(sp.sympify(y).evalf()) for y in y_in_strs])
        except:
            st.error("Error al leer puntos.")
    elif metodo_sel == "Simpson 1/3":
        func_input = st.text_input("f(x):", key="fx_input")
        a_simp_str = st.text_input("Límite inferior a", value="0", help="Podés escribir expresiones como pi/2, sqrt(2), 2*pi, etc.")
        b_simp_str = st.text_input("Límite superior b", value="1", help="Podés escribir expresiones como pi/2, sqrt(2), 2*pi, etc.")
        n_simp = int(st.number_input("Nº subintervalos n (debe ser par)", value=4, min_value=2, step=2))
        if n_simp % 2 != 0:
            st.warning("n debe ser par — se ajustará a n+1 automáticamente.")
        usar_xi_13 = st.checkbox("Definir ξ para el error", value=False, key="xi_chk_13", help="Evaluar f⁴(ξ) en un punto exacto en vez de usar el máximo del intervalo.")
        xi_val_13 = None
        if usar_xi_13:
            xi_str_13 = st.text_input("Punto ξ (epsilon)", value="0.5", key="xi_input_13", help="Expresión válida: 0.5, pi/4, sqrt(2), etc.")
            try:
                xi_val_13 = float(sp.sympify(xi_str_13).evalf())
            except:
                st.error("Valor de ξ inválido.")
                xi_val_13 = None
        try:
            a_simp = float(sp.sympify(a_simp_str).evalf())
            b_simp = float(sp.sympify(b_simp_str).evalf())
        except:
            st.error("Límites inválidos. Usá expresiones como: pi/2, sqrt(2), 1.5, 2*pi")
            a_simp, b_simp = 0.0, 1.0
    elif metodo_sel == "Simpson 3/8":
        func_input = st.text_input("f(x):", key="fx_input")
        a_simp38_str = st.text_input("Límite inferior a", value="0", help="Podés escribir expresiones como pi/2, sqrt(2), 2*pi, etc.")
        b_simp38_str = st.text_input("Límite superior b", value="1", help="Podés escribir expresiones como pi/2, sqrt(2), 2*pi, etc.")
        n_simp38 = int(st.number_input("Nº subintervalos n (múltiplo de 3)", value=3, min_value=3, step=3))
        if n_simp38 % 3 != 0:
            st.warning("n debe ser múltiplo de 3 — se ajustará automáticamente.")
        usar_xi_38 = st.checkbox("Definir ξ para el error", value=False, key="xi_chk_38", help="Evaluar f⁴(ξ) en un punto exacto en vez de usar el máximo del intervalo.")
        xi_val_38 = None
        if usar_xi_38:
            xi_str_38 = st.text_input("Punto ξ (epsilon)", value="0.5", key="xi_input_38", help="Expresión válida: 0.5, pi/4, sqrt(2), etc.")
            try:
                xi_val_38 = float(sp.sympify(xi_str_38).evalf())
            except:
                st.error("Valor de ξ inválido.")
                xi_val_38 = None
        try:
            a_simp38 = float(sp.sympify(a_simp38_str).evalf())
            b_simp38 = float(sp.sympify(b_simp38_str).evalf())
        except:
            st.error("Límites inválidos. Usá expresiones como: pi/2, sqrt(2), 1.5, 2*pi")
            a_simp38, b_simp38 = 0.0, 1.0
    elif metodo_sel == "Trapecios":
        func_input = st.text_input("f(x):", key="fx_input")
        a_trap_str = st.text_input("Límite inferior a", value="0", help="Podés escribir expresiones como pi/2, sqrt(2), 2*pi, etc.")
        b_trap_str = st.text_input("Límite superior b", value="1", help="Podés escribir expresiones como pi/2, sqrt(2), 2*pi, etc.")
        n_trap = int(st.number_input("Nº subintervalos n", value=4, min_value=1, step=1))
        usar_xi_trap = st.checkbox("Definir ξ para el error", value=False, key="xi_chk_trap", help="Evaluar f''(ξ) en un punto exacto en vez de usar el máximo del intervalo.")
        xi_val_trap = None
        if usar_xi_trap:
            xi_str_trap = st.text_input("Punto ξ (epsilon)", value="0.5", key="xi_input_trap", help="Expresión válida: 0.5, pi/4, sqrt(2), etc.")
            try:
                xi_val_trap = float(sp.sympify(xi_str_trap).evalf())
            except:
                st.error("Valor de ξ inválido.")
                xi_val_trap = None
        try:
            a_trap = float(sp.sympify(a_trap_str).evalf())
            b_trap = float(sp.sympify(b_trap_str).evalf())
        except:
            st.error("Límites inválidos. Usá expresiones como: pi/2, sqrt(2), 1.5, 2*pi")
            a_trap, b_trap = 0.0, 1.0
    elif metodo_sel == "Rectángulo Medio":
        func_input = st.text_input("f(x):", key="fx_input")
        a_rect_str = st.text_input("Límite inferior a", value="0", help="Podés escribir expresiones como pi/2, sqrt(2), 2*pi, etc.")
        b_rect_str = st.text_input("Límite superior b", value="1", help="Podés escribir expresiones como pi/2, sqrt(2), 2*pi, etc.")
        n_rect = int(st.number_input("Nº subintervalos n", value=4, min_value=1, step=1))
        try:
            a_rect = float(sp.sympify(a_rect_str).evalf())
            b_rect = float(sp.sympify(b_rect_str).evalf())
        except:
            st.error("Límites inválidos. Usá expresiones como: pi/2, sqrt(2), 1.5, 2*pi")
            a_rect, b_rect = 0.0, 1.0
    elif metodo_sel == "Montecarlo":
        func_input = st.text_input("f(x):", key="fx_input")
        a_mc_str = st.text_input("Límite inferior a", value="0", help="Usar expresiones: pi/2, sqrt(2)")
        b_mc_str = st.text_input("Límite superior b", value="pi", help="Usar expressions: pi/2, sqrt(2)")
        n_mc = int(st.number_input("Cantidad de Puntos N", value=10000, step=1000))
        conf_mc = st.number_input("Nivel de Confianza (%)", value=95.0, min_value=1.0, max_value=99.9)
        
        col_m1, col_m2 = st.columns(2)
        use_seed_mc = col_m1.checkbox("Fijar Semilla", value=True, key="smc1")
        seed_mc = col_m2.number_input("Semilla", value=42, step=1, key="smc2") if use_seed_mc else None
        antithetic_mc = st.checkbox("Variables Antitéticas", value=False, help="Técnica de reducción de varianza.", key="amc1")
        
        try:
            a_mc = float(sp.sympify(a_mc_str).evalf())
            b_mc = float(sp.sympify(b_mc_str).evalf())
        except:
            st.error("Límites inválidos.")
            a_mc, b_mc = 0.0, 3.14159
    elif metodo_sel == "Montecarlo Doble":
        func_input = st.text_input("f(x, y):", key="fx_input")
        col_ax, col_bx = st.columns(2)
        a_x_mc_str = col_ax.text_input("Lím. inf. X (a)", value="0")
        b_x_mc_str = col_bx.text_input("Lím. sup. X (b)", value="1")
        col_ay, col_by = st.columns(2)
        a_y_mc_str = col_ay.text_input("Lím. inf. Y (c)", value="0")
        b_y_mc_str = col_by.text_input("Lím. sup. Y (d)", value="2")
        n_mc2 = int(st.number_input("Cantidad de Puntos N", value=10000, step=1000))
        conf_mc2 = st.number_input("Nivel de Confianza (%)", value=95.0, min_value=1.0, max_value=99.9)
        
        col_md1, col_md2 = st.columns(2)
        use_seed_mc2 = col_md1.checkbox("Fijar Semilla", value=True, key="smcd1")
        seed_mc2 = col_md2.number_input("Semilla", value=42, step=1, key="smcd2") if use_seed_mc2 else None
        antithetic_mc2 = st.checkbox("Variables Antitéticas", value=False, help="Técnica de reducción de varianza.", key="amcd1")
        
        try:
            a_x_mc = float(sp.sympify(a_x_mc_str).evalf())
            b_x_mc = float(sp.sympify(b_x_mc_str).evalf())
            a_y_mc = float(sp.sympify(a_y_mc_str).evalf())
            b_y_mc = float(sp.sympify(b_y_mc_str).evalf())
        except:
            st.error("Límites inválidos.")
            a_x_mc, b_x_mc, a_y_mc, b_y_mc = 0.0, 1.0, 0.0, 2.0
    elif metodo_sel in ["Punto Fijo", "Punto Fijo y Aitken"]:
        func_input = st.text_input("g(x):", key="fx_input", help="Función de iteración de punto fijo. Ej: cos(x), (x + 2/x)/2, sqrt(2 + x)")
        x0_pf = st.number_input("x₀ (valor inicial)", value=1.0, format=f"%.{precision}f")
        tol_pf_str = st.text_input("Tolerancia (%)", value="1e-3")
        try:
            tol_pf = float(sp.sympify(tol_pf_str).evalf())
        except:
            tol_pf = 0.001
        iter_pf = st.slider("Max Iteraciones", 5, 200, 50)
    elif metodo_sel == "Runge-Kutta":
        rk_tipo = st.radio("Tipo de EDO", ["EDO Simple", "Sistema de 2 EDOs"], horizontal=True)
        if rk_tipo == "EDO Simple":
            rk_orden = st.selectbox("Orden del Método", ["Euler (Orden 1)", "RK2 (Orden 2)", "RK4 (Orden 4)"])
            rk_variante = None
            if "RK2" in rk_orden:
                rk_variante = st.selectbox("Variante RK2", ["Heun", "Punto Medio", "Ralston"])
            func_input = st.text_input("dy/dx = f(x, y):", key="fx_input", help="Usá 'x' e 'y' como variables. Ej: x + y, -2*x*y, sin(x)*y")
            col_ci1, col_ci2 = st.columns(2)
            rk_x0_str = col_ci1.text_input("x₀ (valor inicial)", value="0.0")
            rk_y0_str = col_ci2.text_input("y₀ = y(x₀)", value="1.0")
            col_h, col_n = st.columns(2)
            rk_h_str = col_h.text_input("Paso h", value="0.1")
            rk_n = int(col_n.number_input("Nº de pasos", value=10, min_value=1, step=1))
            rk_exacta = st.text_input("Solución exacta y(x) (opcional):", value="", help="Ej: 2*exp(x) - x - 1. Dejá vacío para comparar con scipy.")
            try:
                rk_x0 = float(sp.sympify(rk_x0_str).evalf())
                rk_y0 = float(sp.sympify(rk_y0_str).evalf())
                rk_h = float(sp.sympify(rk_h_str).evalf())
            except:
                st.error("Valores iniciales o paso inválidos.")
                rk_x0, rk_y0, rk_h = 0.0, 1.0, 0.1
        else:
            func_input_1 = st.text_input("dy₁/dx = f₁(x, y₁, y₂):", key="fx_input", help="Usá x, y1, y2 (o y, z)")
            func_input_2 = st.text_input("dy₂/dx = f₂(x, y₁, y₂):", value="-y1", help="Usá x, y1, y2 (o y, z)")
            col_ci1, col_ci2, col_ci3 = st.columns(3)
            rk_x0_str = col_ci1.text_input("x₀", value="0.0")
            rk_y1_0_str = col_ci2.text_input("y₁(x₀)", value="0.0")
            rk_y2_0_str = col_ci3.text_input("y₂(x₀)", value="1.0")
            col_h, col_n = st.columns(2)
            rk_h_str = col_h.text_input("Paso h", value="0.1")
            rk_n = int(col_n.number_input("Nº de pasos", value=10, min_value=1, step=1))
            try:
                rk_x0 = float(sp.sympify(rk_x0_str).evalf())
                rk_y1_0 = float(sp.sympify(rk_y1_0_str).evalf())
                rk_y2_0 = float(sp.sympify(rk_y2_0_str).evalf())
                rk_h = float(sp.sympify(rk_h_str).evalf())
            except:
                st.error("Valores iniciales o paso inválidos.")
                rk_x0, rk_y1_0, rk_y2_0, rk_h = 0.0, 0.0, 1.0, 0.1
    else:
        func_input = st.text_input("f(x):", key="fx_input")
        if metodo_sel == "Bisección":
            a_in, b_in = st.number_input("a", value=0.0), st.number_input("b", value=2.0)
        else:
            x0_in = st.number_input("x0", value=1.0)
        tol_in_str = st.text_input("Tolerancia (%)", value="1e-3")
        try:
            tol_in = float(sp.sympify(tol_in_str).evalf())
        except:
            tol_in = 0.001
        iter_in = st.slider("Max Iter", 5, 100, 20)

    ejecutar = st.button("Calcular")

with col2:
    if ejecutar:
        if metodo_sel == "Interpolación Lagrange":
            poly, lista_L, x_exact, y_exact = calcular_lagrange_avanzado(x_in_strs, y_in_strs)
            n_pts = len(x_exact)
            x_sym = sp.symbols('x')
            if mostrar_formulas:
                st.subheader("Fórmulas")
                st.latex(
                    r"P(x) = \sum_{i=0}^{n} y_i \, L_i(x)"
                )
                st.latex(
                    r"L_i(x) = \prod_{\substack{j=0 \\ j \neq i}}^{n} \frac{x - x_j}{x_i - x_j}"
                )
                with st.expander("📖 Notación"):
                    st.markdown("""
| Símbolo | Significado |
|---|---|
| $P(x)$ | Polinomio interpolante de Lagrange |
| $n$ | Grado del polinomio (número de puntos − 1) |
| $x_i,\\, y_i$ | Nodos e imágenes conocidos |
| $L_i(x)$ | $i$-ésimo polinomio base de Lagrange |
| $x$ | Punto en el que se evalúa $P(x)$ |
""")

            # --- PROCEDIMIENTO PASO A PASO ---
            with st.expander("📋 Desarrollo paso a paso", expanded=False):
                # Mostrar cada L_i(x)
                for i in range(n_pts):
                    st.markdown(f"**Polinomio base $L_{{{i}}}(x)$:**")
                    # Forma de producto (sin simplificar)
                    num_parts = []
                    den_parts = []
                    for j in range(n_pts):
                        if i != j:
                            num_parts.append(f"(x - {sp.latex(x_exact[j])})")
                            den_parts.append(f"({sp.latex(x_exact[i])} - {sp.latex(x_exact[j])})")
                    producto_latex = r"L_{" + str(i) + r"}(x) = \frac{" + r" \cdot ".join(num_parts) + r"}{" + r" \cdot ".join(den_parts) + r"}"
                    st.latex(producto_latex)
                    # Forma simplificada
                    st.latex(f"L_{{{i}}}(x) = {sp.latex(lista_L[i])}")
                    st.markdown("---")

                # Mostrar armado de P(x) = Σ yi * Li(x)
                st.markdown("**Construcción de $P(x)$:**")
                terminos_latex = []
                for i in range(n_pts):
                    yi_latex = sp.latex(y_exact[i])
                    li_latex = sp.latex(lista_L[i])
                    terminos_latex.append(f"({yi_latex}) \\cdot ({li_latex})")
                suma_latex = r"P(x) = " + " + ".join(terminos_latex)
                st.latex(suma_latex)

                if func_teorica:
                    x_ev_sym = sp.sympify(x_eval)
                    val_p_tmp = poly.subs(sp.symbols('x'), x_ev_sym)
                    x_eval_float_tmp = float(x_ev_sym.evalf())
                    v_real_tmp = evaluar_f(func_teorica, x_eval_float_tmp)
                    
                    if v_real_tmp is not None:
                        _err_local = abs(float(val_p_tmp.evalf()) - v_real_tmp)
                        st.markdown("**Error Local:**")
                        bloque_err = (
                            f"E = |P(x) - f(x)|\n"
                            f"E = |{float(val_p_tmp.evalf()):{fmt}} - {v_real_tmp:{fmt}}|\n"
                            f"E = {_err_local:{fmt}}"
                        )
                        st.code(bloque_err, language="text")
                        
                    try:
                        import math
                        f_sym_teorica = sp.sympify(func_teorica.replace("^", "**").replace("sen", "sin").replace("ln", "log"))
                        deriv_n = sp.diff(f_sym_teorica, sp.symbols('x'), n_pts)
                        if not deriv_n.has(sp.Derivative):
                            min_x_val = min(min(x_in_num), x_eval_float_tmp)
                            max_x_val = max(max(x_in_num), x_eval_float_tmp)
                            grid_xi = np.linspace(min_x_val, max_x_val, 200)
                            
                            deriv_n_func = sp.lambdify(sp.symbols('x'), deriv_n, modules=['numpy', 'math'])
                            max_deriv = 0.0
                            for xi_v in grid_xi:
                                try:
                                    val_d = abs(float(deriv_n_func(xi_v)))
                                    if val_d > max_deriv and not np.isnan(val_d) and not np.isinf(val_d):
                                        max_deriv = val_d
                                except:
                                    pass
                            
                            prod_str_list = []
                            prod_val = 1.0
                            for xi_n in x_in_num:
                                dif = abs(x_eval_float_tmp - xi_n)
                                prod_val *= dif
                                prod_str_list.append(f"|{x_eval_float_tmp:{fmt}} - {xi_n:{fmt}}|")
                                
                            prod_str = " · ".join(prod_str_list)
                            fact_n = math.factorial(n_pts)
                            cota_error_tmp = (max_deriv / fact_n) * prod_val
                            
                            st.markdown("**Cota de Error (Error Global):**")
                            bloque_cota = (
                                f"E_G <= (max|f^({n_pts})(ξ)| / {n_pts}!) · Π |x - x_i|\n"
                                f"Derivada {n_pts}-ésima: {deriv_n}\n"
                                f"max|f^({n_pts})(ξ)| en [{min_x_val:{fmt}}, {max_x_val:{fmt}}] ≈ {max_deriv:{fmt}}\n\n"
                                f"E_G <= ({max_deriv:{fmt}} / {fact_n}) · ({prod_str})\n"
                                f"E_G <= ({max_deriv / fact_n:{fmt}}) · {prod_val:{fmt}}\n"
                                f"E_G <= {cota_error_tmp:{fmt}}"
                            )
                            st.code(bloque_cota, language="text")
                    except Exception:
                        pass

            st.subheader("Resultado")
            st.latex(f"P(x) = {sp.latex(poly)}")

            # Evaluación y Error
            x_ev_sym = sp.sympify(x_eval)
            val_p = poly.subs(sp.symbols('x'), x_ev_sym)
            st.info(f"Evaluación: P({x_eval}) = {float(val_p.evalf()):{fmt}}")
            st.latex(f"P({sp.latex(x_ev_sym)}) = {sp.latex(sp.simplify(val_p))}")
            
            if func_teorica:
                x_eval_float = float(x_ev_sym.evalf())
                v_real = evaluar_f(func_teorica, x_eval_float)
                
                cota_error_val = None
                try:
                    import math
                    f_sym_teorica = sp.sympify(func_teorica.replace("^", "**").replace("sen", "sin").replace("ln", "log"))
                    deriv_n = sp.diff(f_sym_teorica, sp.symbols('x'), n_pts)
                    if not deriv_n.has(sp.Derivative):
                        min_x_val = min(min(x_in_num), x_eval_float)
                        max_x_val = max(max(x_in_num), x_eval_float)
                        grid_xi = np.linspace(min_x_val, max_x_val, 200)
                        
                        deriv_n_func = sp.lambdify(sp.symbols('x'), deriv_n, modules=['numpy', 'math'])
                        max_deriv = 0.0
                        for xi_v in grid_xi:
                            try:
                                val_d = abs(float(deriv_n_func(xi_v)))
                                if val_d > max_deriv and not np.isnan(val_d) and not np.isinf(val_d):
                                    max_deriv = val_d
                            except:
                                pass
                        
                        prod_val = 1.0
                        for xi_n in x_in_num:
                            prod_val *= abs(x_eval_float - xi_n)
                            
                        fact_n = math.factorial(n_pts)
                        cota_error_val = (max_deriv / fact_n) * prod_val
                except:
                    pass

                if v_real is not None:
                    err_local = abs(float(val_p.evalf()) - v_real)
                    if cota_error_val is not None:
                        col_m1, col_m2 = st.columns(2)
                        col_m1.metric("Error Local  |P(x) − f(x)|", f"{err_local:{fmt}}")
                        col_m2.metric("Cota de Error Global", f"{cota_error_val:{fmt}}")
                    else:
                        st.metric("Error Local  |P(x) − f(x)|", f"{err_local:{fmt}}")
                else:
                    if cota_error_val is not None:
                        st.metric("Cota de Error Global", f"{cota_error_val:{fmt}}")

            # Gráfico (solo si no hay variables b, c, etc en y)
            if all(sp.sympify(y).is_number for y in y_in_strs):
                x_range = np.linspace(min(x_in_num)-1, max(x_in_num)+1, 150)
                
                # Evaluación segura iterativa para el gráfico
                y_poly_np = [float(poly.subs(sp.symbols('x'), xi).evalf()) for xi in x_range]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_range, y=y_poly_np, name="P(x)", line=dict(color='#00cfcc')))
                
                # Agregamos la Función Real al gráfico
                if func_teorica:
                    y_real_np = [evaluar_f(func_teorica, xi) for xi in x_range]
                    if None not in y_real_np:
                        fig.add_trace(go.Scatter(x=x_range, y=y_real_np, name="Función Real", line=dict(color='#ff4b4b', dash='dash')))
                
                fig.add_trace(go.Scatter(x=x_in_num, y=y_in_num, mode='markers', name="Nodos", marker=dict(size=10, color='white')))
                fig.update_layout(template="plotly_dark", title="Interpolación vs Función Real")
                st.plotly_chart(fig, use_container_width=True)

        elif metodo_sel == "Diferencias Centrales":
            if mostrar_formulas:
                st.subheader("Fórmulas")
                st.latex(
                    r"f'(x_i) \approx \frac{f(x_{i+1}) - f(x_{i-1})}{2h}"
                )
                st.latex(
                    r"f''(x_i) \approx \frac{f(x_{i+1}) - 2f(x_i) + f(x_{i-1})}{h^2}"
                )
                st.latex(
                    r"\text{Error de truncamiento: } O(h^2)"
                )
                with st.expander("📖 Notación"):
                    st.markdown("""
| Símbolo | Significado |
|---|---|
| $x_i$ | Punto interior donde se estima la derivada |
| $x_{i-1},\\, x_{i+1}$ | Puntos vecinos anterior y siguiente |
| $f(x_i)$ | Valor de la función en $x_i$ |
| $h$ | Espaciado uniforme entre puntos: $h = x_{i+1} - x_i$ |
| $f'(x_i)$ | Aproximación de la primera derivada en $x_i$ |
| $f''(x_i)$ | Aproximación de la segunda derivada en $x_i$ |
| $O(h^2)$ | El error es proporcional a $h^2$ (orden de precisión) |
""")
            st.subheader("Resultado")
            df = metodo_diferencias_centrales(x_in_num, y_in_num)
            if df is not None:
                df_filtrado = df
                _x_val_dc = None
                if x_buscar_dc.strip() != "":
                    try:
                        _x_val_dc = float(sp.sympify(x_buscar_dc).evalf())
                        df_filtrado = df[np.isclose(df["Punto x"].astype(float), _x_val_dc)]
                        if df_filtrado.empty:
                            st.warning(f"El valor x = {_x_val_dc} no se encuentra en los puntos interiores calculados.")
                    except:
                        st.error("Valor de x a buscar inválido.")
                st.table(format_df(df_filtrado))
                # --- DESARROLLO PASO A PASO ---
                with st.expander("📋 Desarrollo paso a paso", expanded=False):
                    _h_dc = x_in_num[1] - x_in_num[0]
                    st.code(f"h = x₁ - x₀ = {x_in_num[1]:{fmt}} - {x_in_num[0]:{fmt}} = {_h_dc:{fmt}}", language="text")
                    for _i_dc in range(1, len(x_in_num) - 1):
                        _xi = x_in_num[_i_dc]
                        if _x_val_dc is not None and not np.isclose(_xi, _x_val_dc):
                            continue
                        _fi_m1 = y_in_num[_i_dc - 1]
                        _fi = y_in_num[_i_dc]
                        _fi_p1 = y_in_num[_i_dc + 1]
                        _d1 = (_fi_p1 - _fi_m1) / (2 * _h_dc)
                        _d2 = (_fi_p1 - 2*_fi + _fi_m1) / (_h_dc**2)
                        bloque = (
                            f"--- Punto x_{_i_dc} = {_xi:{fmt}} ---\n\n"
                            f"f'(x_{_i_dc}) = [f(x_{_i_dc+1}) - f(x_{_i_dc-1})] / (2h)\n"
                            f"f'({_xi:{fmt}}) = [{_fi_p1:{fmt}} - {_fi_m1:{fmt}}] / (2·{_h_dc:{fmt}})\n"
                            f"f'({_xi:{fmt}}) = {_fi_p1 - _fi_m1:{fmt}} / {2*_h_dc:{fmt}}\n"
                            f"f'({_xi:{fmt}}) = {_d1:{fmt}}\n\n"
                            f"f''(x_{_i_dc}) = [f(x_{_i_dc+1}) - 2·f(x_{_i_dc}) + f(x_{_i_dc-1})] / h²\n"
                            f"f''({_xi:{fmt}}) = [{_fi_p1:{fmt}} - 2·{_fi:{fmt}} + {_fi_m1:{fmt}}] / {_h_dc:{fmt}}²\n"
                            f"f''({_xi:{fmt}}) = {_fi_p1 - 2*_fi + _fi_m1:{fmt}} / {_h_dc**2:{fmt}}\n"
                            f"f''({_xi:{fmt}}) = {_d2:{fmt}}\n\n"
                            f"Error O(h²) = {_h_dc:{fmt}}² = {_h_dc**2:{fmt}}"
                        )
                        st.code(bloque, language="text")
            else: st.error("Se necesitan al menos 3 puntos.")

        elif metodo_sel == "Simpson 1/3":
            res_simp = metodo_simpson_13(func_input, a_simp, b_simp, n_simp, xi_punto=xi_val_13)
            if res_simp[0] is not None:
                integral, err_trunc, h_step, df_tabla, indets, _y_s13 = res_simp
                if mostrar_formulas:
                    st.subheader("Fórmulas")
                    st.latex(
                        r"h = \frac{b - a}{n}"
                    )
                    st.latex(
                        r"I \approx \frac{h}{3}\left[f(x_0) + 4f(x_1) + 2f(x_2) + \cdots + 4f(x_{n-1}) + f(x_n)\right]"
                    )
                    st.latex(
                        r"|E_T| \leq \frac{(b-a)\,h^4}{180}\,\max_{\xi \in [a,b]}\left|f^{(4)}(\xi)\right|"
                    )
                    with st.expander("📖 Notación"):
                        st.markdown("""
| Símbolo | Significado |
|---|---|
| $a,\\, b$ | Límites inferior y superior del intervalo de integración |
| $n$ | Número de subintervalos (debe ser par) |
| $h$ | Ancho de cada subintervalo: $h = (b-a)/n$ |
| $x_0, x_1, \\ldots, x_n$ | Nodos equiespaciados: $x_k = a + k\\,h$ |
| $f(x_k)$ | Valor de la función en el nodo $x_k$ |
| $I$ | Valor aproximado de la integral $\\int_a^b f(x)\\,dx$ |
| $E_T$ | Error de truncamiento de la fórmula compuesta |
| $f^{(4)}(\\xi)$ | Cuarta derivada de $f$ en algún punto $\\xi \\in [a,b]$ |
""")
                st.subheader("Resultado")
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Integral ≈", f"{integral:{fmt}}")
                col_r2.metric("Paso h", f"{h_step:{fmt}}")
                col_r3.metric("Error Trunc. |Eₜ|", formatear_error(err_trunc))
                st.dataframe(format_df(df_tabla), use_container_width=True)
                if indets:
                    for xi, val in indets:
                        st.warning(f"⚠️ Indeterminación detectada en x = {xi:{fmt}}. Se resolvió usando límite/L'Hôpital obteniendo: f(x) ≈ {val:{fmt}}")
                # --- DESARROLLO PASO A PASO ---
                with st.expander("📋 Desarrollo paso a paso", expanded=False):
                    st.code(f"h = (b - a) / n = ({b_simp:{fmt}} - {a_simp:{fmt}}) / {n_simp} = {h_step:{fmt}}", language="text")
                    _x_s13 = np.linspace(a_simp, b_simp, n_simp + 1)
                    _terms = []
                    _lines = "Nodos y evaluaciones:\n"
                    if indets:
                        _lines += "(Se aplicó cálculo de límites en los puntos de indeterminación)\n"
                    for _k in range(n_simp + 1):
                        if _k == 0 or _k == n_simp:
                            _c = 1
                        elif _k % 2 != 0:
                            _c = 4
                        else:
                            _c = 2
                        _lines += f"  x_{_k} = {_x_s13[_k]:{fmt}},  f(x_{_k}) = {_y_s13[_k]:{fmt}},  coef = {_c}\n"
                        _terms.append(f"{_c}·{_y_s13[_k]:{fmt}}")
                    _suma_val = sum((1 if (k==0 or k==n_simp) else (4 if k%2!=0 else 2)) * _y_s13[k] for k in range(n_simp+1))
                    bloque = (
                        _lines + f"\nSuma = {' + '.join(_terms)}\n"
                        f"Suma = {_suma_val:{fmt}}\n\n"
                        f"I = (h/3) · Suma\n"
                        f"I = ({h_step:{fmt}}/3) · {_suma_val:{fmt}}\n"
                        f"I = {h_step/3:{fmt}} · {_suma_val:{fmt}}\n"
                        f"I = {integral:{fmt}}"
                    )
                    if h_step > 0:
                        _f4_max = (err_trunc * 180) / abs((b_simp - a_simp) * h_step**4)
                        bloque += (
                            f"\n\nError de Truncamiento:\n"
                            f"|E_T| = ((b - a) · h⁴ / 180) · max|f⁴(ξ)|\n"
                            f"|E_T| = (({b_simp:{fmt}} - {a_simp:{fmt}}) · {h_step:{fmt}}⁴ / 180) · {_f4_max:{fmt}}\n"
                            f"|E_T| = {err_trunc:{fmt}}"
                        )
                    st.code(bloque, language="text")
                # Gráfico con área sombreada
                x_plot = np.linspace(a_simp, b_simp, 300)
                y_plot = [evaluar_f(func_input, xi) for xi in x_plot]
                if None not in y_plot:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_plot, y=y_plot, name="f(x)",
                        line=dict(color='#00cfcc', width=2),
                        fill='tozeroy', fillcolor='rgba(0,207,204,0.15)'
                    ))
                    x_nodes = np.linspace(a_simp, b_simp, n_simp + 1)
                    y_nodes = [evaluar_f(func_input, xi) for xi in x_nodes]
                    fig.add_trace(go.Scatter(
                        x=x_nodes, y=y_nodes, mode='markers',
                        name="Nodos Simpson", marker=dict(size=8, color='#ff4b4b')
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        title=f"Simpson 1/3 — ∫f(x)dx ≈ {integral:{fmt}}",
                        xaxis_title="x", yaxis_title="f(x)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se pudo evaluar f(x) en el intervalo. Verificá la función.")

        elif metodo_sel == "Simpson 3/8":
            res_simp38 = metodo_simpson_38(func_input, a_simp38, b_simp38, n_simp38, a_simp38_str, b_simp38_str, xi_punto=xi_val_38)
            if res_simp38[0] is not None:
                integral, err_trunc, h_step, df_tabla, indets, _y_s38 = res_simp38
                if mostrar_formulas:
                    st.subheader("Fórmulas")
                    st.latex(
                        r"h = \frac{b - a}{n}"
                    )
                    st.latex(
                        r"I \approx \frac{3h}{8}\left[f(x_0) + 3f(x_1) + 3f(x_2) + 2f(x_3) + \cdots + f(x_n)\right]"
                    )
                    st.latex(
                        r"|E_T| \leq \frac{(b-a)\,h^4}{80}\,\max_{\xi \in [a,b]}\left|f^{(4)}(\xi)\right|"
                    )
                    with st.expander("📖 Notación"):
                        st.markdown("""
| Símbolo | Significado |
|---|---|
| $a,\\, b$ | Límites inferior y superior del intervalo de integración |
| $n$ | Número de subintervalos (debe ser múltiplo de 3) |
| $h$ | Ancho de cada subintervalo: $h = (b-a)/n$ |
| $x_0, x_1, \\ldots, x_n$ | Nodos equiespaciados: $x_k = a + k\\,h$ |
| $f(x_k)$ | Valor de la función en el nodo $x_k$ |
| $I$ | Valor aproximado de la integral $\\int_a^b f(x)\\,dx$ |
| $E_T$ | Error de truncamiento de la fórmula compuesta |
| $f^{(4)}(\\xi)$ | Cuarta derivada de $f$ en algún punto $\\xi \\in [a,b]$ |
""")
                st.subheader("Resultado")
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Integral ≈", f"{integral:{fmt}}")
                col_r2.metric("Paso h", f"{h_step:{fmt}}")
                col_r3.metric("Error Trunc. |Eₜ|", formatear_error(err_trunc))
                st.dataframe(format_df(df_tabla), use_container_width=True)
                if indets:
                    for xi, val in indets:
                        st.warning(f"⚠️ Indeterminación detectada en x = {xi:{fmt}}. Se resolvió usando límite/L'Hôpital obteniendo: f(x) ≈ {val:{fmt}}")
                # --- DESARROLLO PASO A PASO ---
                with st.expander("📋 Desarrollo paso a paso", expanded=False):
                    st.code(f"h = (b - a) / n = ({b_simp38:{fmt}} - {a_simp38:{fmt}}) / {n_simp38} = {h_step:{fmt}}", language="text")
                    _x_s38 = np.linspace(a_simp38, b_simp38, n_simp38 + 1)
                    _terms = []
                    _lines = "Nodos y evaluaciones:\n"
                    if indets:
                        _lines += "(Se aplicó cálculo de límites en los puntos de indeterminación)\n"
                    for _k in range(n_simp38 + 1):
                        if _k == 0 or _k == n_simp38:
                            _c = 1
                        elif _k % 3 == 0:
                            _c = 2
                        else:
                            _c = 3
                        _lines += f"  x_{_k} = {_x_s38[_k]:{fmt}},  f(x_{_k}) = {_y_s38[_k]:{fmt}},  coef = {_c}\n"
                        _terms.append(f"{_c}·{_y_s38[_k]:{fmt}}")
                    _suma_val = sum((1 if (k==0 or k==n_simp38) else (2 if k%3==0 else 3)) * _y_s38[k] for k in range(n_simp38+1))
                    bloque = (
                        _lines + f"\nSuma = {' + '.join(_terms)}\n"
                        f"Suma = {_suma_val:{fmt}}\n\n"
                        f"I = (3h/8) · Suma\n"
                        f"I = (3·{h_step:{fmt}}/8) · {_suma_val:{fmt}}\n"
                        f"I = {3*h_step/8:{fmt}} · {_suma_val:{fmt}}\n"
                        f"I = {integral:{fmt}}"
                    )
                    if h_step > 0:
                        _f4_max = (err_trunc * 80) / abs((b_simp38 - a_simp38) * h_step**4)
                        bloque += (
                            f"\n\nError de Truncamiento:\n"
                            f"|E_T| = ((b - a) · h⁴ / 80) · max|f⁴(ξ)|\n"
                            f"|E_T| = (({b_simp38:{fmt}} - {a_simp38:{fmt}}) · {h_step:{fmt}}⁴ / 80) · {_f4_max:{fmt}}\n"
                            f"|E_T| = {err_trunc:{fmt}}"
                        )
                    st.code(bloque, language="text")
                # Gráfico con área sombreada
                x_plot = np.linspace(a_simp38, b_simp38, 300)
                y_plot = [evaluar_f(func_input, xi) for xi in x_plot]
                if None not in y_plot:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_plot, y=y_plot, name="f(x)",
                        line=dict(color='#e03ce6', width=2),
                        fill='tozeroy', fillcolor='rgba(224,60,230,0.15)'
                    ))
                    x_nodes = np.linspace(a_simp38, b_simp38, n_simp38 + 1)
                    y_nodes = [evaluar_f(func_input, xi) for xi in x_nodes]
                    fig.add_trace(go.Scatter(
                        x=x_nodes, y=y_nodes, mode='markers',
                        name="Nodos Simpson 3/8", marker=dict(size=8, color='#00cfcc')
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        title=f"Simpson 3/8 — ∫f(x)dx ≈ {integral:{fmt}}",
                        xaxis_title="x", yaxis_title="f(x)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se pudo evaluar f(x) en el intervalo. Verificá la función.")

        elif metodo_sel == "Trapecios":
            res_trap = metodo_trapecios(func_input, a_trap, b_trap, n_trap, xi_punto=xi_val_trap)
            if res_trap[0] is not None:
                integral, err_trunc, h_step, df_tabla, indets, _y_tr = res_trap
                if mostrar_formulas:
                    st.subheader("Fórmulas")
                    st.latex(
                        r"h = \frac{b - a}{n}"
                    )
                    st.latex(
                        r"I \approx \frac{h}{2}\left[f(x_0) + 2f(x_1) + 2f(x_2) + \cdots + 2f(x_{n-1}) + f(x_n)\right]"
                    )
                    st.latex(
                        r"|E_T| \leq \frac{(b-a)\,h^2}{12}\,\max_{\xi \in [a,b]}\left|f''(\xi)\right|"
                    )
                    with st.expander("📖 Notación"):
                        st.markdown("""
| Símbolo | Significado |
|---|---|
| $a,\\, b$ | Límites inferior y superior del intervalo de integración |
| $n$ | Número de subintervalos |
| $h$ | Ancho de cada subintervalo: $h = (b-a)/n$ |
| $x_0, x_1, \\ldots, x_n$ | Nodos equiespaciados: $x_k = a + k\\,h$ |
| $f(x_k)$ | Valor de la función en el nodo $x_k$ |
| $I$ | Valor aproximado de la integral $\\int_a^b f(x)\\,dx$ |
| $E_T$ | Error de truncamiento de la fórmula compuesta |
| $f''(\\xi)$ | Segunda derivada de $f$ en algún punto $\\xi \\in [a,b]$ |
""")
                st.subheader("Resultado")
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Integral ≈", f"{integral:{fmt}}")
                col_r2.metric("Paso h", f"{h_step:{fmt}}")
                col_r3.metric("Error Trunc. |Eₜ|", formatear_error(err_trunc))
                st.dataframe(format_df(df_tabla), use_container_width=True)
                if indets:
                    for xi, val in indets:
                        st.warning(f"⚠️ Indeterminación detectada en x = {xi:{fmt}}. Se resolvió usando límite/L'Hôpital obteniendo: f(x) ≈ {val:{fmt}}")
                # --- DESARROLLO PASO A PASO ---
                with st.expander("📋 Desarrollo paso a paso", expanded=False):
                    st.code(f"h = (b - a) / n = ({b_trap:{fmt}} - {a_trap:{fmt}}) / {n_trap} = {h_step:{fmt}}", language="text")
                    _x_tr = np.linspace(a_trap, b_trap, n_trap + 1)
                    _terms = []
                    _lines = "Nodos y evaluaciones:\n"
                    if indets:
                        _lines += "(Se aplicó cálculo de límites en los puntos de indeterminación)\n"
                    for _k in range(n_trap + 1):
                        _c = 1 if (_k == 0 or _k == n_trap) else 2
                        _lines += f"  x_{_k} = {_x_tr[_k]:{fmt}},  f(x_{_k}) = {_y_tr[_k]:{fmt}},  coef = {_c}\n"
                        _terms.append(f"{_c}·{_y_tr[_k]:{fmt}}")
                    _suma_val = sum((1 if (k==0 or k==n_trap) else 2) * _y_tr[k] for k in range(n_trap+1))
                    bloque = (
                        _lines + f"\nSuma = {' + '.join(_terms)}\n"
                        f"Suma = {_suma_val:{fmt}}\n\n"
                        f"I = (h/2) · Suma\n"
                        f"I = ({h_step:{fmt}}/2) · {_suma_val:{fmt}}\n"
                        f"I = {h_step/2:{fmt}} · {_suma_val:{fmt}}\n"
                        f"I = {integral:{fmt}}"
                    )
                    if h_step > 0:
                        _f2_max = (err_trunc * 12) / abs((b_trap - a_trap) * h_step**2)
                        bloque += (
                            f"\n\nError de Truncamiento:\n"
                            f"|E_T| = ((b - a) · h² / 12) · max|f''(ξ)|\n"
                            f"|E_T| = (({b_trap:{fmt}} - {a_trap:{fmt}}) · {h_step:{fmt}}² / 12) · {_f2_max:{fmt}}\n"
                            f"|E_T| = {err_trunc:{fmt}}"
                        )
                    st.code(bloque, language="text")
                # Gráfico con área sombreada y trapecios
                x_plot = np.linspace(a_trap, b_trap, 300)
                y_plot = [evaluar_f(func_input, xi) for xi in x_plot]
                if None not in y_plot:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_plot, y=y_plot, name="f(x)",
                        line=dict(color='#f0a500', width=2),
                        fill='tozeroy', fillcolor='rgba(240,165,0,0.12)'
                    ))
                    x_nodes = np.linspace(a_trap, b_trap, n_trap + 1)
                    y_nodes = [evaluar_f(func_input, xi) for xi in x_nodes]
                    # Dibuja los trapecios como líneas verticales desde el eje
                    for xi, yi in zip(x_nodes, y_nodes):
                        fig.add_shape(type="line",
                            x0=xi, y0=0, x1=xi, y1=yi,
                            line=dict(color='rgba(240,165,0,0.5)', width=1, dash='dot')
                        )
                    fig.add_trace(go.Scatter(
                        x=x_nodes, y=y_nodes, mode='markers+lines',
                        name="Nodos Trapecios",
                        line=dict(color='#f0a500', width=1, dash='dash'),
                        marker=dict(size=8, color='white')
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        title=f"Trapecios — ∫f(x)dx ≈ {integral:{fmt}}",
                        xaxis_title="x", yaxis_title="f(x)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se pudo evaluar f(x) en el intervalo. Verificá la función.")

        elif metodo_sel == "Rectángulo Medio":
            integral, err_trunc, h_step, df_tabla = metodo_rectangulo_medio(func_input, a_rect, b_rect, n_rect)
            if integral is not None:
                if mostrar_formulas:
                    st.subheader("Fórmulas")
                    st.latex(
                        r"h = \frac{b - a}{n}"
                    )
                    st.latex(
                        r"x_{m_i} = \frac{x_i + x_{i+1}}{2}"
                    )
                    st.latex(
                        r"I \approx h \sum_{i=0}^{n-1} f(x_{m_i})"
                    )
                    st.latex(
                        r"|E_T| \leq \frac{(b-a)\,h^2}{24}\,\max_{\xi \in [a,b]}\left|f''(\xi)\right|"
                    )
                    with st.expander("📖 Notación"):
                        st.markdown("""
| Símbolo | Significado |
|---|---|
| $a,\\, b$ | Límites inferior y superior del intervalo de integración |
| $n$ | Número de subintervalos |
| $h$ | Ancho de cada subintervalo: $h = (b-a)/n$ |
| $x_{m_i}$ | Punto medio del subintervalo $i$ |
| $f(x_{m_i})$ | Valor de la función en el punto medio |
| $I$ | Valor aproximado de la integral $\\int_a^b f(x)\\,dx$ |
| $E_T$ | Error de truncamiento de la fórmula compuesta |
| $f''(\\xi)$ | Segunda derivada de $f$ en algún punto $\\xi \\in [a,b]$ |
""")
                st.subheader("Resultado")
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Integral ≈", f"{integral:{fmt}}")
                col_r2.metric("Paso h", f"{h_step:{fmt}}")
                col_r3.metric("Error Trunc. |Eₜ|", formatear_error(err_trunc))
                st.dataframe(format_df(df_tabla), use_container_width=True)
                # --- DESARROLLO PASO A PASO ---
                with st.expander("📋 Desarrollo paso a paso", expanded=False):
                    st.code(f"h = (b - a) / n = ({b_rect:{fmt}} - {a_rect:{fmt}}) / {n_rect} = {h_step:{fmt}}", language="text")
                    _x_rc = np.linspace(a_rect, b_rect, n_rect + 1)
                    _x_mid_rc = (_x_rc[:-1] + _x_rc[1:]) / 2
                    _y_mid_rc = [evaluar_f(func_input, xi) for xi in _x_mid_rc]
                    _terms = []
                    _lines = "Puntos medios y evaluaciones:\n"
                    for _k in range(n_rect):
                        _lines += f"  xₘ_{_k+1} = ({_x_rc[_k]:{fmt}} + {_x_rc[_k+1]:{fmt}})/2 = {_x_mid_rc[_k]:{fmt}},  f(xₘ) = {_y_mid_rc[_k]:{fmt}}\n"
                        _terms.append(f"{_y_mid_rc[_k]:{fmt}}")
                    _suma_val = sum(_y_mid_rc)
                    bloque = (
                        _lines + f"\nSuma = {' + '.join(_terms)}\n"
                        f"Suma = {_suma_val:{fmt}}\n\n"
                        f"I = h · Suma\n"
                        f"I = {h_step:{fmt}} · {_suma_val:{fmt}}\n"
                        f"I = {integral:{fmt}}"
                    )
                    if h_step > 0:
                        _f2_max = (err_trunc * 24) / abs((b_rect - a_rect) * h_step**2)
                        bloque += (
                            f"\n\nError de Truncamiento:\n"
                            f"|E_T| = ((b - a) · h² / 24) · max|f''(ξ)|\n"
                            f"|E_T| = (({b_rect:{fmt}} - {a_rect:{fmt}}) · {h_step:{fmt}}² / 24) · {_f2_max:{fmt}}\n"
                            f"|E_T| = {err_trunc:{fmt}}"
                        )
                    st.code(bloque, language="text")
                # Gráfico con área sombreada y rectángulos
                x_plot = np.linspace(a_rect, b_rect, 300)
                y_plot = [evaluar_f(func_input, xi) for xi in x_plot]
                if None not in y_plot:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_plot, y=y_plot, name="f(x)",
                        line=dict(color='#86e012', width=2),
                        fill='tozeroy', fillcolor='rgba(134,224,18,0.12)'
                    ))
                    x_nodes = np.linspace(a_rect, b_rect, n_rect + 1)
                    x_mid = (x_nodes[:-1] + x_nodes[1:]) / 2
                    y_mid = [evaluar_f(func_input, xi) for xi in x_mid]
                    # Dibujar los rectángulos
                    for x0, x1, ym in zip(x_nodes[:-1], x_nodes[1:], y_mid):
                        fig.add_shape(type="rect",
                            x0=x0, y0=0, x1=x1, y1=ym,
                            line=dict(color='rgba(134,224,18,0.8)', width=1, dash='solid'),
                            fillcolor='rgba(134,224,18,0.3)'
                        )
                    fig.add_trace(go.Scatter(
                        x=x_mid, y=y_mid, mode='markers',
                        name="Puntos Medios", marker=dict(size=8, color='white')
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        title=f"Rectángulo Medio — ∫f(x)dx ≈ {integral:{fmt}}",
                        xaxis_title="x", yaxis_title="f(x)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se pudo evaluar f(x) en el intervalo. Verificá la función.")

        elif metodo_sel == "Montecarlo":
            integral, err_est, s_dev, vol, ic_low, ic_up, df_tabla, x_r, y_r, invalid_count = metodo_montecarlo(
                func_input, a_mc, b_mc, n_mc, conf_mc, seed=seed_mc, antithetic=antithetic_mc)
            
            if invalid_count > 0:
                st.warning(f"⚠️ Se omitieron {invalid_count} muestras que resultaron en valores indefinidos en la función.")
            
            # --- CÁLCULO DE ERROR VERDADERO CON SYMPY / SCIPY ---
            integral_sym_str = None
            try:
                x_sym = sp.symbols('x')
                f_p = func_input.replace("^", "**").replace("sen", "sin").replace("ln", "log")
                f_p = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', f_p)
                f_expr = sp.sympify(f_p)
                integral_sym = sp.integrate(f_expr, x_sym)
                integral_def_sym = sp.integrate(f_expr, (x_sym, a_mc, b_mc))
                exact_val = float(integral_def_sym.evalf())
                true_error_perc = abs(integral - exact_val) / abs(exact_val) * 100 if exact_val != 0 else 0.0
                integral_sym_str = str(integral_sym)
            except:
                def func_scipy(x):
                    val = evaluar_f(func_input, x)
                    return val if val is not None else 0.0
                try:
                    exact_val, exact_err = spi.quad(func_scipy, a_mc, b_mc)
                    true_error_perc = abs(integral - exact_val) / abs(exact_val) * 100 if exact_val != 0 else 0.0
                except:
                    exact_val, true_error_perc = None, None

            if mostrar_formulas:
                st.subheader("Fórmulas")
                st.latex(r"I \approx \text{Area} \cdot \frac{1}{N}\sum_{i=1}^{N} f(x_i)")
                st.latex(r"\text{Var} = \frac{1}{N-1}\sum_{i=1}^{N} \left(f(x_i) - \bar{f}\right)^2")
                st.latex(r"EE = \text{Area} \cdot \frac{S_{\text{desv}}}{\sqrt{N}}")
                st.latex(r"IC = I \pm z_{\alpha/2} EE")
                with st.expander("📖 Notación"):
                    st.markdown("""
| Símbolo | Significado |
|---|---|
| $I$ | Valor estimado de la integral de Montecarlo |
| Area | Volumen/Área del dominio de integración: $b - a$ |
| $N$ | Cantidad de puntos aleatorios generados |
| $f(x_i)$ | Valor de la función en el punto muestreado $x_i$ |
| Var | Varianza muestral de las evaluaciones de $f$ |
| $S_{\\text{desv}}$ | Desviación estándar muestral ($\\sqrt{\\text{Var}}$) |
| $EE$ | Error Estándar de la media ponderado por el área |
| $IC$ | Intervalo de confianza estadístico |
""")
            
            st.subheader("Resultado")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Integral ≈", f"{integral:{fmt}}")
            if exact_val is not None:
                lbl = "Valor Exacto (Analítico)" if integral_sym_str else "Valor Exacto (Scipy)"
                c2.metric(lbl, f"{exact_val:{fmt}}", f"Err: {true_error_perc:{fmt}}%", delta_color="inverse")
            else:
                c2.metric("Valor Exacto", "N/A")
            c3.metric("Desv. Estándar (S_desv)", f"{s_dev:{fmt}}")
            c4.metric("Error Estándar (EE)", formatear_error(err_est))
            c5.metric(f"IC {conf_mc}%", f"[{ic_low:{fmt}}, {ic_up:{fmt}}]")
            st.write(f"**Área/Volumen Región:** {vol:{fmt}}")
            st.dataframe(format_df(df_tabla), use_container_width=True)
            # --- DESARROLLO PASO A PASO ---
            with st.expander("📋 Desarrollo paso a paso", expanded=False):
                _f_mean = np.mean(y_r[~np.isnan(y_r)])
                _n_valid_mc = int(np.sum(~np.isnan(y_r)))
                bloque = (
                    f"Area = b - a = {b_mc:{fmt}} - {a_mc:{fmt}} = {vol:{fmt}}\n"
                    f"N = {n_mc} puntos aleatorios en [{a_mc:{fmt}}, {b_mc:{fmt}}]\n\n"
                    f"f̄ = (1/N) · Σf(xᵢ) = {_f_mean:{fmt}}\n\n"
                    f"I = Area · f̄\n"
                    f"I = {vol:{fmt}} · {_f_mean:{fmt}}\n"
                    f"I = {integral:{fmt}}\n\n"
                    f"S_desv² = (1/(N-1)) · Σ(f(xᵢ) - f̄)² = {s_dev**2:{fmt}}\n"
                    f"S_desv = √(S_desv²) = {s_dev:{fmt}}\n\n"
                    f"EE = Area · (S_desv / √N) = {vol:{fmt}} · ({s_dev:{fmt}} / √{n_mc})\n"
                    f"EE = {err_est:{fmt}}\n\n"
                    f"IC = I ± z·EE = {integral:{fmt}} ± {abs(integral - ic_low):{fmt}}\n"
                    f"IC = [{ic_low:{fmt}}, {ic_up:{fmt}}]"
                )
                if integral_sym_str is not None:
                    bloque += (
                        f"\n\nIntegración Analítica (Exacta):\n"
                        f"∫ f(x) dx = {integral_sym_str}\n"
                        f"Evaluando en límites [{a_mc:{fmt}}, {b_mc:{fmt}}]: {exact_val:{fmt}}\n"
                        f"Error Verdadero = |{integral:{fmt}} - {exact_val:{fmt}}| = {abs(integral - exact_val):{fmt}}"
                    )
                elif exact_val is not None:
                    bloque += f"\n\nError Verdadero (Numérico Scipy) = |{integral:{fmt}} - {exact_val:{fmt}}| = {abs(integral - exact_val):{fmt}}"
                st.code(bloque, language="text")
            
            # --- TABS PARA GRÁFICOS AVANZADOS ---
            tab1, tab2, tab3 = st.tabs(["Muestreo y Función", "Convergencia y Confianza", "Distribución de f(x)"])
            
            with tab1:
                fig = go.Figure()
                x_plot = np.linspace(a_mc, b_mc, 300)
                y_plot = evaluar_f_array(func_input, x_plot)
                fig.add_trace(go.Scatter(x=x_plot, y=y_plot, name="f(x)", line=dict(color='#86e012')))
                
                n_plot = min(n_mc, 1000)
                fig.add_trace(go.Scatter(x=x_r[:n_plot], y=y_r[:n_plot], mode='markers', name="Muestras", marker=dict(size=4, color='rgba(255,255,255,0.4)')))
                fig.update_layout(template="plotly_dark", title=f"Montecarlo — Función Integrada")
                st.plotly_chart(fig, use_container_width=True)
                
            with tab2:
                # Calculo de Convergencia Vectorizado ignorando NaNs
                valid_mask = ~np.isnan(y_r)
                y_valid = y_r[valid_mask]
                n_valid = len(y_valid)
                
                if n_valid > 0:
                    cum_mean = np.cumsum(y_valid) / np.arange(1, n_valid + 1)
                    cum_integral = vol * cum_mean
                    
                    if n_valid > 1:
                        cum_var = pd.Series(y_valid).expanding().var(ddof=1).fillna(0).values
                        cum_std_error = vol * np.sqrt(cum_var / np.arange(1, n_valid + 1))
                    else:
                        cum_std_error = np.zeros(n_valid)
                else:
                    cum_integral, cum_std_error = np.zeros(0), np.zeros(0)
                
                alpha = 1 - (conf_mc / 100.0)
                z_val = st_stats.norm.ppf(1 - alpha/2)
                margin = z_val * cum_std_error
                
                fig_conv = go.Figure()
                
                if n_valid > 5000:
                    idx = np.unique(np.geomspace(1, n_valid, 1000).astype(int)) - 1
                else:
                    idx = np.arange(n_valid)
                
                n_idx = idx + 1
                
                if exact_val is not None:
                    fig_conv.add_trace(go.Scatter(x=n_idx, y=np.full_like(n_idx, exact_val, dtype=float), mode='lines', name='Valor Exacto', line=dict(color='white', dash='dash')))

                fig_conv.add_trace(go.Scatter(
                    x=n_idx, y=cum_integral[idx] + margin[idx],
                    mode='lines', line=dict(color='rgba(255, 75, 75, 0)'),
                    name='Límite Superior', showlegend=False
                ))
                fig_conv.add_trace(go.Scatter(
                    x=n_idx, y=cum_integral[idx] - margin[idx],
                    mode='lines', fill='tonexty', fillcolor='rgba(255, 75, 75, 0.2)',
                    line=dict(color='rgba(255, 75, 75, 0)'),
                    name=f'Intervalo Confianza {conf_mc}%'
                ))
                fig_conv.add_trace(go.Scatter(x=n_idx, y=cum_integral[idx], mode='lines', name='Valor Estimado', line=dict(color='#00cfcc', width=2)))
                
                fig_conv.update_layout(template="plotly_dark", title="Convergencia de la Aproximación (N progresivo)", xaxis_title="Número de Muestras (N)", yaxis_title="Integral Estimada")
                st.plotly_chart(fig_conv, use_container_width=True)

            with tab3:
                fig_hist = go.Figure(data=[go.Histogram(x=y_r, nbinsx=50, marker_color='#e03ce6', opacity=0.75)])
                fig_hist.update_layout(template="plotly_dark", title="Distribución de Evaluaciones Aleatorias f(x)", xaxis_title="f(x)", yaxis_title="Frecuencia")
                st.plotly_chart(fig_hist, use_container_width=True)

        elif metodo_sel == "Montecarlo Doble":
            integral, err_est, s_dev, area_xy, ic_low, ic_up, df_tabla, x_r, y_r, z_r, invalid_count = metodo_montecarlo_doble(
                func_input, a_x_mc, b_x_mc, a_y_mc, b_y_mc, n_mc2, conf_mc2, seed=seed_mc2, antithetic=antithetic_mc2)
            
            if invalid_count > 0:
                st.warning(f"⚠️ Se omitieron {invalid_count} muestras que resultaron en valores indefinidos en la función.")
            
            # --- CÁLCULO DE ERROR VERDADERO DOBLE CON SYMPY / SCIPY ---
            integral_sym_str = None
            try:
                x_sym, y_sym = sp.symbols('x y')
                f_p = func_input.replace("^", "**").replace("sen", "sin").replace("ln", "log")
                f_p = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', f_p)
                f_expr = sp.sympify(f_p)
                integral_sym = sp.integrate(f_expr, y_sym, x_sym)
                integral_def_sym = sp.integrate(f_expr, (y_sym, a_y_mc, b_y_mc), (x_sym, a_x_mc, b_x_mc))
                exact_val = float(integral_def_sym.evalf())
                true_error_perc = abs(integral - exact_val) / abs(exact_val) * 100 if exact_val != 0 else 0.0
                integral_sym_str = str(integral_sym)
            except:
                def func_scipy_dbl(y, x):
                    return float(evaluar_f_array(func_input, np.array([x]), np.array([y]))[0])
                try:
                    exact_val, exact_err = spi.dblquad(func_scipy_dbl, a_x_mc, b_x_mc, lambda x: a_y_mc, lambda x: b_y_mc)
                    true_error_perc = abs(integral - exact_val) / abs(exact_val) * 100 if exact_val != 0 else 0.0
                except:
                    exact_val, true_error_perc = None, None
                
            if mostrar_formulas:
                st.subheader("Fórmulas")
                st.latex(r"I \approx \text{Area} \cdot \frac{1}{N}\sum_{i=1}^{N} f(x_i, y_i)")
                st.latex(r"\text{Var} = \frac{1}{N-1}\sum_{i=1}^{N} \left(f(x_i, y_i) - \bar{f}\right)^2")
                st.latex(r"EE = \text{Area} \cdot \frac{S_{\text{desv}}}{\sqrt{N}}")
                st.latex(r"IC = I \pm z_{\alpha/2} EE")
                with st.expander("📖 Notación"):
                    st.markdown("""
| Símbolo | Significado |
|---|---|
| $I$ | Valor estimado de la integral doble de Montecarlo |
| Area | Área del dominio de integración: $(b_x - a_x) \\cdot (b_y - a_y)$ |
| $N$ | Cantidad de puntos aleatorios generados |
| $f(x_i, y_i)$ | Valor de la función en el punto muestreado $(x_i, y_i)$ |
| Var | Varianza muestral de las evaluaciones de $f$ |
| $S_{\\text{desv}}$ | Desviación estándar muestral ($\\sqrt{\\text{Var}}$) |
| $EE$ | Error Estándar de la media ponderado por el área |
| $IC$ | Intervalo de confianza estadístico |
""")
            
            st.subheader("Resultado")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Integral Acumulada ≈", f"{integral:{fmt}}")
            if exact_val is not None:
                lbl = "Valor Exacto (Analítico)" if integral_sym_str else "Valor Exacto (Scipy)"
                c2.metric(lbl, f"{exact_val:{fmt}}", f"Err: {true_error_perc:{fmt}}%", delta_color="inverse")
            else:
                c2.metric("Valor Exacto", "N/A")
            c3.metric("Desv. Estándar (S_desv)", f"{s_dev:{fmt}}")
            c4.metric("Error Estándar (EE)", formatear_error(err_est))
            c5.metric(f"IC {conf_mc2}%", f"[{ic_low:{fmt}}, {ic_up:{fmt}}]")
            st.write(f"**Área Integración:** {area_xy:{fmt}}")
            st.dataframe(format_df(df_tabla), use_container_width=True)
            # --- DESARROLLO PASO A PASO ---
            with st.expander("📋 Desarrollo paso a paso", expanded=False):
                _f_mean_d = np.mean(z_r[~np.isnan(z_r)])
                bloque = (
                    f"Area = (bx-ax)·(by-ay) = ({b_x_mc:{fmt}}-{a_x_mc:{fmt}})·({b_y_mc:{fmt}}-{a_y_mc:{fmt}}) = {area_xy:{fmt}}\n"
                    f"N = {n_mc2} puntos aleatorios\n\n"
                    f"f̄ = (1/N) · Σf(xᵢ,yᵢ) = {_f_mean_d:{fmt}}\n\n"
                    f"I = Area · f̄\n"
                    f"I = {area_xy:{fmt}} · {_f_mean_d:{fmt}}\n"
                    f"I = {integral:{fmt}}\n\n"
                    f"S_desv² = {s_dev**2:{fmt}},  S_desv = {s_dev:{fmt}}\n"
                    f"EE = Area · (S_desv / √N) = {err_est:{fmt}}\n\n"
                    f"IC = [{ic_low:{fmt}}, {ic_up:{fmt}}]"
                )
                if integral_sym_str is not None:
                    bloque += (
                        f"\n\nIntegración Analítica (Exacta):\n"
                        f"∬ f(x,y) dy dx = {integral_sym_str}\n"
                        f"Evaluando en límites x:[{a_x_mc:{fmt}}, {b_x_mc:{fmt}}], y:[{a_y_mc:{fmt}}, {b_y_mc:{fmt}}]: {exact_val:{fmt}}\n"
                        f"Error Verdadero = |{integral:{fmt}} - {exact_val:{fmt}}| = {abs(integral - exact_val):{fmt}}"
                    )
                elif exact_val is not None:
                    bloque += f"\n\nError Verdadero (Numérico Scipy) = |{integral:{fmt}} - {exact_val:{fmt}}| = {abs(integral - exact_val):{fmt}}"
                st.code(bloque, language="text")
            
            tab1, tab2, tab3 = st.tabs(["Muestreo 3D", "Convergencia y Confianza", "Distribución de f(x,y)"])
            
            with tab1:
                fig = go.Figure()
                x_m = np.linspace(a_x_mc, b_x_mc, 40)
                y_m = np.linspace(a_y_mc, b_y_mc, 40)
                X, Y = np.meshgrid(x_m, y_m)
                Z = evaluar_f_array(func_input, X.ravel(), Y.ravel()).reshape(X.shape)
                fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.8, name="f(x,y)"))
                
                n_plot = min(n_mc2, 500)
                fig.add_trace(go.Scatter3d(x=x_r[:n_plot], y=y_r[:n_plot], z=z_r[:n_plot], mode='markers', name="Muestras", marker=dict(size=3, color='red')))
                fig.update_layout(template="plotly_dark", title=f"Montecarlo Doble — Visualización 3D")
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                valid_mask = ~np.isnan(z_r)
                z_valid = z_r[valid_mask]
                n_valid = len(z_valid)
                
                if n_valid > 0:
                    cum_mean = np.cumsum(z_valid) / np.arange(1, n_valid + 1)
                    cum_integral = area_xy * cum_mean
                    
                    if n_valid > 1:
                        cum_var = pd.Series(z_valid).expanding().var(ddof=1).fillna(0).values
                        cum_std_error = area_xy * np.sqrt(cum_var / np.arange(1, n_valid + 1))
                    else:
                        cum_std_error = np.zeros(n_valid)
                else:
                    cum_integral, cum_std_error = np.zeros(0), np.zeros(0)
                
                alpha = 1 - (conf_mc2 / 100.0)
                z_val = st_stats.norm.ppf(1 - alpha/2)
                margin = z_val * cum_std_error
                
                fig_conv = go.Figure()
                if n_valid > 5000:
                    idx = np.unique(np.geomspace(1, n_valid, 1000).astype(int)) - 1
                else:
                    idx = np.arange(n_valid)
                n_idx = idx + 1
                
                if exact_val is not None:
                    fig_conv.add_trace(go.Scatter(x=n_idx, y=np.full_like(n_idx, exact_val, dtype=float), mode='lines', name='Valor Exacto', line=dict(color='white', dash='dash')))

                fig_conv.add_trace(go.Scatter(
                    x=n_idx, y=cum_integral[idx] + margin[idx],
                    mode='lines', line=dict(color='rgba(255, 75, 75, 0)'),
                    name='Límite Superior', showlegend=False
                ))
                fig_conv.add_trace(go.Scatter(
                    x=n_idx, y=cum_integral[idx] - margin[idx],
                    mode='lines', fill='tonexty', fillcolor='rgba(255, 75, 75, 0.2)',
                    line=dict(color='rgba(255, 75, 75, 0)'),
                    name=f'Intervalo Confianza {conf_mc2}%'
                ))
                fig_conv.add_trace(go.Scatter(x=n_idx, y=cum_integral[idx], mode='lines', name='Valor Estimado', line=dict(color='#00cfcc', width=2)))
                
                fig_conv.update_layout(template="plotly_dark", title="Convergencia Iterativa de Montecarlo Doble", xaxis_title="Número de Muestras (N)", yaxis_title="Integral Estimada")
                st.plotly_chart(fig_conv, use_container_width=True)

            with tab3:
                fig_hist = go.Figure(data=[go.Histogram(x=z_r, nbinsx=50, marker_color='#00cfcc', opacity=0.75)])
                fig_hist.update_layout(template="plotly_dark", title="Distribución de Evaluaciones Aleatorias f(x,y)", xaxis_title="f(x,y)", yaxis_title="Frecuencia")
                st.plotly_chart(fig_hist, use_container_width=True)

        elif metodo_sel == "Runge-Kutta":
            if rk_tipo == "EDO Simple":
                # --- FÓRMULAS ---
                if mostrar_formulas:
                    st.subheader("Fórmulas")
                    if "Euler" in rk_orden:
                        st.latex(r"y_{i+1} = y_i + h \cdot f(x_i, y_i)")
                        st.latex(r"\text{Error local: } O(h^2) \quad | \quad \text{Error global: } O(h)")
                    elif "RK2" in rk_orden:
                        st.latex(r"y_{i+1} = y_i + h \left( b_1 k_1 + b_2 k_2 \right)")
                        st.latex(r"k_1 = f(x_i,\; y_i)")
                        st.latex(r"k_2 = f(x_i + p_1 h,\; y_i + p_1 h \, k_1)")
                        if rk_variante == "Heun":
                            st.info("**Heun:** b₁=½, b₂=½, p₁=1")
                        elif rk_variante == "Punto Medio":
                            st.info("**Punto Medio:** b₁=0, b₂=1, p₁=½")
                        else:
                            st.info("**Ralston:** b₁=¼, b₂=¾, p₁=⅔")
                        st.latex(r"\text{Error local: } O(h^3) \quad | \quad \text{Error global: } O(h^2)")
                    else:  # RK4
                        st.latex(r"y_{i+1} = y_i + h \cdot \varphi(x_i, y_i, h)")
                        st.latex(r"\varphi = \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)")
                        st.latex(r"k_1 = f(x_i,\; y_i)")
                        st.latex(r"k_2 = f\!\left(x_i + \tfrac{h}{2},\; y_i + \tfrac{h}{2}k_1\right)")
                        st.latex(r"k_3 = f\!\left(x_i + \tfrac{h}{2},\; y_i + \tfrac{h}{2}k_2\right)")
                        st.latex(r"k_4 = f(x_i + h,\; y_i + h \, k_3)")
                        st.latex(r"\text{Error local: } O(h^5) \quad | \quad \text{Error global: } O(h^4)")
                    with st.expander("📖 Notación"):
                        st.markdown("""
| Símbolo | Significado |
|---|---|
| $y_i$ | Aproximación de $y$ en el paso $i$ |
| $x_i$ | Variable independiente en el paso $i$: $x_i = x_0 + i \\cdot h$ |
| $h$ | Tamaño del paso |
| $k_j$ | Pendiente evaluada en el punto $j$-ésimo del paso |
| $\\varphi$ | Pendiente promedio ponderada (RK4) |
| $f(x,y)$ | Función que define la EDO: $dy/dx = f(x,y)$ |
""")

                # --- CÁLCULO ---
                st.subheader("Resultado")
                if "Euler" in rk_orden:
                    df_rk, xs_rk, ys_rk = metodo_euler(func_input, rk_x0, rk_y0, rk_h, rk_n)
                elif "RK2" in rk_orden:
                    df_rk, xs_rk, ys_rk = metodo_rk2(func_input, rk_x0, rk_y0, rk_h, rk_n, rk_variante)
                else:
                    df_rk, xs_rk, ys_rk = metodo_rk4(func_input, rk_x0, rk_y0, rk_h, rk_n)

                # Solución exacta
                xs_arr = np.array(xs_rk)
                y_exacta = None
                rk_exacta_usada = rk_exacta
                if not rk_exacta or not rk_exacta.strip():
                    try:
                        x_sym = sp.symbols('x')
                        y_sym = sp.Function('y')
                        f_p = func_input.replace("^", "**").replace("sen", "sin").replace("ln", "log")
                        f_p = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', f_p)
                        f_expr = sp.sympify(f_p, locals={"y": y_sym(x_sym)})
                        eq = sp.Eq(y_sym(x_sym).diff(x_sym), f_expr)
                        x0_exact = sp.nsimplify(sp.sympify(rk_x0_str))
                        y0_exact = sp.nsimplify(sp.sympify(rk_y0_str))
                        sol = sp.dsolve(eq, y_sym(x_sym), ics={y_sym(x0_exact): y0_exact})
                        if isinstance(sol, list): sol = sol[0]
                        rk_exacta_usada = str(sol.rhs)
                        st.success(f"Y(x) Analítica (Sympy): {rk_exacta_usada}")
                    except:
                        pass

                if rk_exacta_usada and rk_exacta_usada.strip():
                    try:
                        y_exacta = np.array([evaluar_f(rk_exacta_usada, xi) for xi in xs_arr])
                        if any(v is None for v in y_exacta):
                            y_exacta = None
                    except:
                        y_exacta = None
                if y_exacta is None:
                    y_exacta = obtener_solucion_exacta_edo(func_input, rk_x0, rk_y0, xs_arr)

                # Agregar error a la tabla si hay solución exacta
                if y_exacta is not None and len(y_exacta) == len(ys_rk):
                    df_rk["y_exacto"] = y_exacta
                    df_rk["Error Abs."] = np.abs(np.array(ys_rk) - y_exacta)

                # Métricas principales
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("y final ≈", f"{ys_rk[-1]:{fmt}}")
                col_r2.metric("x final", f"{xs_rk[-1]:{fmt}}")
                if y_exacta is not None:
                    err_final = abs(ys_rk[-1] - y_exacta[-1])
                    col_r3.metric("Error Final |y - y_exact|", formatear_error(err_final))

                st.dataframe(format_df(df_rk), use_container_width=True)

                # --- DESARROLLO PASO A PASO (REEMPLAZO TEXTUAL) ---
                with st.expander("📋 Desarrollo paso a paso", expanded=False):
                    # Recalcular paso a paso con formato textual
                    _x, _y = rk_x0, rk_y0
                    for _i in range(rk_n):
                        st.markdown(f"### Paso {_i} → {_i+1}")
                        st.markdown(f"**Datos:** $x_{{{_i}}} = {_x:{fmt}}$, $y_{{{_i}}} = {_y:{fmt}}$, $h = {rk_h:{fmt}}$")

                        if "Euler" in rk_orden:
                            _fval = evaluar_edo(func_input, _x, _y)
                            if _fval is None:
                                st.error(f"No se pudo evaluar f({_x:{fmt}}, {_y:{fmt}})")
                                break
                            _y_next = _y + rk_h * _fval
                            bloque = (
                                f"f(x_{_i}, y_{_i}) = f({_x:{fmt}}, {_y:{fmt}}) = {_fval:{fmt}}\n\n"
                                f"y_{{{_i+1}}} = y_{_i} + h · f(x_{_i}, y_{_i})\n"
                                f"y_{{{_i+1}}} = {_y:{fmt}} + {rk_h:{fmt}} · {_fval:{fmt}}\n"
                                f"y_{{{_i+1}}} = {_y_next:{fmt}}"
                            )
                            if y_exacta is not None and _i+1 < len(y_exacta):
                                _err_local = abs(_y_next - y_exacta[_i+1])
                                bloque += f"\n\nError vs Analítica = |{_y_next:{fmt}} - {y_exacta[_i+1]:{fmt}}| = {_err_local:{fmt}}"
                            st.code(bloque, language="text")
                            _y = _y_next
                            _x = _x + rk_h

                        elif "RK2" in rk_orden:
                            if rk_variante == "Heun":
                                _a2, _b1, _b2, _p1 = 1.0, 0.5, 0.5, 1.0
                            elif rk_variante == "Punto Medio":
                                _a2, _b1, _b2, _p1 = 0.5, 0.0, 1.0, 0.5
                            else:
                                _a2, _b1, _b2, _p1 = 2/3, 0.25, 0.75, 2/3
                            _k1 = evaluar_edo(func_input, _x, _y)
                            if _k1 is None: break
                            _k2 = evaluar_edo(func_input, _x + _p1 * rk_h, _y + _p1 * rk_h * _k1)
                            if _k2 is None: break
                            _y_next = _y + rk_h * (_b1 * _k1 + _b2 * _k2)

                            _x_k2 = _x + _p1 * rk_h
                            _y_k2 = _y + _p1 * rk_h * _k1
                            bloque = (
                                f"k₁ = f(x_{_i}, y_{_i}) = f({_x:{fmt}}, {_y:{fmt}}) = {_k1:{fmt}}\n\n"
                                f"k₂ = f(x_{_i} + p₁·h, y_{_i} + p₁·h·k₁)\n"
                                f"k₂ = f({_x:{fmt}} + {_p1:{fmt}}·{rk_h:{fmt}}, {_y:{fmt}} + {_p1:{fmt}}·{rk_h:{fmt}}·{_k1:{fmt}})\n"
                                f"k₂ = f({_x_k2:{fmt}}, {_y_k2:{fmt}}) = {_k2:{fmt}}\n\n"
                                f"y_{{{_i+1}}} = y_{_i} + h·(b₁·k₁ + b₂·k₂)\n"
                                f"y_{{{_i+1}}} = {_y:{fmt}} + {rk_h:{fmt}}·({_b1:{fmt}}·{_k1:{fmt}} + {_b2:{fmt}}·{_k2:{fmt}})\n"
                                f"y_{{{_i+1}}} = {_y:{fmt}} + {rk_h:{fmt}}·({_b1*_k1:{fmt}} + {_b2*_k2:{fmt}})\n"
                                f"y_{{{_i+1}}} = {_y:{fmt}} + {rk_h * (_b1*_k1 + _b2*_k2):{fmt}}\n"
                                f"y_{{{_i+1}}} = {_y_next:{fmt}}"
                            )
                            if y_exacta is not None and _i+1 < len(y_exacta):
                                _err_local = abs(_y_next - y_exacta[_i+1])
                                bloque += f"\n\nError vs Analítica = |{_y_next:{fmt}} - {y_exacta[_i+1]:{fmt}}| = {_err_local:{fmt}}"
                            st.code(bloque, language="text")
                            _y = _y_next
                            _x = _x + rk_h

                        else:  # RK4
                            _k1 = evaluar_edo(func_input, _x, _y)
                            if _k1 is None: break
                            _x_k2 = _x + rk_h/2
                            _y_k2 = _y + rk_h/2 * _k1
                            _k2 = evaluar_edo(func_input, _x_k2, _y_k2)
                            if _k2 is None: break
                            _x_k3 = _x + rk_h/2
                            _y_k3 = _y + rk_h/2 * _k2
                            _k3 = evaluar_edo(func_input, _x_k3, _y_k3)
                            if _k3 is None: break
                            _x_k4 = _x + rk_h
                            _y_k4 = _y + rk_h * _k3
                            _k4 = evaluar_edo(func_input, _x_k4, _y_k4)
                            if _k4 is None: break
                            _phi = (_k1 + 2*_k2 + 2*_k3 + _k4) / 6
                            _y_next = _y + rk_h * _phi

                            bloque = (
                                f"k₁ = f(x_{_i}, y_{_i}) = f({_x:{fmt}}, {_y:{fmt}}) = {_k1:{fmt}}\n\n"
                                f"k₂ = f(x_{_i} + h/2, y_{_i} + h/2·k₁)\n"
                                f"k₂ = f({_x:{fmt}} + {rk_h:{fmt}}/2, {_y:{fmt}} + {rk_h:{fmt}}/2·{_k1:{fmt}})\n"
                                f"k₂ = f({_x_k2:{fmt}}, {_y_k2:{fmt}}) = {_k2:{fmt}}\n\n"
                                f"k₃ = f(x_{_i} + h/2, y_{_i} + h/2·k₂)\n"
                                f"k₃ = f({_x:{fmt}} + {rk_h:{fmt}}/2, {_y:{fmt}} + {rk_h:{fmt}}/2·{_k2:{fmt}})\n"
                                f"k₃ = f({_x_k3:{fmt}}, {_y_k3:{fmt}}) = {_k3:{fmt}}\n\n"
                                f"k₄ = f(x_{_i} + h, y_{_i} + h·k₃)\n"
                                f"k₄ = f({_x:{fmt}} + {rk_h:{fmt}}, {_y:{fmt}} + {rk_h:{fmt}}·{_k3:{fmt}})\n"
                                f"k₄ = f({_x_k4:{fmt}}, {_y_k4:{fmt}}) = {_k4:{fmt}}\n\n"
                                f"φ = (1/6)·(k₁ + 2·k₂ + 2·k₃ + k₄)\n"
                                f"φ = (1/6)·({_k1:{fmt}} + 2·{_k2:{fmt}} + 2·{_k3:{fmt}} + {_k4:{fmt}})\n"
                                f"φ = (1/6)·({_k1 + 2*_k2 + 2*_k3 + _k4:{fmt}})\n"
                                f"φ = {_phi:{fmt}}\n\n"
                                f"y_{{{_i+1}}} = y_{_i} + h·φ\n"
                                f"y_{{{_i+1}}} = {_y:{fmt}} + {rk_h:{fmt}}·{_phi:{fmt}}\n"
                                f"y_{{{_i+1}}} = {_y_next:{fmt}}"
                            )
                            if y_exacta is not None and _i+1 < len(y_exacta):
                                _err_local = abs(_y_next - y_exacta[_i+1])
                                bloque += f"\n\nError vs Analítica = |{_y_next:{fmt}} - {y_exacta[_i+1]:{fmt}}| = {_err_local:{fmt}}"
                            st.code(bloque, language="text")
                            _y = _y_next
                            _x = _x + rk_h

                        st.markdown("---")

                # --- GRÁFICOS ---
                tab1, tab2, tab3 = st.tabs(["Solución y(x)", "Campo de Pendientes", "Análisis de Error"])

                with tab1:
                    fig = go.Figure()
                    # Curva numérica
                    fig.add_trace(go.Scatter(x=xs_rk, y=ys_rk, mode='lines+markers',
                        name=f"RK Numérico", line=dict(color='#00cfcc', width=2),
                        marker=dict(size=6)))
                    # Curva exacta (densa)
                    if y_exacta is not None:
                        x_dense = np.linspace(rk_x0, xs_rk[-1], 300)
                        if rk_exacta_usada and rk_exacta_usada.strip():
                            y_dense = [evaluar_f(rk_exacta_usada, xi) for xi in x_dense]
                        else:
                            y_dense = obtener_solucion_exacta_edo(func_input, rk_x0, rk_y0, x_dense)
                        if y_dense is not None:
                            fig.add_trace(go.Scatter(x=x_dense, y=y_dense, mode='lines',
                                name="Solución Exacta", line=dict(color='#ff4b4b', dash='dash', width=2)))
                    fig.update_layout(template="plotly_dark",
                        title=f"Solución de dy/dx = {func_input}",
                        xaxis_title="x", yaxis_title="y(x)")
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    # Campo de pendientes (slope field)
                    x_min_sf = rk_x0 - 0.5
                    x_max_sf = xs_rk[-1] + 0.5
                    y_min_sf = min(ys_rk) - abs(max(ys_rk) - min(ys_rk)) * 0.3 - 0.5
                    y_max_sf = max(ys_rk) + abs(max(ys_rk) - min(ys_rk)) * 0.3 + 0.5
                    nx_sf, ny_sf = 20, 20
                    x_sf = np.linspace(x_min_sf, x_max_sf, nx_sf)
                    y_sf = np.linspace(y_min_sf, y_max_sf, ny_sf)
                    fig_sf = go.Figure()
                    scale = (x_max_sf - x_min_sf) / nx_sf * 0.35
                    for xi in x_sf:
                        for yi in y_sf:
                            slope = evaluar_edo(func_input, xi, yi)
                            if slope is not None:
                                angle = np.arctan(slope)
                                dx = scale * np.cos(angle)
                                dy = scale * np.sin(angle)
                                fig_sf.add_shape(type="line",
                                    x0=xi-dx/2, y0=yi-dy/2, x1=xi+dx/2, y1=yi+dy/2,
                                    line=dict(color='rgba(255,255,255,0.3)', width=1))
                    fig_sf.add_trace(go.Scatter(x=xs_rk, y=ys_rk, mode='lines+markers',
                        name="Solución RK", line=dict(color='#00cfcc', width=2.5),
                        marker=dict(size=5)))
                    fig_sf.update_layout(template="plotly_dark",
                        title="Campo de Pendientes + Solución",
                        xaxis_title="x", yaxis_title="y",
                        xaxis=dict(range=[x_min_sf, x_max_sf]),
                        yaxis=dict(range=[y_min_sf, y_max_sf], scaleanchor="x"))
                    st.plotly_chart(fig_sf, use_container_width=True)

                with tab3:
                    if y_exacta is not None and len(y_exacta) == len(ys_rk):
                        errores_abs = np.abs(np.array(ys_rk) - y_exacta)
                        fig_err = go.Figure()
                        fig_err.add_trace(go.Bar(x=xs_rk, y=errores_abs,
                            name="Error Absoluto", marker_color='#e03ce6'))
                        fig_err.update_layout(template="plotly_dark",
                            title="Error Absoluto por Paso",
                            xaxis_title="x", yaxis_title="|y_num - y_exact|")
                        st.plotly_chart(fig_err, use_container_width=True)

                        st.info(f"**Error máximo:** {formatear_error(max(errores_abs))} en x = {xs_rk[np.argmax(errores_abs)]:{fmt}}")
                        st.info(f"**Error promedio:** {formatear_error(np.mean(errores_abs))}")
                    else:
                        st.warning("No se pudo obtener la solución exacta. Ingresá una solución analítica o verificá la función.")

            else:  # Sistema de 2 EDOs
                if mostrar_formulas:
                    st.subheader("Fórmulas — RK4 Sistema")
                    st.latex(r"\frac{dy_1}{dx} = f_1(x, y_1, y_2) \qquad \frac{dy_2}{dx} = f_2(x, y_1, y_2)")
                    st.latex(r"y_{j,i+1} = y_{j,i} + \frac{h}{6}\left(k_1^{(j)} + 2k_2^{(j)} + 2k_3^{(j)} + k_4^{(j)}\right)")
                    with st.expander("📖 Notación"):
                        st.markdown("""
| Símbolo | Significado |
|---|---|
| $y_{j,i}$ | Aproximación de la variable dependiente $j$ (ej. $y_1$ o $y_2$) en el paso $i$ |
| $h$ | Tamaño del paso |
| $k_m^{(j)}$ | Pendiente $m$-ésima evaluada para la variable $j$ |
| $f_j$ | Función que define la derivada de la variable $j$: $dy_j/dx$ |
""")

                st.subheader("Resultado — Sistema RK4")
                df_sys, xs_sys, y1s_sys, y2s_sys = metodo_rk4_sistema(
                    func_input_1, func_input_2, rk_x0, rk_y1_0, rk_y2_0, rk_h, rk_n)

                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("y₁ final ≈", f"{y1s_sys[-1]:{fmt}}")
                col_r2.metric("y₂ final ≈", f"{y2s_sys[-1]:{fmt}}")
                col_r3.metric("x final", f"{xs_sys[-1]:{fmt}}")
                st.dataframe(format_df(df_sys), use_container_width=True)

                # --- DESARROLLO PASO A PASO ---
                with st.expander("📋 Desarrollo paso a paso", expanded=False):
                    for _i in range(len(df_sys) - 1):
                        _r = df_sys.iloc[_i]
                        _r_next = df_sys.iloc[_i+1]
                        _x, _y1, _y2 = _r["xᵢ"], _r["y₁ᵢ"], _r["y₂ᵢ"]
                        _k1_1, _k2_1, _k3_1, _k4_1 = _r["k₁⁽¹⁾"], _r["k₂⁽¹⁾"], _r["k₃⁽¹⁾"], _r["k₄⁽¹⁾"]
                        _k1_2, _k2_2, _k3_2, _k4_2 = _r["k₁⁽²⁾"], _r["k₂⁽²⁾"], _r["k₃⁽²⁾"], _r["k₄⁽²⁾"]
                        
                        st.markdown(f"### Paso {_i} → {_i+1}")
                        bloque = (
                            f"Datos: x_{_i} = {_x:{fmt}}, y1_{_i} = {_y1:{fmt}}, y2_{_i} = {_y2:{fmt}}, h = {rk_h:{fmt}}\n\n"
                            f"--- Ecuación 1 ---\n"
                            f"k1⑴ = {_k1_1:{fmt}}\n"
                            f"k2⑴ = {_k2_1:{fmt}}\n"
                            f"k3⑴ = {_k3_1:{fmt}}\n"
                            f"k4⑴ = {_k4_1:{fmt}}\n"
                            f"y1_{{{_i+1}}} = y1_{_i} + (h/6)·(k1⑴ + 2·k2⑴ + 2·k3⑴ + k4⑴)\n"
                            f"y1_{{{_i+1}}} = {_y1:{fmt}} + ({rk_h:{fmt}}/6)·({_k1_1:{fmt}} + 2·{_k2_1:{fmt}} + 2·{_k3_1:{fmt}} + {_k4_1:{fmt}})\n"
                            f"y1_{{{_i+1}}} = {_r_next['y₁ᵢ']:{fmt}}\n\n"
                            f"--- Ecuación 2 ---\n"
                            f"k1⑵ = {_k1_2:{fmt}}\n"
                            f"k2⑵ = {_k2_2:{fmt}}\n"
                            f"k3⑵ = {_k3_2:{fmt}}\n"
                            f"k4⑵ = {_k4_2:{fmt}}\n"
                            f"y2_{{{_i+1}}} = y2_{_i} + (h/6)·(k1⑵ + 2·k2⑵ + 2·k3⑵ + k4⑵)\n"
                            f"y2_{{{_i+1}}} = {_y2:{fmt}} + ({rk_h:{fmt}}/6)·({_k1_2:{fmt}} + 2·{_k2_2:{fmt}} + 2·{_k3_2:{fmt}} + {_k4_2:{fmt}})\n"
                            f"y2_{{{_i+1}}} = {_r_next['y₂ᵢ']:{fmt}}"
                        )
                        st.code(bloque, language="text")

                tab1, tab2 = st.tabs(["Soluciones y₁(x), y₂(x)", "Retrato de Fase"])

                with tab1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=xs_sys, y=y1s_sys, mode='lines+markers',
                        name="y₁(x)", line=dict(color='#00cfcc', width=2), marker=dict(size=5)))
                    fig.add_trace(go.Scatter(x=xs_sys, y=y2s_sys, mode='lines+markers',
                        name="y₂(x)", line=dict(color='#e03ce6', width=2), marker=dict(size=5)))
                    fig.update_layout(template="plotly_dark",
                        title="Soluciones del Sistema de EDOs",
                        xaxis_title="x", yaxis_title="y")
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    fig_phase = go.Figure()
                    fig_phase.add_trace(go.Scatter(x=y1s_sys, y=y2s_sys, mode='lines+markers',
                        name="Trayectoria", line=dict(color='#f0a500', width=2), marker=dict(size=5)))
                    fig_phase.add_trace(go.Scatter(x=[y1s_sys[0]], y=[y2s_sys[0]], mode='markers',
                        name="Inicio", marker=dict(size=12, color='#00ff00', symbol='star')))
                    fig_phase.add_trace(go.Scatter(x=[y1s_sys[-1]], y=[y2s_sys[-1]], mode='markers',
                        name="Final", marker=dict(size=12, color='#ff4b4b', symbol='diamond')))
                    fig_phase.update_layout(template="plotly_dark",
                        title="Retrato de Fase (y₁ vs y₂)",
                        xaxis_title="y₁", yaxis_title="y₂")
                    st.plotly_chart(fig_phase, use_container_width=True)

        elif metodo_sel in ["Punto Fijo", "Punto Fijo y Aitken"]:
            if mostrar_formulas:
                st.subheader("Fórmulas")
                st.latex(r"x_{n+1} = g(x_n)")
                st.latex(r"\hat{x}_n = x_n - \frac{(x_{n+1} - x_n)^2}{x_{n+2} - 2x_{n+1} + x_n}")
                st.latex(r"\text{Error} = \left|\frac{\hat{x}_n - x_n}{\hat{x}_n}\right| \times 100\%")
                with st.expander("📖 Notación"):
                    st.markdown("""
| Símbolo | Significado |
|---|---|
| $g(x)$ | Función de iteración de punto fijo |
| $x_n$ | Aproximación actual en la iteración $n$ |
| $x_{n+1}$ | Siguiente valor: $g(x_n)$ |
| $\\hat{x}_n$ | Valor acelerado de Aitken (Δ²) |
| Error (%) | Error relativo porcentual basado en $\\hat{x}_n$ |
""")

            st.subheader("Resultado")
            df_pf, estado_pf, raiz_pf, err_pf = metodo_punto_fijo_aitken(func_input, x0_pf, tol_pf, iter_pf)

            if df_pf is not None and len(df_pf) > 0:
                if estado_pf == "convergencia":
                    st.success(f"✅ Convergencia alcanzada — Punto fijo ≈ {raiz_pf:{fmt}}")
                elif estado_pf == "error_eval":
                    st.error("❌ Error al evaluar g(x). Verificá la función ingresada.")
                else:
                    st.warning(f"⚠️ Máximo de iteraciones alcanzado. Última estimación ≈ {raiz_pf:{fmt}}")

                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Punto Fijo ≈", f"{raiz_pf:{fmt}}")
                col_r2.metric("Iteraciones", len(df_pf))
                col_r3.metric("Error Final (%)", formatear_error(err_pf) if err_pf is not None else "N/A")

                st.dataframe(format_df(df_pf), use_container_width=True)

                # --- DESARROLLO PASO A PASO ---
                with st.expander("📋 Desarrollo paso a paso", expanded=False):
                    _xn_pf = x0_pf
                    for _i_pf in range(len(df_pf)):
                        _row = df_pf.iloc[_i_pf]
                        _xn = _row["Xn"]
                        _xn1 = _row["Xn+1"]
                        _xn2 = _row["Xn+2"]
                        _xn_star = _row["Xn*"]
                        _err = _row["Error %"]
                        bloque = (
                            f"--- Iteración n={int(_row['n'])} ---\n"
                            f"Xn = {_xn}\n\n"
                            f"Xn+1 = g(Xn) = g({_xn}) = {_xn1}\n"
                            f"Xn+2 = g(Xn+1) = g({_xn1}) = {_xn2}\n\n"
                        )
                        if _xn_star != "":
                            _denom = _xn2 - 2*_xn1 + _xn
                            bloque += (
                                f"Xn* = Xn - (Xn+1 - Xn)² / (Xn+2 - 2·Xn+1 + Xn)\n"
                                f"Xn* = {_xn} - ({_xn1} - {_xn})² / ({_xn2} - 2·{_xn1} + {_xn})\n"
                                f"Xn* = {_xn} - {(_xn1 - _xn)**2:{fmt}} / {_denom:{fmt}}\n"
                                f"Xn* = {_xn_star:{fmt}}\n\n"
                                f"Error = |Xn* - Xn| / |Xn*| × 100%\n"
                                f"Error = |{_xn_star:{fmt}} - {_xn}| / |{_xn_star:{fmt}}| × 100%\n"
                                f"Error = {_err:{fmt}}%"
                            )
                        else:
                            bloque += "Xn* = indefinido (denominador ≈ 0)"
                        st.code(bloque, language="text")

                # --- GRÁFICO: convergencia punto fijo vs Aitken ---
                xn_vals = df_pf["Xn"].tolist()
                xhat_vals = [v if v != "" else None for v in df_pf["Xn*"].tolist()]
                n_vals = df_pf["n"].tolist()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=n_vals, y=xn_vals, mode='lines+markers',
                    name='xₙ (Punto Fijo)', line=dict(color='#00cfcc', width=2),
                    marker=dict(size=6)
                ))

                # Solo trazar x_hat donde existe
                xhat_n = [n for n, v in zip(n_vals, xhat_vals) if v is not None]
                xhat_y = [v for v in xhat_vals if v is not None]
                if xhat_y:
                    fig.add_trace(go.Scatter(
                        x=xhat_n, y=xhat_y, mode='lines+markers',
                        name='x̂ₙ (Aitken)', line=dict(color='#e03ce6', width=2, dash='dash'),
                        marker=dict(size=7, symbol='diamond')
                    ))

                fig.update_layout(
                    template="plotly_dark",
                    title=f"Convergencia — Punto Fijo vs Aitken (g(x) = {func_input})",
                    xaxis_title="Iteración n",
                    yaxis_title="Valor"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se pudo calcular. Verificá la función g(x) y el valor inicial.")

        else: # Raíces
            if mostrar_formulas:
                st.subheader("Fórmulas")
                if metodo_sel == "Bisección":
                    st.latex(
                        r"x_n = \frac{a + b}{2}"
                    )
                    st.latex(
                        r"\text{Error} = \left|\frac{x_n - x_{n-1}}{x_n}\right| \times 100\%"
                    )
                    st.latex(
                        r"\text{Criterio de cambio de signo: } f(a) \cdot f(b) < 0"
                    )
                    with st.expander("📖 Notación"):
                        st.markdown("""
| Símbolo | Significado |
|---|---|
| $a,\\, b$ | Extremos del intervalo activo en cada iteración |
| $x_n$ | Punto medio del intervalo: aproximación a la raíz |
| $x_{n-1}$ | Aproximación de la iteración anterior |
| $f(a),\\, f(b)$ | Valores de $f$ en los extremos del intervalo |
| Error (%) | Error relativo porcentual entre iteraciones sucesivas |
""")
                else:  # Newton-Raphson
                    st.latex(
                        r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}"
                    )
                    st.latex(
                        r"f'(x_n) \approx \frac{-f(x_n+2h) + 8f(x_n+h) - 8f(x_n-h) + f(x_n-2h)}{12h}"
                    )
                    st.latex(
                        r"\text{Error} = \left|\frac{x_{n+1} - x_n}{x_{n+1}}\right| \times 100\%"
                    )
                    with st.expander("📖 Notación"):
                        st.markdown("""
| Símbolo | Significado |
|---|---|
| $x_n$ | Aproximación actual a la raíz |
| $x_{n+1}$ | Nueva aproximación calculada en la iteración |
| $f(x_n)$ | Valor de la función en $x_n$ |
| $f'(x_n)$ | Derivada de $f$ en $x_n$ (calculada numéricamente) |
| $h$ | Paso para la diferenciación numérica ($h = 10^{-5}$) |
| Error (%) | Error relativo porcentual entre iteraciones sucesivas |
""")
            st.subheader("Resultado")
            df, estado, raiz, err = metodo_biseccion(func_input, a_in, b_in, tol_in, iter_in) if metodo_sel == "Bisección" else metodo_newton_raphson(func_input, x0_in, tol_in, iter_in)
            if df is not None:
                st.success(f"Raíz: {raiz:{fmt}}")
                st.dataframe(df)

                # --- DESARROLLO PASO A PASO ---
                with st.expander("📋 Desarrollo paso a paso", expanded=False):
                    if metodo_sel == "Bisección":
                        for _i_b in range(len(df)):
                            _r = df.iloc[_i_b]
                            _a_b = _r["a"]
                            _b_b = _r["b"]
                            _xn_b = _r["x_n"]
                            _fxn_b = _r["f(x_n)"]
                            _fa_b = evaluar_f(func_input, _a_b)
                            bloque = (
                                f"--- Iteración {int(_r['Iter'])} ---\n"
                                f"a = {_a_b:{fmt}},  b = {_b_b:{fmt}}\n\n"
                                f"xₙ = (a + b) / 2 = ({_a_b:{fmt}} + {_b_b:{fmt}}) / 2 = {_xn_b:{fmt}}\n\n"
                                f"f(xₙ) = f({_xn_b:{fmt}}) = {_fxn_b:{fmt}}\n"
                            )
                            if _fa_b is not None:
                                _prod = _fa_b * _fxn_b
                                bloque += f"f(a)·f(xₙ) = {_fa_b:{fmt}} · {_fxn_b:{fmt}} = {_prod:{fmt}}"
                                if _prod < 0:
                                    bloque += f" < 0 → b = xₙ = {_xn_b:{fmt}}"
                                else:
                                    bloque += f" > 0 → a = xₙ = {_xn_b:{fmt}}"
                            if _r['Error (%)'] != '' and _r['Error (%)'] != 0 and _i_b > 0:
                                _x_ant = df.iloc[_i_b - 1]["x_n"]
                                bloque += (
                                    f"\n\nError = |xₙ - xₙ₋₁| / |xₙ| × 100%\n"
                                    f"Error = |{_xn_b:{fmt}} - {_x_ant:{fmt}}| / |{_xn_b:{fmt}}| × 100%\n"
                                    f"Error = {_r['Error (%)']:{fmt}}%"
                                )
                            st.code(bloque, language="text")
                    else:  # Newton-Raphson
                        for _i_nr in range(len(df)):
                            _r = df.iloc[_i_nr]
                            _xn_nr = _r["x_n"]
                            _fxn_nr = _r["f(x_n)"]
                            _dfxn_nr = _r["f'(x_n)"]
                            bloque = (
                                f"--- Iteración {int(_r['Iter'])} ---\n"
                                f"xₙ = {_xn_nr:{fmt}}\n\n"
                                f"f(xₙ) = f({_xn_nr:{fmt}}) = {_fxn_nr:{fmt}}\n"
                                f"f'(xₙ) = {_dfxn_nr:{fmt}}\n\n"
                                f"xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)\n"
                                f"xₙ₊₁ = {_xn_nr:{fmt}} - {_fxn_nr:{fmt}}/{_dfxn_nr:{fmt}}\n"
                                f"xₙ₊₁ = {_xn_nr:{fmt}} - {_fxn_nr/_dfxn_nr if _dfxn_nr != 0 else 0:{fmt}}\n"
                                f"xₙ₊₁ = {_xn_nr - _fxn_nr/_dfxn_nr if _dfxn_nr != 0 else _xn_nr:{fmt}}"
                            )
                            _err_nr = _r["Error (%)"]
                            if _err_nr != '' and _i_nr > 0:
                                _x_ant = df.iloc[_i_nr - 1]["x_n"]
                                bloque += (
                                    f"\n\nError = |xₙ - xₙ₋₁| / |xₙ| × 100%\n"
                                    f"Error = |{_xn_nr:{fmt}} - {_x_ant:{fmt}}| / |{_xn_nr:{fmt}}| × 100%\n"
                                    f"Error = {_err_nr:{fmt}}%"
                                )
                            st.code(bloque, language="text")

                # --- GRÁFICOS DE CONVERGENCIA Y FUNCIÓN ---
                st.subheader("Gráficos")
                tab1, tab2 = st.tabs(["Función y Raíz", "Convergencia de xₙ"])
                
                with tab1:
                    fig_func = go.Figure()
                    
                    x_vals_df = df["x_n"].values
                    margen = (max(x_vals_df) - min(x_vals_df)) * 0.5
                    if margen == 0: margen = 1.0
                    x_plot = np.linspace(min(x_vals_df) - margen, max(x_vals_df) + margen, 300)
                    y_plot = [evaluar_f(func_input, xi) for xi in x_plot]
                    
                    if None not in y_plot:
                        fig_func.add_trace(go.Scatter(x=x_plot, y=y_plot, name="f(x)", line=dict(color='#86e012')))
                        fig_func.add_hline(y=0, line_color='white', line_dash='dash')
                        
                        y_vals_df = df["f(x_n)"].values
                        fig_func.add_trace(go.Scatter(
                            x=x_vals_df, y=y_vals_df, mode='markers+lines',
                            name="Iteraciones xₙ", line=dict(color='#e03ce6', dash='dot'),
                            marker=dict(size=8)
                        ))
                        
                        fig_func.add_trace(go.Scatter(
                            x=[raiz], y=[evaluar_f(func_input, raiz)], mode='markers',
                            name="Raíz Final", marker=dict(size=12, color='white', symbol='star')
                        ))
                        
                        fig_func.update_layout(template="plotly_dark", title=f"Búsqueda de Raíz ({metodo_sel})", xaxis_title="x", yaxis_title="f(x)")
                        st.plotly_chart(fig_func, use_container_width=True)
                    else:
                        st.warning("No se pudo dibujar f(x).")

                with tab2:
                    fig_conv = go.Figure()
                    fig_conv.add_trace(go.Scatter(
                        x=df["Iter"].values, y=x_vals_df, mode='lines+markers',
                        name="Aproximación xₙ", line=dict(color='#00cfcc', width=2),
                        marker=dict(size=8)
                    ))
                    fig_conv.update_layout(template="plotly_dark", title="Convergencia de xₙ", xaxis_title="Iteración n", yaxis_title="xₙ")
                    st.plotly_chart(fig_conv, use_container_width=True)
            else:
                st.error("No se pudo calcular la raíz. Verificá la función y los valores iniciales.")