
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
import requests
import io
import datetime

# --- CONFIGURACIÓN PARA COLAB Y NUBE ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ---------------------------------------

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable

warnings.filterwarnings('ignore')

st.set_page_config(page_title="⚙️ Calculadora Taguchi Pro", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #475569; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚙️ Calculadora de Metodología Taguchi Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Análisis de Arreglos Ortogonales, Cociente S/N, Varianza y Medias.</div>', unsafe_allow_html=True)
st.markdown("---")

OA_MATRICES = {
    "L4 (2^3) - 4 corridas": [[1,1,1], [1,2,2], [2,1,2], [2,2,1]],
    "L8 (2^7) - 8 corridas": [
        [1,1,1,1,1,1,1], [1,1,1,2,2,2,2], [1,2,2,1,1,2,2], [1,2,2,2,2,1,1],
        [2,1,2,1,2,1,2], [2,1,2,2,1,2,1], [2,2,1,1,2,2,1], [2,2,1,2,1,1,2]
    ],
    "L9 (3^4) - 9 corridas": [
        [1,1,1,1], [1,2,2,2], [1,3,3,3], [2,1,2,3], [2,2,3,1],
        [2,3,1,2], [3,1,3,2], [3,2,1,3], [3,3,2,1]
    ],
    "L16 (4^5) - 16 corridas": [
        [1,1,1,1,1], [1,2,2,2,2], [1,3,3,3,3], [1,4,4,4,4],
        [2,1,2,3,4], [2,2,1,4,3], [2,3,4,1,2], [2,4,3,2,1],
        [3,1,3,4,2], [3,2,4,3,1], [3,3,1,2,4], [3,4,2,1,3],
        [4,1,4,2,3], [4,2,3,1,4], [4,3,2,4,1], [4,4,1,3,2]
    ]
}

def calculate_sn_ratio(values, criterion="larger", nominal_type="type1"):
    y = np.array(values, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)
    if n == 0: return np.nan
    if criterion == "larger":
        mean_inv_sq = np.mean(1.0 / (y ** 2))
        return -10.0 * np.log10(mean_inv_sq) if mean_inv_sq > 0 else np.nan
    elif criterion == "smaller":
        mean_sq = np.mean(y ** 2)
        return -10.0 * np.log10(mean_sq) if mean_sq > 0 else np.nan
    elif criterion == "nominal":
        mean_val = np.mean(y)
        var_val = np.var(y, ddof=1) if n > 1 else 1e-9
        if var_val <= 0: var_val = 1e-9
        if nominal_type == "type1":
            return 10.0 * np.log10((mean_val ** 2) / var_val) if mean_val != 0 else np.nan
        else:
            return -10.0 * np.log10(var_val)
    return np.nan

def build_response_table(df, factors, metric_col):
    max_lvls = max(len(df[f].unique()) for f in factors)
    row_names = [f"Nivel {i}" for i in range(1, max_lvls + 1)] + ["Delta", "Clasificar"]
    df_res = pd.DataFrame(index=row_names, columns=factors)
    for f in factors:
        means = df.groupby(f)[metric_col].mean()
        levels = sorted(list(means.index))
        for idx, lvl in enumerate(levels, 1):
            df_res.loc[f"Nivel {idx}", f] = means[lvl]
        df_res.loc["Delta", f] = means.max() - means.min()
    deltas = df_res.loc["Delta"].astype(float)
    ranks = deltas.rank(ascending=False, method="min").astype(int)
    for f in factors:
        df_res.loc["Clasificar", f] = int(ranks[f])
    return df_res

def plot_main_effects_figure(df, factors, metric_col, title_metric, ylabel):
    k = len(factors)
    fig, axes = plt.subplots(1, k, figsize=(max(3.2 * k, 6), 3.6), sharey=True, dpi=130)
    if k == 1: axes = [axes]
    grand_mean = df[metric_col].mean()
    for ax, f in zip(axes, factors):
        means = df.groupby(f)[metric_col].mean()
        levels_str = [str(x) for x in means.index]
        ax.plot(levels_str, means.values, marker='o', linewidth=2.2, markersize=6, color='#2563EB', zorder=3)
        ax.axhline(grand_mean, color='#94A3B8', linestyle='--', linewidth=1.2)
        ax.set_title(f"{f}", fontsize=10.5, fontweight='bold', color='#1E293B', pad=8)
        ax.set_xlabel("Nivel", fontsize=8.5, labelpad=4)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=8)
    axes[0].set_ylabel(ylabel, fontsize=9, fontweight='bold', labelpad=6)
    fig.suptitle(f"Gráfica de Efectos Principales: {title_metric}", fontsize=11.5, fontweight='bold', color='#0F172A', y=1.03)
    plt.tight_layout()
    return fig

def generate_pdf_report(study_name, criterion_desc, df_runs, factors, rep_cols, resp_sn, resp_mean, df_anova, opt_info, conclusions, fig_sn_buf, fig_mean_buf):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, textColor=colors.HexColor('#1E3A8A'), spaceAfter=4)
    subtitle_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontName='Helvetica', fontSize=9, textColor=colors.HexColor('#475569'), spaceAfter=10)
    h2_style = ParagraphStyle('H2_Custom', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=11.5, textColor=colors.HexColor('#1E3A8A'), spaceBefore=10, spaceAfter=5)
    body_style = ParagraphStyle('Body_Custom', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, textColor=colors.HexColor('#1F2937'))

    story = [
        Paragraph(f"Reporte de Optimización y Diseño Robusto (Taguchi)", title_style),
        Paragraph(f"Estudio: <b>{study_name}</b> | Fecha: {datetime.date.today().strftime('%d/%m/%Y')} | Criterio S/N: <b>{criterion_desc}</b>", subtitle_style),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=8),
        Paragraph("1. Resumen de la Configuración Experimental", h2_style),
        Paragraph(f"• <b>Número de corridas:</b> {len(df_runs)} | <b>Factores de control:</b> {len(factors)}", body_style),
        Spacer(1, 6),
        Paragraph("2. Tabla de Respuesta para la Relación Señal/Ruido (S/N en dB)", h2_style)
    ]
    table_data_sn = [["Nivel / Parámetro"] + list(resp_sn.columns)]
    for idx_label in resp_sn.index:
        row = [str(idx_label)]
        for col in resp_sn.columns:
            val = resp_sn.loc[idx_label, col]
            row.append(f"{val:.3f}" if isinstance(val, (float, int)) and idx_label != "Clasificar" else str(int(val) if idx_label=="Clasificar" else val))
        table_data_sn.append(row)
    t_sn = Table(table_data_sn, hAlign='LEFT')
    t_sn.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 8), ('ALIGN', (1, 0), (-1, -1), 'CENTER'), ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])]))
    story.extend([t_sn, Spacer(1, 8), Paragraph("3. Tabla de Respuesta para las Medias", h2_style)])

    table_data_m = [["Nivel / Parámetro"] + list(resp_mean.columns)]
    for idx_label in resp_mean.index:
        row = [str(idx_label)]
        for col in resp_mean.columns:
            val = resp_mean.loc[idx_label, col]
            row.append(f"{val:.3f}" if isinstance(val, (float, int)) and idx_label != "Clasificar" else str(int(val) if idx_label=="Clasificar" else val))
        table_data_m.append(row)
    t_m = Table(table_data_m, hAlign='LEFT')
    t_m.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F766E')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 8), ('ALIGN', (1, 0), (-1, -1), 'CENTER'), ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])]))
    story.extend([t_m, Spacer(1, 8)])

    if df_anova is not None and not df_anova.empty:
        story.append(Paragraph("4. Análisis de Varianza (ANOVA)", h2_style))
        anova_data = [list(df_anova.columns)]
        for _, row in df_anova.iterrows():
            r_vals = [("-" if pd.isna(row[col]) else (f"{row[col]:.4f}" if isinstance(row[col], float) and col in ["SS", "MS", "F", "P-Valor"] else f"{row[col]:.2f}%" if isinstance(row[col], float) else str(row[col]))) for col in df_anova.columns]
            anova_data.append(r_vals)
        t_anova = Table(anova_data, hAlign='LEFT')
        t_anova.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#334155')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 7.5), ('ALIGN', (1, 0), (-1, -1), 'CENTER'), ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])]))
        story.extend([t_anova, Spacer(1, 8)])

    if fig_sn_buf:
        story.extend([Paragraph("5. Gráficos de Efectos Principales", h2_style), Image(fig_sn_buf, width=520, height=140), Spacer(1, 4)])
    if fig_mean_buf:
        story.extend([Image(fig_mean_buf, width=520, height=140), Spacer(1, 8)])

    story.extend([Paragraph("6. Condiciones Óptimas y Recomendaciones", h2_style), Paragraph(f"• <b>Combinación Óptima para Robustez (S/N):</b> {opt_info['sn_levels']}", body_style), Paragraph(f"• <b>Valor S/N Predicho:</b> {opt_info['pred_sn']:.3f} dB", body_style), Spacer(1, 4)])
    for c in conclusions: story.append(Paragraph(f"• {c}", body_style))
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

def get_ejemplo_dataset():
    l16_factors = [[1, 0.0, 10, 0], [2, 0.0, 20, 5], [3, 0.0, 30, 10], [4, 0.0, 40, 20], [5, 0.5, 10, 5], [6, 0.5, 20, 0], [7, 0.5, 30, 20], [8, 0.5, 40, 10], [9, 1.0, 10, 10], [10, 1.0, 20, 20], [11, 1.0, 30, 0], [12, 1.0, 40, 5], [13, 3.0, 10, 20], [14, 3.0, 20, 10], [15, 3.0, 30, 5], [16, 3.0, 40, 0]]
    l16_results = [[8.27, 7.50, 8.38, 10.58, 10.32, 10.32, 13.23, 14.11, 13.66], [5.73, 6.17, 6.17, 7.50, 7.15, 7.50, 10.14, 10.58, 10.73], [4.41, 4.30, 4.41, 8.82, 9.00, 9.00, 8.82, 9.52, 9.20], [5.73, 5.29, 5.56, 10.20, 11.30, 11.45, 12.17, 12.17, 13.23], [7.06, 7.50, 7.06, 9.26, 8.55, 8.80, 13.23, 12.70, 12.08], [6.17, 6.40, 6.61, 8.20, 8.60, 9.08, 9.70, 9.70, 9.70], [7.06, 6.61, 7.06, 9.70, 10.58, 10.14, 15.43, 15.70, 14.29], [5.18, 4.63, 4.41, 7.76, 7.23, 7.23, 8.82, 8.38, 8.82], [7.94, 7.06, 7.50, 11.91, 11.02, 11.40, 12.35, 12.35, 13.23], [7.50, 7.06, 7.50, 12.21, 12.17, 12.17, 12.78, 13.23, 14.29], [4.76, 4.41, 4.76, 6.53, 6.61, 6.70, 8.82, 9.08, 9.26], [3.97, 3.97, 3.97, 6.35, 6.35, 6.79, 8.38, 7.94, 8.11], [8.35, 8.38, 8.47, 9.88, 9.70, 10.30, 14.11, 15.43, 15.87], [7.06, 7.28, 7.06, 11.29, 11.29, 11.29, 14.20, 13.05, 15.26], [4.92, 4.41, 4.85, 8.20, 8.38, 8.20, 11.91, 11.75, 11.64], [3.79, 3.92, 3.97, 5.56, 5.47, 5.29, 6.00, 6.00, 6.00]]
    cols_c = ['Corrida', 'Bentonita_pct', 'Cenizas_Volantes_pct', 'Humo_Silice_pct', 'Resistencia_7d_R1', 'Resistencia_7d_R2', 'Resistencia_7d_R3', 'Resistencia_14d_R1', 'Resistencia_14d_R2', 'Resistencia_14d_R3', 'Resistencia_28d_R1', 'Resistencia_28d_R2', 'Resistencia_28d_R3']
    return pd.DataFrame([f + r for f, r in zip(l16_factors, l16_results)], columns=cols_c)

with st.sidebar:
    st.header("📂 1. Carga de Datos y Configuración")
    metodo_carga = st.radio("Selecciona el origen de los datos:", ["✏️ Generar Tabla Manual", "📁 Cargar archivo CSV", "🔗 Cargar desde URL", "🔬 Dataset Ejemplo (Lechadas L16)"], index=3)
    df_raw = None

    if metodo_carga == "✏️ Generar Tabla Manual":
        tipo_arreglo = st.selectbox("Selecciona el Arreglo Ortogonal:", list(OA_MATRICES.keys()), index=3)
        if "4^5" in tipo_arreglo: max_niveles, max_factores = 4, 5
        elif "3^4" in tipo_arreglo: max_niveles, max_factores = 3, 4
        elif "2^3" in tipo_arreglo: max_niveles, max_factores = 2, 3
        elif "2^7" in tipo_arreglo: max_niveles, max_factores = 2, 7
        else: max_niveles, max_factores = 2, 15

        n_factores = st.number_input("Número de Factores:", min_value=2, max_value=max_factores, value=min(max_factores, 3), step=1)
        nombres_factores = []
        for i in range(n_factores):
            nombre = st.text_input(f"Nombre F{i+1}", value=f"Factor_{chr(65+i)}", key=f"f_name_{i}")
            niveles = [st.text_input(f"Nivel {n} (F{i+1})", value=f"N{n}", key=f"f_lvl_{i}_{n}") for n in range(1, max_niveles + 1)]
            nombres_factores.append({'nombre': nombre, 'niveles': niveles})

        st.markdown("---")
        respuestas_input = st.text_input("Nombres de las respuestas (separadas por coma):", value="Respuesta", key="resp_input")
        lista_respuestas = [r.strip() for r in respuestas_input.split(',') if r.strip()]
        n_replicas = st.number_input("Réplicas por corrida:", min_value=1, max_value=10, value=3, step=1)

        if st.button("🎲 Generar Tabla", type="primary"):
            oa_matrix = OA_MATRICES[tipo_arreglo]
            df_diseno = pd.DataFrame(oa_matrix, columns=[f"F{i+1}" for i in range(len(oa_matrix[0]))])
            df_diseno = df_diseno.iloc[:, :n_factores]
            df_diseno.columns = [f['nombre'] for f in nombres_factores]
            for i, f in enumerate(nombres_factores):
                mapeo = {j+1: f['niveles'][j] for j in range(len(f['niveles']))}
                df_diseno[f['nombre']] = df_diseno[f['nombre']].map(mapeo)
            df_diseno.insert(0, 'Run', range(1, len(df_diseno) + 1))
            for r in lista_respuestas:
                for rep in range(1, n_replicas + 1): df_diseno[f"{r}_R{rep}"] = np.nan
            st.session_state['df_taguchi'] = df_diseno
            st.session_state['factores_taguchi'] = [f['nombre'] for f in nombres_factores]
            st.session_state['respuestas_taguchi'] = lista_respuestas
            st.rerun()

    elif metodo_carga == "📁 Cargar archivo CSV":
        archivo = st.file_uploader("Sube tu archivo CSV", type=['csv'])
        if archivo is not None:
            try: df_raw = pd.read_csv(io.StringIO(archivo.getvalue().decode('utf-8')))
            except Exception as e: st.error(f"❌ Error al leer el CSV: {e}")

    elif metodo_carga == "🔗 Cargar desde URL":
        url = st.text_input("URL del archivo CSV:")
        if st.button("Cargar datos"):
            if url:
                try: df_raw = pd.read_csv(io.StringIO(requests.get(url).text))
                except Exception as e: st.error(f"❌ Error: {e}")

    elif metodo_carga == "🔬 Dataset Ejemplo (Lechadas L16)":
        df_raw = get_ejemplo_dataset()

    if df_raw is not None:
        all_cols = list(df_raw.columns)
        if "Resistencia_7d_R1" in all_cols:
            selected_factors = ['Bentonita_pct', 'Cenizas_Volantes_pct', 'Humo_Silice_pct']
            selected_reps = ['Resistencia_7d_R1', 'Resistencia_7d_R2', 'Resistencia_7d_R3']
        else:
            candidate_factors, candidate_responses = [], []
            for c in all_cols:
                c_lower = c.lower()
                if c_lower in ["corrida", "run", "prueba", "id", "orden", "ensayo"]: continue
                unique_cnt = df_raw[c].nunique()
                if (unique_cnt >= 2 and unique_cnt <= 8) or df_raw[c].dtype == object or "factor" in c_lower:
                    if not any(k in c_lower for k in ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "fuerza_", "resistencia_"]): candidate_factors.append(c)
                    else: candidate_responses.append(c)
                else: candidate_responses.append(c)
            if len(candidate_factors) == 0:
                n_rows = len(df_raw)
                for k, v in OA_MATRICES.items():
                    if len(v) == n_rows:
                        oa_df = pd.DataFrame(v, columns=[f"Factor_{chr(65+i)}" for i in range(len(v[0]))])
                        for col in oa_df.columns: df_raw[col] = oa_df[col].values
                        candidate_factors = list(oa_df.columns)
                        break
            selected_factors = st.multiselect("Factores de Control:", options=[c for c in df_raw.columns if c not in candidate_responses and c.lower() not in ["corrida", "run"]], default=candidate_factors)
            selected_reps = st.multiselect("Columnas de Respuestas:", options=[c for c in df_raw.columns if c not in selected_factors and c.lower() not in ["corrida", "run"]], default=candidate_responses)
        st.session_state['df_taguchi'] = df_raw
        st.session_state['factores_taguchi'] = selected_factors
        st.session_state['respuestas_taguchi'] = selected_reps

    st.markdown("---")
    criterion = st.selectbox("Función Objetivo Taguchi:", ["Más grande es mejor", "Más pequeño es mejor", "Nominal es lo mejor"], index=0)
    nominal_type = "type1"
    if criterion == "Nominal es lo mejor":
        nominal_type = st.radio("Tipo de Nominal:", ["Tipo I: 10*log(media^2 / var)", "Tipo II: -10*log(var)"], index=0)
        nominal_type = "type1" if "Tipo I" in nominal_type else "type2"

if 'df_taguchi' in st.session_state and len(st.session_state.get('factores_taguchi', [])) > 0 and len(st.session_state.get('respuestas_taguchi', [])) > 0:
    df_work = st.session_state['df_taguchi'].copy()
    factors = st.session_state['factores_taguchi']
    reps = st.session_state['respuestas_taguchi']

    df_work["Media"] = df_work[reps].mean(axis=1)
    df_work["Desv_Std"] = df_work[reps].std(axis=1, ddof=1).fillna(0.0)
    crit_key = "larger" if "grande" in criterion else ("smaller" if "pequeño" in criterion else "nominal")
    df_work["SN_Ratio"] = df_work[reps].apply(lambda row: calculate_sn_ratio(row, crit_key, nominal_type), axis=1)

    resp_sn = build_response_table(df_work, factors, "SN_Ratio")
    resp_mean = build_response_table(df_work, factors, "Media")

    gm_m = df_work["Media"].mean()
    ss_total_m = ((df_work["Media"] - gm_m) ** 2).sum()
    df_total_m = len(df_work) - 1
    anova_rows, ss_factors_sum, df_factors_sum = [], 0, 0
    for f in factors:
        l_means = df_work.groupby(f)["Media"].mean()
        l_counts = df_work.groupby(f)["Media"].count()
        ss_f = ((l_means - gm_m) ** 2 * l_counts).sum()
        df_f = len(l_means) - 1
        ms_f = ss_f / df_f if df_f > 0 else 0
        ss_factors_sum += ss_f
        df_factors_sum += df_f
        anova_rows.append({"Factor": f, "GL": df_f, "SS": ss_f, "MS": ms_f, "Contrib_pct": (ss_f / ss_total_m * 100.0) if ss_total_m > 0 else 0})

    ss_res = max(0.0, ss_total_m - ss_factors_sum)
    df_res = max(0, df_total_m - df_factors_sum)
    ms_res = (ss_res / df_res) if df_res > 0 else 0.0
    for r in anova_rows:
        r["F"] = (r["MS"] / ms_res) if ms_res > 0 else np.nan
        r["P-Valor"] = stats.f.sf(r["F"], r["GL"], df_res) if ms_res > 0 and not np.isnan(r["F"]) else np.nan

    df_anova = pd.DataFrame(anova_rows)
    if df_res > 0:
        df_anova = pd.concat([df_anova, pd.DataFrame([{"Factor": "Error", "GL": df_res, "SS": ss_res, "MS": ms_res, "F": np.nan, "P-Valor": np.nan, "Contrib_pct": (ss_res / ss_total_m * 100.0) if ss_total_m > 0 else 0}]), pd.DataFrame([{"Factor": "Total", "GL": df_total_m, "SS": ss_total_m, "MS": np.nan, "F": np.nan, "P-Valor": np.nan, "Contrib_pct": 100.0}])], ignore_index=True)

    opt_sn_levels = {f: df_work.groupby(f)["SN_Ratio"].mean().idxmax() for f in factors}
    opt_mean_levels = {}
    for f in factors:
        means = df_work.groupby(f)["Media"].mean()
        if crit_key == "larger": opt_mean_levels[f] = means.idxmax()
        elif crit_key == "smaller": opt_mean_levels[f] = means.idxmin()
        else: opt_mean_levels[f] = means.iloc[(means - gm_m).abs().argsort()[:1]].index[0]

    grand_mean_sn = df_work["SN_Ratio"].mean()
    pred_sn = grand_mean_sn + sum(df_work.groupby(f)["SN_Ratio"].mean()[opt_sn_levels[f]] - grand_mean_sn for f in factors)
    grand_mean_y = df_work["Media"].mean()
    pred_mean = grand_mean_y + sum(df_work.groupby(f)["Media"].mean()[opt_mean_levels[f]] - grand_mean_y for f in factors)
    opt_info = {"sn_levels": ", ".join([f"{f}={opt_sn_levels[f]}" for f in factors]), "mean_levels": ", ".join([f"{f}={opt_mean_levels[f]}" for f in factors]), "pred_sn": pred_sn, "pred_mean": pred_mean}
    conclusions = [f"<b>Factor Dominante:</b> <code>{resp_sn.loc['Clasificar'].astype(int).idxmin()}</code> ocupa el Rango 1 (Delta = {resp_sn.loc['Delta'].max():.3f} dB).", f"<b>Condición Óptima:</b> Seleccionar {opt_info['sn_levels']}, logrando un S/N predicho de {pred_sn:.2f} dB.", "<b>Validación:</b> Se recomienda realizar corridas de confirmación en la condición óptima."]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Corridas", len(df_work)); c2.metric("Factores", len(factors)); c3.metric("Media Global (Y)", f"{grand_mean_y:.3f}"); c4.metric("Media Global S/N", f"{grand_mean_sn:.3f} dB")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Matriz de Corridas", "📊 Tablas de Respuesta", "📈 Gráficos de Efectos", "📐 ANOVA", "🏆 Optimización & Reporte"])
    with tab1:
        show_cols = (["Run", "Corrida"] if "Run" in df_work.columns else []) + factors + reps + ["Media", "Desv_Std", "SN_Ratio"]
        st.dataframe(df_work[[c for c in show_cols if c in df_work.columns]].style.format({"Media": "{:.3f}", "Desv_Std": "{:.3f}", "SN_Ratio": "{:.3f}"}), use_container_width=True)
    with tab2:
        c_t1, c_t2 = st.columns(2)
        with c_t1: st.dataframe(resp_sn.style.format(precision=3), use_container_width=True)
        with c_t2: st.dataframe(resp_mean.style.format(precision=3), use_container_width=True)
    with tab3:
        fig_sn = plot_main_effects_figure(df_work, factors, "SN_Ratio", "Relación S/N", "Media S/N (dB)")
        st.pyplot(fig_sn)
        fig_mean = plot_main_effects_figure(df_work, factors, "Media", "Media de Respuesta", "Media (Y)")
        st.pyplot(fig_mean)
    with tab4:
        st.dataframe(df_anova.style.format({"SS": "{:.4f}", "MS": "{:.4f}", "F": "{:.3f}", "P-Valor": "{:.4f}", "Contrib_pct": "{:.2f}%"}), use_container_width=True)
    with tab5:
        st.markdown(f"**Condición Óptima (S/N):** `{opt_info['sn_levels']}` | **Predicción:** {pred_sn:.3f} dB")
        st.markdown(f"**Condición Óptima (Media):** `{opt_info['mean_levels']}` | **Predicción:** {pred_mean:.3f}")
        st.markdown("### 📝 Conclusiones")
        for c in conclusions: st.markdown(f"- {c}", unsafe_allow_html=True)
        if st.button("Generar PDF"):
            buf_sn, buf_m = io.BytesIO(), io.BytesIO()
            fig_sn.savefig(buf_sn, format='png', bbox_inches='tight'); buf_sn.seek(0)
            fig_mean.savefig(buf_m, format='png', bbox_inches='tight'); buf_m.seek(0)
            pdf_bytes = generate_pdf_report("Estudio Taguchi", criterion, df_work, factors, reps, resp_sn, resp_mean, df_anova, opt_info, conclusions, buf_sn, buf_m)
            st.download_button(label="📥 Descargar Reporte PDF", data=pdf_bytes, file_name=f"Reporte_Taguchi_{datetime.date.today().strftime('%Y%m%d')}.pdf", mime="application/pdf", use_container_width=True)
else:
    st.info("👈 Por favor, genera una tabla o carga datos desde la barra lateral para comenzar el análisis.")
