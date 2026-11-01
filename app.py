import io
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st

# ReportLab for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable

# Page configuration
st.set_page_config(
    page_title="Calculadora Taguchi Pro · Diseño Robusto",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .highlight-box {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .conclusion-box {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 1rem 1.2rem;
        border-radius: 4px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Standard Taguchi Orthogonal Arrays library for auto-injection
ORTHOGONAL_ARRAYS = {
    4: pd.DataFrame([
        [1, 1, 1],
        [1, 2, 2],
        [2, 1, 2],
        [2, 2, 1]
    ], columns=["Factor_A", "Factor_B", "Factor_C"]),
    8: pd.DataFrame([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 2, 2, 2],
        [1, 2, 2, 1, 1, 2, 2],
        [1, 2, 2, 2, 2, 1, 1],
        [2, 1, 2, 1, 2, 1, 2],
        [2, 1, 2, 2, 1, 2, 1],
        [2, 2, 1, 1, 2, 2, 1],
        [2, 2, 1, 2, 1, 1, 2]
    ], columns=[f"Factor_{chr(65+i)}" for i in range(7)]),
    9: pd.DataFrame([
        [1, 1, 1, 1],
        [1, 2, 2, 2],
        [1, 3, 3, 3],
        [2, 1, 2, 3],
        [2, 2, 3, 1],
        [2, 3, 1, 2],
        [3, 1, 3, 2],
        [3, 2, 1, 3],
        [3, 3, 2, 1]
    ], columns=["Factor_A", "Factor_B", "Factor_C", "Factor_D"]),
    16: pd.DataFrame([
        [1, 1, 1], [1, 2, 2], [1, 3, 3], [1, 4, 4],
        [2, 1, 2], [2, 2, 1], [2, 3, 4], [2, 4, 3],
        [3, 1, 3], [3, 2, 4], [3, 3, 1], [3, 4, 2],
        [4, 1, 4], [4, 2, 3], [4, 3, 2], [4, 4, 1]
    ], columns=["Factor_A", "Factor_B", "Factor_C"])
}

def calculate_sn_ratio(values, criterion="larger", nominal_type="type1"):
    y = np.array(values, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)
    if n == 0:
        return np.nan
    if criterion == "larger":
        # -10 * log10( (1/n) * sum(1/y^2) )
        mean_inv_sq = np.mean(1.0 / (y ** 2))
        return -10.0 * np.log10(mean_inv_sq) if mean_inv_sq > 0 else np.nan
    elif criterion == "smaller":
        # -10 * log10( (1/n) * sum(y^2) )
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
        delta_val = means.max() - means.min()
        df_res.loc["Delta", f] = delta_val

    # Calculate ranks based on Delta
    deltas = df_res.loc["Delta"].astype(float)
    ranks = deltas.rank(ascending=False, method="min").astype(int)
    for f in factors:
        df_res.loc["Clasificar", f] = int(ranks[f])

    return df_res

def plot_main_effects_figure(df, factors, metric_col, title_metric, ylabel):
    k = len(factors)
    fig, axes = plt.subplots(1, k, figsize=(max(3.2 * k, 6), 3.6), sharey=True, dpi=130)
    if k == 1:
        axes = [axes]

    grand_mean = df[metric_col].mean()

    for ax, f in zip(axes, factors):
        means = df.groupby(f)[metric_col].mean()
        levels_str = [str(x) for x in means.index]
        vals = means.values

        ax.plot(levels_str, vals, marker='o', linewidth=2.2, markersize=6, color='#2563EB', zorder=3)
        ax.axhline(grand_mean, color='#94A3B8', linestyle='--', linewidth=1.2, label='Media global' if ax == axes[0] else None)
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
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#475569'),
        spaceAfter=10
    )
    h2_style = ParagraphStyle(
        'H2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=10,
        spaceAfter=5
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1F2937')
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1F2937'),
        leftIndent=12,
        spaceAfter=3
    )

    story = []
    # Header
    story.append(Paragraph(f"Reporte de Optimización y Diseño Robusto (Taguchi)", title_style))
    story.append(Paragraph(f"Estudio: <b>{study_name}</b> | Fecha: {datetime.date.today().strftime('%d/%m/%Y')} | Criterio S/N: <b>{criterion_desc}</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=8))

    # 1. Configuration summary
    story.append(Paragraph("1. Resumen de la Configuración Experimental", h2_style))
    story.append(Paragraph(f"• <b>Número de corridas:</b> {len(df_runs)} | <b>Factores de control:</b> {len(factors)} ({', '.join(factors)})", bullet_style))
    story.append(Paragraph(f"• <b>Número de réplicas por corrida:</b> {len(rep_cols)} | <b>Media global (Y):</b> {df_runs['Media'].mean():.3f} | <b>Media global S/N:</b> {df_runs['SN_Ratio'].mean():.3f} dB", bullet_style))
    story.append(Spacer(1, 6))

    # 2. Response Table S/N
    story.append(Paragraph("2. Tabla de Respuesta para la Relación Señal/Ruido (S/N en dB)", h2_style))
    table_data_sn = [["Nivel / Parámetro"] + list(resp_sn.columns)]
    for idx_label in resp_sn.index:
        row = [str(idx_label)]
        for col in resp_sn.columns:
            val = resp_sn.loc[idx_label, col]
            row.append(f"{val:.3f}" if isinstance(val, (float, int)) and idx_label != "Clasificar" else str(int(val) if idx_label=="Clasificar" else val))
        table_data_sn.append(row)

    t_sn = Table(table_data_sn, hAlign='LEFT')
    t_sn.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    story.append(t_sn)
    story.append(Spacer(1, 8))

    # 3. Response Table Means
    story.append(Paragraph("3. Tabla de Respuesta para las Medias", h2_style))
    table_data_m = [["Nivel / Parámetro"] + list(resp_mean.columns)]
    for idx_label in resp_mean.index:
        row = [str(idx_label)]
        for col in resp_mean.columns:
            val = resp_mean.loc[idx_label, col]
            row.append(f"{val:.3f}" if isinstance(val, (float, int)) and idx_label != "Clasificar" else str(int(val) if idx_label=="Clasificar" else val))
        table_data_m.append(row)

    t_m = Table(table_data_m, hAlign='LEFT')
    t_m.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F766E')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    story.append(t_m)
    story.append(Spacer(1, 8))

    # 4. ANOVA Table
    if df_anova is not None and not df_anova.empty:
        story.append(Paragraph("4. Análisis de Varianza (ANOVA)", h2_style))
        anova_data = [list(df_anova.columns)]
        for _, row in df_anova.iterrows():
            r_vals = []
            for col in df_anova.columns:
                v = row[col]
                if pd.isna(v):
                    r_vals.append("-")
                elif isinstance(v, float):
                    r_vals.append(f"{v:.4f}" if col in ["SS", "MS", "F", "P-Valor"] else f"{v:.2f}%")
                else:
                    r_vals.append(str(v))
            anova_data.append(r_vals)
        t_anova = Table(anova_data, hAlign='LEFT')
        t_anova.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#334155')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
            ('TOPPADDING', (0, 0), (-1, -1), 2.5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
        ]))
        story.append(t_anova)
        story.append(Spacer(1, 8))

    # 5. Embedded Figures
    if fig_sn_buf:
        story.append(Paragraph("5. Gráficos de Efectos Principales", h2_style))
        fig_sn_buf.seek(0)
        story.append(Image(fig_sn_buf, width=520, height=140))
        story.append(Spacer(1, 4))
    if fig_mean_buf:
        fig_mean_buf.seek(0)
        story.append(Image(fig_mean_buf, width=520, height=140))
        story.append(Spacer(1, 8))

    # 6. Optimal & Conclusions
    story.append(Paragraph("6. Condiciones Óptimas y Recomendaciones de Ingeniería", h2_style))
    story.append(Paragraph(f"• <b>Combinación Óptima para Robustez (S/N):</b> {opt_info['sn_levels']}", bullet_style))
    story.append(Paragraph(f"• <b>Valor S/N Predicho:</b> {opt_info['pred_sn']:.3f} dB", bullet_style))
    story.append(Paragraph(f"• <b>Combinación Óptima para la Media:</b> {opt_info['mean_levels']}", bullet_style))
    story.append(Paragraph(f"• <b>Media Predicha:</b> {opt_info['pred_mean']:.3f}", bullet_style))
    story.append(Spacer(1, 4))

    for c in conclusions:
        story.append(Paragraph(f"• {c}", bullet_style))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================
st.markdown('<div class="main-title">⚙️ Calculadora de Metodología Taguchi Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Diseño de Experimentos Robusto · Arreglos Ortogonales, S/N, ANOVA, Medias, Gráficos y Reporte PDF</div>', unsafe_allow_html=True)

# SIDEBAR: DATA INPUT & CONFIGURATION
with st.sidebar:
    st.header("📂 1. Carga de Datos")
    data_source = st.radio(
        "Selecciona el origen de los datos:",
        ["📁 Cargar archivo CSV propio", "🔬 Dataset Ejemplo: Lechadas Cemento (L16)", "🔧 Dataset Ejemplo: Ensamble Nylon (L9)"],
        index=1
    )

    df_raw = None
    if data_source == "📁 Cargar archivo CSV propio":
        uploaded_file = st.file_uploader("Sube tu archivo .CSV:", type=["csv"])
        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file)
                st.success("✅ CSV cargado correctamente.")
            except Exception as e:
                st.error(f"Error al leer el CSV: {e}")
    elif data_source == "🔬 Dataset Ejemplo: Lechadas Cemento (L16)":
        # Load preloaded cement dataset if exists, or generate in-memory
        if os.path.exists("dataset_lechadas_cemento_L16.csv"):
            df_raw = pd.read_csv("dataset_lechadas_cemento_L16.csv")
        elif os.path.exists("artifacts/file_generation/ttl=63d/output/dataset_lechadas_cemento_taguchi_L16.csv"):
            df_raw = pd.read_csv("artifacts/file_generation/ttl=63d/output/dataset_lechadas_cemento_taguchi_L16.csv")
        else:
            st.info("Generando dataset L16 en memoria...")
    else: # Nylon L9
        if os.path.exists("dataset_ensamble_nylon_L9.csv"):
            df_raw = pd.read_csv("dataset_ensamble_nylon_L9.csv")
        elif os.path.exists("artifacts/file_generation/ttl=63d/output/dataset_ensamble_nylon_taguchi_L9.csv"):
            df_raw = pd.read_csv("artifacts/file_generation/ttl=63d/output/dataset_ensamble_nylon_taguchi_L9.csv")

if df_raw is not None:
    # -------------------------------------------------------------------------
    # SMART COLUMN AUTO-DETECTION
    # -------------------------------------------------------------------------
    all_cols = list(df_raw.columns)

    # Check if this is the cement dataset with multiple curing periods
    curing_periods = []
    if any("7d" in c for c in all_cols): curing_periods.append("Resistencia a 7 días")
    if any("14d" in c for c in all_cols): curing_periods.append("Resistencia a 14 días")
    if any("28d" in c for c in all_cols): curing_periods.append("Resistencia a 28 días")

    selected_period = None
    if curing_periods:
        with st.sidebar:
            st.markdown("---")
            st.subheader("⏱️ Periodo de Ensayo Detectado")
            selected_period = st.selectbox("Selecciona la respuesta temporal a optimizar:", curing_periods, index=0)

    # Auto-detect potential factors (discrete/categorical with 2 to 8 unique values, excluding obvious runs/ids)
    candidate_factors = []
    candidate_responses = []

    for c in all_cols:
        c_lower = c.lower()
        if c_lower in ["corrida", "run", "prueba", "id", "orden", "ensayo"]:
            continue
        unique_cnt = df_raw[c].nunique()
        if (unique_cnt >= 2 and unique_cnt <= 8) or df_raw[c].dtype == object or "factor" in c_lower or "pct" in c_lower:
            # Check if it doesn't look like a response replicate
            if not any(k in c_lower for k in ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "fuerza_r", "resistencia_"]):
                candidate_factors.append(c)
            else:
                candidate_responses.append(c)
        else:
            candidate_responses.append(c)

    # Filter candidate responses if curing period is selected
    if selected_period == "Resistencia a 7 días":
        candidate_responses = [c for c in candidate_responses if "7d" in c or "7_dias" in c]
    elif selected_period == "Resistencia a 14 días":
        candidate_responses = [c for c in candidate_responses if "14d" in c or "14_dias" in c]
    elif selected_period == "Resistencia a 28 días":
        candidate_responses = [c for c in candidate_responses if "28d" in c or "28_dias" in c]

    # Handle case where CSV has NO factors (e.g. only responses or runs)
    auto_injected = False
    if len(candidate_factors) == 0:
        with st.sidebar:
            st.warning("⚠️ No se detectaron columnas de factores explícitas.")
            n_rows = len(df_raw)
            if n_rows in ORTHOGONAL_ARRAYS:
                st.info(f"💡 Se detectaron {n_rows} corridas. Puedes asignar automáticamente la matriz estándar L{n_rows}.")
                use_oa = st.checkbox(f"Inyectar automáticamente arreglo ortogonal L{n_rows}", value=True)
                if use_oa:
                    oa_df = ORTHOGONAL_ARRAYS[n_rows].copy()
                    for col in oa_df.columns:
                        df_raw[col] = oa_df[col].values
                    candidate_factors = list(oa_df.columns)
                    auto_injected = True

    with st.sidebar:
        st.markdown("---")
        st.subheader("🎯 2. Configuración de Variables")
        st.caption("La calculadora ha auto-detectado tus columnas. Puedes modificarlas si lo deseas:")

        selected_factors = st.multiselect(
            "Factores de Control:",
            options=[c for c in df_raw.columns if c not in candidate_responses and c.lower() not in ["corrida", "run", "prueba"]],
            default=candidate_factors if candidate_factors else []
        )

        available_resp_cols = [c for c in df_raw.columns if c not in selected_factors and c.lower() not in ["corrida", "run", "prueba"]]
        selected_reps = st.multiselect(
            "Columnas de Réplicas / Respuestas:",
            options=available_resp_cols,
            default=[c for c in candidate_responses if c in available_resp_cols]
        )

        st.markdown("---")
        st.subheader("📈 3. Criterio de Calidad (S/N)")
        criterion = st.selectbox(
            "Función Objetivo Taguchi:",
            ["Más grande es mejor (Larger is better)", "Más pequeño es mejor (Smaller is better)", "Nominal es lo mejor (Nominal is best)"],
            index=0
        )

        nominal_type = "type1"
        if criterion == "Nominal es lo mejor (Nominal is best)":
            nominal_type = st.radio(
                "Tipo de Nominal:",
                ["Tipo I: 10*log(media^2 / var)", "Tipo II: -10*log(var)"],
                index=0
            )
            nominal_type = "type1" if "Tipo I" in nominal_type else "type2"

    # MAIN APPLICATION LOGIC
    if len(selected_factors) > 0 and len(selected_reps) > 0:
        crit_key = "larger" if "grande" in criterion else ("smaller" if "pequeño" in criterion else "nominal")
        crit_label = "Más grande es mejor" if crit_key == "larger" else ("Más pequeño es mejor" if crit_key == "smaller" else "Nominal es lo mejor")

        # Copy and calculate run metrics
        df_calc = df_raw.copy()
        df_calc["Media"] = df_calc[selected_reps].mean(axis=1)
        df_calc["Desv_Std"] = df_calc[selected_reps].std(axis=1, ddof=1).fillna(0.0)
        df_calc["SN_Ratio"] = df_calc[selected_reps].apply(lambda row: calculate_sn_ratio(row, crit_key, nominal_type), axis=1)

        # Build Response Tables
        resp_sn = build_response_table(df_calc, selected_factors, "SN_Ratio")
        resp_mean = build_response_table(df_calc, selected_factors, "Media")

        # Build ANOVA
        # Calculate SS on trial means (standard Taguchi response table ANOVA)
        gm_m = df_calc["Media"].mean()
        ss_total_m = ((df_calc["Media"] - gm_m) ** 2).sum()
        df_total_m = len(df_calc) - 1

        anova_rows = []
        ss_factors_sum = 0
        df_factors_sum = 0
        for f in selected_factors:
            l_means = df_calc.groupby(f)["Media"].mean()
            l_counts = df_calc.groupby(f)["Media"].count()
            ss_f = ((l_means - gm_m) ** 2 * l_counts).sum()
            df_f = len(l_means) - 1
            ms_f = ss_f / df_f if df_f > 0 else 0
            ss_factors_sum += ss_f
            df_factors_sum += df_f
            anova_rows.append({
                "Factor": f, "GL": df_f, "SS": ss_f, "MS": ms_f,
                "Contrib_pct": (ss_f / ss_total_m * 100.0) if ss_total_m > 0 else 0
            })

        ss_res = max(0.0, ss_total_m - ss_factors_sum)
        df_res = max(0, df_total_m - df_factors_sum)
        ms_res = (ss_res / df_res) if df_res > 0 else 0.0

        for r in anova_rows:
            r["F"] = (r["MS"] / ms_res) if ms_res > 0 else np.nan
            r["P-Valor"] = stats.f.sf(r["F"], r["GL"], df_res) if ms_res > 0 and not np.isnan(r["F"]) else np.nan

        df_anova = pd.DataFrame(anova_rows)
        if df_res > 0:
            df_anova = pd.concat([
                df_anova,
                pd.DataFrame([{
                    "Factor": "Error / Residual", "GL": df_res, "SS": ss_res, "MS": ms_res,
                    "F": np.nan, "P-Valor": np.nan, "Contrib_pct": (ss_res / ss_total_m * 100.0) if ss_total_m > 0 else 0
                }]),
                pd.DataFrame([{
                    "Factor": "Total", "GL": df_total_m, "SS": ss_total_m, "MS": np.nan,
                    "F": np.nan, "P-Valor": np.nan, "Contrib_pct": 100.0
                }])
            ], ignore_index=True)

        # Optimal condition identification
        opt_sn_levels = {}
        for f in selected_factors:
            means = df_calc.groupby(f)["SN_Ratio"].mean()
            opt_sn_levels[f] = means.idxmax()

        opt_mean_levels = {}
        for f in selected_factors:
            means = df_calc.groupby(f)["Media"].mean()
            if crit_key == "larger":
                opt_mean_levels[f] = means.idxmax()
            elif crit_key == "smaller":
                opt_mean_levels[f] = means.idxmin()
            else:
                opt_mean_levels[f] = means.iloc[(means - gm_m).abs().argsort()[:1]].index[0]

        # Additive model prediction
        grand_mean_sn = df_calc["SN_Ratio"].mean()
        pred_sn = grand_mean_sn + sum(df_calc.groupby(f)["SN_Ratio"].mean()[opt_sn_levels[f]] - grand_mean_sn for f in selected_factors)

        grand_mean_y = df_calc["Media"].mean()
        pred_mean = grand_mean_y + sum(df_calc.groupby(f)["Media"].mean()[opt_mean_levels[f]] - grand_mean_y for f in selected_factors)

        opt_info = {
            "sn_levels": ", ".join([f"{f}={opt_sn_levels[f]}" for f in selected_factors]),
            "mean_levels": ", ".join([f"{f}={opt_mean_levels[f]}" for f in selected_factors]),
            "pred_sn": pred_sn,
            "pred_mean": pred_mean
        }

        # Build short conclusions
        ranks_sn = resp_sn.loc["Clasificar"].astype(int)
        dominant_factor = ranks_sn.idxmin()
        max_delta = resp_sn.loc["Delta", dominant_factor]
        least_factor = ranks_sn.idxmax()
        min_delta = resp_sn.loc["Delta", least_factor]

        conclusions = [
            f"<b>Factor Crítico Dominante:</b> <code>{dominant_factor}</code> ocupa el Rango 1 con un impacto Delta de {max_delta:.3f} dB. Es la variable principal a controlar para garantizar robustez.",
            f"<b>Condición Óptima para Máxima Robustez:</b> Seleccionar {opt_info['sn_levels']}, alcanzando un cociente S/N previsto de {pred_sn:.2f} dB.",
            f"<b>Factor de Menor Impacto / Tolerancia Económica:</b> <code>{least_factor}</code> (Rango {ranks_sn[least_factor]}, Delta {min_delta:.3f} dB). Su nivel puede ajustarse según criterios de menor costo de materia prima sin perjudicar significativamente el rendimiento.",
            "<b>Recomendación de Validación:</b> Se sugiere realizar corridas de confirmación en la condición óptima identificada para verificar que el valor experimental se encuentre dentro del intervalo de confianza del modelo aditivo."
        ]

        # Quick stats cards
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Corridas", len(df_calc))
        col2.metric("Factores Activos", len(selected_factors))
        col3.metric("Réplicas / Corrida", len(selected_reps))
        col4.metric("Media Global (Y)", f"{grand_mean_y:.2f}")
        col5.metric("Media Global S/N", f"{grand_mean_sn:.2f} dB")

        # TABS FOR VISUALIZATION
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Matriz de Corridas",
            "📊 Tablas de Respuesta",
            "📈 Gráficos de Efectos",
            "📐 ANOVA",
            "🏆 Optimización & Conclusiones"
        ])

        with tab1:
            st.subheader("Matriz Experimental con Métricas Calculadas")
            show_cols = (["Corrida"] if "Corrida" in df_calc.columns else []) + selected_factors + selected_reps + ["Media", "Desv_Std", "SN_Ratio"]
            st.dataframe(df_calc[show_cols].style.format({
                "Media": "{:.3f}", "Desv_Std": "{:.3f}", "SN_Ratio": "{:.3f}"
            }), use_container_width=True)

        with tab2:
            st.subheader("Tablas de Respuesta Taguchi")
            c_resp1, c_resp2 = st.columns(2)
            with c_resp1:
                st.markdown("##### 🔊 Relación Señal/Ruido (S/N en dB)")
                st.dataframe(resp_sn.style.format(precision=3), use_container_width=True)
                st.info(f"🏆 **Factor de mayor efecto en S/N:** `{dominant_factor}` (Delta = {max_delta:.3f} dB)")
            with c_resp2:
                st.markdown("##### 📍 Medias de la Respuesta")
                st.dataframe(resp_mean.style.format(precision=3), use_container_width=True)
                dominant_m = resp_mean.loc["Clasificar"].astype(int).idxmin()
                st.info(f"🏆 **Factor de mayor efecto en la Media:** `{dominant_m}` (Delta = {resp_mean.loc['Delta', dominant_m]:.3f})")

        with tab3:
            st.subheader("Gráficas de Efectos Principales")
            fig_sn = plot_main_effects_figure(df_calc, selected_factors, "SN_Ratio", "Relación S/N (dB)", "Media de S/N (dB)")
            st.pyplot(fig_sn)

            fig_mean = plot_main_effects_figure(df_calc, selected_factors, "Media", "Medias de la Respuesta", "Media de Respuesta")
            st.pyplot(fig_mean)

            # Pareto/Contribution plot
            fig_contrib, ax_c = plt.subplots(figsize=(6, 2.8), dpi=130)
            deltas_sn = resp_sn.loc["Delta"].astype(float).sort_values(ascending=True)
            ax_c.barh(deltas_sn.index, deltas_sn.values, color='#3B82F6', edgecolor='#1D4ED8')
            ax_c.set_xlabel("Impacto Delta S/N (dB)", fontsize=9, fontweight='bold')
            ax_c.set_title("Diagrama de Impacto Relativo por Factor (Delta S/N)", fontsize=10.5, fontweight='bold', color='#1E293B')
            ax_c.grid(axis='x', linestyle=':', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig_contrib)

        with tab4:
            st.subheader("Tabla de Análisis de Varianza (ANOVA)")
            st.caption("Cálculo sobre las medias de corrida para evaluar significancia e impacto porcentual de cada factor:")
            st.dataframe(df_anova.style.format({
                "SS": "{:.4f}", "MS": "{:.4f}", "F": "{:.3f}", "P-Valor": "{:.4f}", "Contrib_pct": "{:.2f}%"
            }), use_container_width=True)
            if df_res == 0:
                st.warning("⚠️ Nota técnica: El diseño se encuentra saturado (sin grados de libertad residuales para el error). Para pruebas de hipótesis formales de F y p-valores se recomienda agrupar (pooling) el factor con menor SS o incorporar réplicas completas.")

        with tab5:
            st.subheader("Predicción del Óptimo y Modelo Aditivo")
            c_opt1, c_opt2 = st.columns(2)
            with c_opt1:
                st.markdown("""
                <div class="metric-card">
                    <h4>Condición Óptima para Robustez (S/N)</h4>
                    <p style="font-size: 1.15rem; font-weight: bold; color: #1E3A8A;">""" + opt_info['sn_levels'] + """</p>
                    <p>Valor S/N Predicho: <b>""" + f"{pred_sn:.3f} dB" + """</b></p>
                </div>
                """, unsafe_allow_html=True)
            with c_opt2:
                st.markdown("""
                <div class="metric-card">
                    <h4>Condición Óptima para la Media</h4>
                    <p style="font-size: 1.15rem; font-weight: bold; color: #0F766E;">""" + opt_info['mean_levels'] + """</p>
                    <p>Media Predicha: <b>""" + f"{pred_mean:.3f}" + """</b></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 📝 Análisis Interpretativo y Conclusiones")
            for c in conclusions:
                st.markdown(f"- {c}", unsafe_allow_html=True)

        # PDF REPORT GENERATOR BUTTON
        st.markdown("---")
        st.subheader("📄 Exportación de Resultados")

        # Pre-render figures to buffer for PDF embedding
        buf_sn = io.BytesIO()
        fig_sn.savefig(buf_sn, format='png', bbox_inches='tight')
        buf_sn.seek(0)

        buf_m = io.BytesIO()
        fig_mean.savefig(buf_m, format='png', bbox_inches='tight')
        buf_m.seek(0)

        study_title = f"{data_source} - {selected_period}" if selected_period else data_source
        pdf_bytes = generate_pdf_report(
            study_title, crit_label, df_calc, selected_factors, selected_reps,
            resp_sn, resp_mean, df_anova, opt_info, conclusions, buf_sn, buf_m
        )

        st.download_button(
            label="📥 Descargar Reporte Completo en PDF",
            data=pdf_bytes,
            file_name=f"Reporte_Taguchi_{datetime.date.today().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    else:
        st.info("👈 Por favor selecciona al menos un Factor de Control y una Columna de Respuesta en la barra lateral.")
else:
    st.info("👈 Por favor carga un archivo CSV o selecciona uno de los datasets de ejemplo en la barra lateral para comenzar.")
