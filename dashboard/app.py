import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# ── Make src/ importable ─────────────────────────
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.recommender import get_recommendations

# ── Page config ──────────────────────────────────
st.set_page_config(
    page_title="Mobile Recommendation System",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Segment colors & order ───────────────────────
SEG_COLORS = {
    'Budget':     '#2ecc71',
    'Affordable': '#3498db',
    'Mid-Range':  '#f39c12',
    'Premium':    '#e74c3c'
}
SEG_ORDER = ['Budget', 'Affordable', 'Mid-Range', 'Premium']


# ── Load data ────────────────────────────────────
@st.cache_data
def load_data():
    base     = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'processed')
    df_model = pd.read_csv(
        os.path.join(base, 'model_features.csv'))
    sim_df   = pd.read_csv(
        os.path.join(base, 'similarity_matrix.csv'),
        index_col=0)
    return df_model, sim_df


# ── Guard — pipeline must run first ─────────────
check_path = os.path.join(
    os.path.dirname(__file__), '..',
    'data', 'processed', 'model_features.csv')

if not os.path.exists(check_path):
    st.error("⚠️ Processed data not found.")
    st.info("Run `python main.py` first, then relaunch the dashboard.")
    st.stop()

df_model, sim_df = load_data()


# ════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════
with st.sidebar:
    st.title("📱 Mobile Recommender")
    st.markdown("---")

    st.markdown("### Dataset Info")
    st.metric("Unique Models", df_model.shape[0])
    st.metric("Brands",        df_model['brand'].nunique())
    st.metric("Segments",      df_model['segment'].nunique())

    st.markdown("---")
    st.markdown("### Segment Legend")
    for seg in SEG_ORDER:
        color = SEG_COLORS[seg]
        st.markdown(
            f'<span style="background:{color};color:white;'
            f'padding:3px 12px;border-radius:12px;'
            f'font-size:13px;font-weight:600;">{seg}</span>',
            unsafe_allow_html=True)
        st.write("")

    st.markdown("---")
    st.markdown("### Price Ranges")
    for seg in SEG_ORDER:
        seg_df = df_model[df_model['segment'] == seg]
        if len(seg_df) > 0:
            lo = seg_df['price_usd'].min()
            hi = seg_df['price_usd'].max()
            st.markdown(f"**{seg}**: ${lo:.0f} – ${hi:.0f}")


# ════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════
st.title("📱 Mobile Product Recommendation System")
st.markdown(
    "Segment-aware recommendations powered by "
    "**Content-Based Filtering** using Cosine Similarity.")
st.divider()


# ════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊 Segment Overview",
    "🔍 Find Similar Phones",
    "🏷️ Browse by Brand"
])


# ════════════════════════════════════════════════
# TAB 1 — Segment Overview
# ════════════════════════════════════════════════
with tab1:
    st.subheader("Phone Market Segments")
    st.markdown(
        "Phones grouped into **4 price-based segments** "
        "using KMeans clustering on `price_usd`.")

    # KPI row
    kpi_cols = st.columns(4)
    for i, seg in enumerate(SEG_ORDER):
        seg_df = df_model[df_model['segment'] == seg]
        kpi_cols[i].metric(
            label=f"💰 {seg}",
            value=f"{len(seg_df)} phones",
            delta=f"Avg ${seg_df['price_usd'].mean():.0f}"
        )

    st.divider()

    # Charts row 1
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Phones per Segment**")
        seg_counts = (df_model['segment']
                      .value_counts()
                      .reindex(SEG_ORDER)
                      .reset_index())
        seg_counts.columns = ['Segment', 'Count']
        fig_bar = px.bar(
            seg_counts, x='Segment', y='Count',
            color='Segment',
            color_discrete_map=SEG_COLORS,
            text='Count'
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20),
            xaxis_title="", yaxis_title="Number of Phones"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.markdown("**Price Distribution by Segment**")
        fig_box = px.box(
            df_model, x='segment', y='price_usd',
            color='segment',
            color_discrete_map=SEG_COLORS,
            category_orders={'segment': SEG_ORDER}
        )
        fig_box.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20),
            xaxis_title="", yaxis_title="Price (USD)"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        #average price for all brands
        st.markdown("**Average Price per Segment**")
        avg_price = df_model.groupby('segment')['price_usd'].mean().reindex(SEG_ORDER).reset_index()
        avg_price.columns = ['Segment', 'Avg Price']

        fig = px.bar(avg_price, x='Segment', y='Avg Price',
             color='Segment',
             color_discrete_map=SEG_COLORS,
             text='Avg Price')
        fig.update_traces(
            texttemplate='$%{text:.0f}',
            textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    
    # Brand vs segment heatmap
    st.markdown("**Brand Distribution Across Segments**")
    ct = pd.crosstab(df_model['brand'], df_model['segment'])
    # reorder columns
    ct = ct.reindex(
        columns=[c for c in SEG_ORDER if c in ct.columns])

    fig_heat = px.imshow(
        ct, text_auto=True,
        color_continuous_scale='Blues',
        aspect='auto'
    )
    fig_heat.update_layout(
        margin=dict(t=20, b=20),
        xaxis_title="Segment",
        yaxis_title="Brand"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Full model table
    st.markdown("**All Phone Models**")
    seg_filter = st.multiselect(
        "Filter by segment:",
        options=SEG_ORDER,
        default=SEG_ORDER
    )
    show_cols = ['brand', 'model', 'segment', 'price_usd',
                 'camera_rating', 'battery_life_rating',
                 'performance_rating', 'rating']
    filtered  = (df_model[df_model['segment'].isin(seg_filter)]
                 [show_cols]
                 .sort_values('price_usd')
                 .rename(columns={
                     'brand':               'Brand',
                     'model':               'Model',
                     'segment':             'Segment',
                     'price_usd':           'Price (USD)',
                     'camera_rating':       'Camera',
                     'battery_life_rating': 'Battery',
                     'performance_rating':  'Performance',
                     'rating':              'Avg Rating'
                 }))
    st.dataframe(filtered, use_container_width=True,
                 hide_index=True)


# ════════════════════════════════════════════════
# TAB 2 — Find Similar Phones
# ════════════════════════════════════════════════
with tab2:
    st.subheader("Find Similar Phones")
    st.markdown(
        "Select a phone to get **top 5 similar phones** "
        "based on specs and price tier.")

    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        selected_model = st.selectbox(
            "Select a phone:",
            options=sorted(df_model['model'].tolist())
        )
    with c2:
        brand_options = ['All'] + sorted(
            df_model['brand'].unique().tolist())
        brand_filter  = st.selectbox(
            "Filter recommendations by brand:", brand_options)
    with c3:
        same_segment  = True
        same_segment  = st.checkbox(
            "Same segment only", value=True,
            help="Only recommend phones in the same price tier")

    st.divider()

    # Selected phone details
    sel       = df_model[df_model['model'] == selected_model].iloc[0]
    seg       = sel['segment']
    seg_color = SEG_COLORS.get(seg, '#888')

    st.markdown(f"### {sel['brand']} — {selected_model}")
    st.markdown(
        f'<span style="background:{seg_color};color:white;'
        f'padding:4px 14px;border-radius:12px;'
        f'font-size:13px;font-weight:600;">{seg}</span>',
        unsafe_allow_html=True)
    st.write("")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Price",       f"${sel['price_usd']:.0f}")
    m2.metric("Camera",      f"{sel['camera_rating']:.2f} ⭐")
    m3.metric("Battery",     f"{sel['battery_life_rating']:.2f} ⭐")
    m4.metric("Performance", f"{sel['performance_rating']:.2f} ⭐")
    m5.metric("Avg Rating",  f"{sel['rating']:.2f} ⭐")

    st.divider()

    # Get recommendations
    recs = get_recommendations(
        model_name        = selected_model,
        df_model          = df_model,
        sim_df            = sim_df,
        top_n             = 5,
        same_segment_only = same_segment
    )

    # Apply brand filter
    if recs is not None and brand_filter != 'All':
        recs = recs[recs['brand'] == brand_filter]

    st.markdown("#### Top Recommendations")

    if recs is None or len(recs) == 0:
        st.warning(
            "No recommendations found with current filters. "
            "Try turning off 'Same segment only' or changing the brand filter.")
    else:
        # Recommendation cards
        for _, row in recs.iterrows():
            rc      = SEG_COLORS.get(row['segment'], '#888')
            sim_pct = int(row['similarity_score'] * 100)
            dot_col = ('#2ecc71' if sim_pct >= 80
                       else '#f39c12' if sim_pct >= 60
                       else '#e74c3c')

            ca, cb, cc = st.columns([3, 5, 2])

            with ca:
                st.markdown(f"**{row['brand']}**  \n{row['model']}")
                st.markdown(
                    f'<span style="background:{rc};color:white;'
                    f'padding:2px 8px;border-radius:10px;'
                    f'font-size:11px;">{row["segment"]}</span>',
                    unsafe_allow_html=True)

            with cb:
                st.markdown(
                    f"💰 **${row['price_usd']:.0f}**&nbsp;&nbsp;"
                    f"📷 {row['camera_rating']:.2f}&nbsp;&nbsp;"
                    f"🔋 {row['battery_life_rating']:.2f}&nbsp;&nbsp;"
                    f"⚡ {row['performance_rating']:.2f}")

            with cc:
                st.markdown(
                    f'<div style="text-align:right;">'
                    f'<span style="font-size:22px;font-weight:700;'
                    f'color:{dot_col};">{sim_pct}%</span><br>'
                    f'<span style="font-size:11px;color:#888;">match</span>'
                    f'</div>',
                    unsafe_allow_html=True)

            st.markdown("---")

        # Feature comparison chart
        st.markdown("#### Feature Comparison")
        compare_feats  = ['battery_life_rating', 'camera_rating',
                          'performance_rating', 'design_rating',
                          'display_rating']
        compare_labels = ['Battery', 'Camera', 'Performance',
                          'Design', 'Display']

        sel_row          = sel[compare_feats].to_frame().T.copy()
        sel_row['model'] = f"{selected_model} ★"

        rec_rows = df_model[
            df_model['model'].isin(
                recs['model'].tolist())][
            compare_feats + ['model']].copy()

        compare_df = pd.concat(
            [sel_row, rec_rows], ignore_index=True)

        melted = compare_df.melt(
            id_vars='model',
            value_vars=compare_feats,
            var_name='Feature',
            value_name='Rating'
        )
        melted['Feature'] = melted['Feature'].map(
            dict(zip(compare_feats, compare_labels)))

        fig_cmp = px.bar(
            melted, x='Feature', y='Rating',
            color='model', barmode='group',
            range_y=[0, 5],
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_cmp.update_layout(
            margin=dict(t=20, b=20),
            xaxis_title="",
            yaxis_title="Rating (out of 5)",
            legend=dict(orientation='h', y=-0.25, title_text='')
        )
        st.plotly_chart(fig_cmp, use_container_width=True)


# ════════════════════════════════════════════════
# TAB 3 — Browse by Brand
# ════════════════════════════════════════════════
with tab3:
    st.subheader("Browse by Brand")
    st.markdown(
        "Explore all phones from a brand "
        "and compare their specs.")

    selected_brand = st.selectbox(
        "Select a brand:",
        options=sorted(df_model['brand'].unique().tolist())
    )

    brand_df = (df_model[df_model['brand'] == selected_brand]
                .sort_values('price_usd'))

    if len(brand_df) == 0:
        st.warning(f"No phones found for '{selected_brand}'.")
    else:
        # Brand KPIs
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Models",     len(brand_df))
        b2.metric("Avg Price",  f"${brand_df['price_usd'].mean():.0f}")
        b3.metric("Avg Rating", f"{brand_df['rating'].mean():.2f} ⭐")
        b4.metric("Avg Camera", f"{brand_df['camera_rating'].mean():.2f} ⭐")

        st.divider()

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown(f"**{selected_brand} — Price by Model**")
            fig_bp = px.bar(
                brand_df, x='model', y='price_usd',
                color='segment',
                color_discrete_map=SEG_COLORS,
                text='price_usd'
            )
            fig_bp.update_traces(
                texttemplate='$%{text:.0f}',
                textposition='outside')
            fig_bp.update_layout(
                showlegend=True,
                margin=dict(t=20, b=20),
                xaxis_title="", yaxis_title="Price (USD)",
                xaxis_tickangle=-30
            )
            st.plotly_chart(fig_bp, use_container_width=True)

        with col_r:
            st.markdown(f"**{selected_brand} — Feature Radar**")
            feat_cols   = ['battery_life_rating', 'camera_rating',
                           'performance_rating', 'design_rating',
                           'display_rating']
            feat_labels = ['Battery', 'Camera', 'Performance',
                           'Design', 'Display']

            fig_br    = go.Figure()
            clr_list  = px.colors.qualitative.Set2

            for i, (_, row) in enumerate(brand_df.iterrows()):
                vals = row[feat_cols].tolist()
                fig_br.add_trace(go.Scatterpolar(
                    r          = vals + [vals[0]],
                    theta      = feat_labels + [feat_labels[0]],
                    fill       = 'toself',
                    name       = row['model'],
                    line_color = clr_list[i % len(clr_list)],
                    opacity    = 0.7
                ))

            fig_br.update_layout(
                polar  = dict(radialaxis=dict(
                    visible=True, range=[0, 5])),
                margin = dict(t=40, b=40)
            )
            st.plotly_chart(fig_br, use_container_width=True)

        # Phone list
        st.markdown(f"**All {selected_brand} Models**")
        for _, row in brand_df.iterrows():
            rc = SEG_COLORS.get(row['segment'], '#888')
            ca, cb = st.columns([3, 5])
            with ca:
                st.markdown(f"**{row['model']}**")
                st.markdown(
                    f'<span style="background:{rc};color:white;'
                    f'padding:2px 10px;border-radius:10px;'
                    f'font-size:11px;">{row["segment"]}</span>'
                    f'&nbsp; 💰 **${row["price_usd"]:.0f}**',
                    unsafe_allow_html=True)
            with cb:
                st.markdown(
                    f"📷 {row['camera_rating']:.2f} &nbsp;"
                    f"🔋 {row['battery_life_rating']:.2f} &nbsp;"
                    f"⚡ {row['performance_rating']:.2f} &nbsp;"
                    f"⭐ {row['rating']:.2f}")
            st.markdown("---")

