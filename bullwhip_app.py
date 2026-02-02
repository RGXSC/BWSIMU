import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="Simulateur Effet Bullwhip", layout="wide")

st.markdown("""
<div style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 30px; border-radius: 15px; margin-bottom: 30px;'>
    <h1 style='color: white; margin: 0;'>🔄 Simulateur Effet Bullwhip</h1>
    <p style='color: #f0f0f0; margin: 10px 0 0 0;'>Beer Game Supply Chain Dynamics - 24 Weeks Simulation</p>
</div>
""", unsafe_allow_html=True)

class BullwhipSimulator:
    def __init__(self):
        self.W = 24
        self.MAX_FACTORY_DEMAND = 1000
        self.DEMAND_BASE = [0,100,200,300,350,380,400,350,280,220,
                           200,200,200,200,200,200,200,200,200,200,
                           200,200,200,200,200]
    
    def simulate(self, bias, panic, lt_usine, stock_init_mag, stock_init_ent, stock_init_usi, pv=125, pc=30):
        lt1, lt2, lt3 = 1, 1, lt_usine
        
        if pc >= pv:
            return {'data': pd.DataFrame(), 'kpis': {}, 'valid': False, 'params': {}}
        
        demand_np = np.array(self.DEMAND_BASE[:self.W + 1], dtype=float)
        demand_np[0] = 0
        
        s_mag = stock_init_mag
        s_ent = stock_init_ent
        s_usi = stock_init_usi
        
        t_mag, t_ent, t_prd = {}, {}, {}
        data = []
        total_cost = 0
        total_ca = 0
        
        for t in range(self.W + 1):
            dem = int(demand_np[t])
            
            # Réceptions
            s_mag += t_mag.get(t, 0)
            s_ent += t_ent.get(t, 0)
            s_usi += t_prd.get(t, 0)
            
            # Ventes
            vnt = min(dem, s_mag)
            perd = dem - vnt
            s_mag -= vnt
            
            total_ca += vnt * pv
            
            # COMMANDE MAGASIN
            cmd_mag = 0
            if t > 0:
                fc = demand_np[max(1, t-2):t+1].mean()
                target = fc * (lt1 + 1)
                
                if s_mag < target:
                    stock_gap = target - s_mag
                    cmd_mag = dem + (stock_gap * 0.3)
                else:
                    cmd_mag = dem * 0.3
                
                if panic > 1 and perd > dem * 0.1:
                    cmd_mag *= panic
                
                cmd_mag = min(cmd_mag, fc * 30)
                if t > self.W * 0.85: 
                    cmd_mag *= 0.1
            
            # COMMANDE ENTREPÔT
            cmd_ent = 0
            if t > 1:
                past_mag = [d['cmd_mag'] for d in data[max(0, t-2):t]]
                fc_ent = np.mean(past_mag) if past_mag else 0
                target_ent = fc_ent * (lt2 + 1)
                
                if s_ent < target_ent:
                    stock_gap = target_ent - s_ent
                    anchor_ent = past_mag[-1] if past_mag else 0
                    cmd_ent = anchor_ent + (stock_gap * 0.3)
                else:
                    cmd_ent = fc_ent * 0.3
                
                if panic > 1 and len(past_mag) >= 2:
                    if past_mag[-1] > np.mean(past_mag[:-1]) * 1.3:
                        cmd_ent *= (1 + (panic - 1) * 0.7)
                
                cmd_ent = min(cmd_ent, fc_ent * 25)
                if t > self.W * 0.85: 
                    cmd_ent *= 0.1
            
            # COMMANDE USINE
            cmd_usi = 0
            if t > 1:
                past_ent = [d['cmd_ent'] for d in data[max(0, t-2):t]]
                fc_usi = np.mean(past_ent) if past_ent else 0
                
                safety_stock = fc_usi * lt3
                target_stock = safety_stock + fc_usi
                
                if s_usi < target_stock:
                    needed = target_stock - s_usi
                    current_demand = past_ent[-1] if past_ent else 0
                    cmd_usi = max(current_demand, needed * 0.5)
                else:
                    cmd_usi = fc_usi * 0.2
                
                if panic > 1 and len(past_ent) >= 2:
                    if past_ent[-1] > np.mean(past_ent[:-1]) * 1.3 and s_usi < target_stock:
                        cmd_usi *= (1 + (panic - 1) * 0.5)
                
                cmd_usi = min(cmd_usi, fc_usi * 20)
                cmd_usi = min(cmd_usi, self.MAX_FACTORY_DEMAND)
                
                if t > self.W * 0.85: 
                    cmd_usi *= 0.05
            
            # Flux physiques
            shp_m = min(int(np.round(cmd_mag)), s_ent)
            s_ent -= shp_m
            t_mag[t + lt1] = t_mag.get(t + lt1, 0) + shp_m
            
            shp_e = min(int(np.round(cmd_ent)), s_usi)
            s_usi -= shp_e
            t_ent[t + lt2] = t_ent.get(t + lt2, 0) + shp_e
            
            prd = int(np.round(cmd_usi))
            t_prd[t + lt3] = t_prd.get(t + lt3, 0) + prd
            total_cost += prd * pc
            
            data.append({
                't': t, 'dem': int(dem), 
                's_mag': s_mag, 's_ent': s_ent, 's_usi': s_usi,
                'cmd_mag': int(np.round(cmd_mag)), 
                'cmd_ent': int(np.round(cmd_ent)), 
                'cmd_usi': int(np.round(cmd_usi)),
                'vnt': vnt, 'perd': perd,
                'svc': 100 * vnt / dem if dem > 0 else 100
            })
        
        df = pd.DataFrame(data)
        tot_d = demand_np[1:].sum()
        tot_v = df['vnt'].sum()
        tot_p = df['perd'].sum()
        stock_final = df.iloc[-1][['s_mag', 's_ent', 's_usi']].sum()
        
        svc = 100 * tot_v / tot_d if tot_d > 0 else 100
        ventes_perdues_k = tot_p * pv / 1000
        marge_k = (total_ca - total_cost - stock_final * pc) / 1000
        ca_k = total_ca / 1000
        stock_restant_euros = stock_final * pc
        
        d_std = demand_np[10:].std()
        f_std = df.iloc[10:]['cmd_usi'].std()
        bw = f_std / d_std if d_std > 10 else 1.0
        
        max_cmd_usi = df['cmd_usi'].max()
        limit_respected = max_cmd_usi <= self.MAX_FACTORY_DEMAND
        
        valid = (
            (df[['s_mag', 's_ent', 's_usi']] >= 0).all().all() and
            (df[['cmd_mag', 'cmd_ent', 'cmd_usi']] >= 0).all().all() and
            (df['vnt'] <= df['dem']).all() and 
            0 <= svc <= 100 and 0 < bw < 1000 and 
            len(df) == self.W + 1 and
            limit_respected
        )
        
        return {
            'data': df,
            'kpis': {
                'service_level': svc,
                'ventes_perdues': ventes_perdues_k,
                'marge': marge_k,
                'ca': ca_k,
                'bullwhip': bw,
                'stock_final': int(stock_final),
                'stock_restant_euros': stock_restant_euros,
                'max_cmd_usine': int(max_cmd_usi)
            },
            'valid': valid,
            'params': {
                'bias': bias, 
                'panic': panic, 
                'lt_usine': lt_usine,
                'stock_init_mag': stock_init_mag,
                'stock_init_ent': stock_init_ent,
                'stock_init_usi': stock_init_usi,
                'total_lt': lt1 + lt2 + lt3
            }
        }

# SIDEBAR
st.sidebar.header("⚙️ Paramètres de Simulation")

bias = st.sidebar.select_slider("Bias Prévision Initial", 
                                 options=[0.5, 1.0, 1.5, 2.0], value=1.0)
panic = st.sidebar.select_slider("Multiplicateur Panique", 
                                  options=[1, 2, 3], value=2)
lt_usine = st.sidebar.select_slider("Délai Production (sem)", 
                                     options=[2, 6, 10], value=6)

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Stocks Initiaux")
stock_init_mag = st.sidebar.slider("Stock Initial Magasin", 100, 1000, 500, 100)
stock_init_ent = st.sidebar.slider("Stock Initial Entrepôt", 100, 1000, 500, 100)
stock_init_usi = st.sidebar.slider("Stock Initial Usine", 100, 1000, 500, 100)

st.sidebar.markdown("---")
pv = st.sidebar.slider("Prix Vente (€)", 50, 250, 125, 5)
pc = st.sidebar.slider("Coût Production (€)", 10, 150, 30, 5)

st.sidebar.info(f"🏭 **Limite usine: 1000 pièces/semaine**")

if st.sidebar.button("▶ LANCER SIMULATION (24 semaines)", type="primary"):
    sim = BullwhipSimulator()
    result = sim.simulate(bias, panic, lt_usine, stock_init_mag, stock_init_ent, stock_init_usi, pv, pc)
    st.session_state['result'] = result

# ✅ TESTS COMPLETS : 4×3×3×3×3×3 = 972 combinaisons
if st.sidebar.button("🔬 LANCER TESTS COMPLETS (972 combinaisons)", type="secondary"):
    with st.spinner('Tests en cours... (972 combinaisons - peut prendre 1-2 minutes)'):
        sim = BullwhipSimulator()
        results = []
        progress = st.progress(0)
        count = 0
        total = 4 * 3 * 3 * 3 * 3 * 3  # 972
        
        for b in [0.5, 1.0, 1.5, 2.0]:
            for p in [1, 2, 3]:
                for lt in [2, 6, 10]:
                    # ✅ TOUTES LES COMBINAISONS DE STOCKS
                    for s_mag in [100, 500, 1000]:
                        for s_ent in [100, 500, 1000]:
                            for s_usi in [100, 500, 1000]:
                                r = sim.simulate(b, p, lt, s_mag, s_ent, s_usi, pv, pc)
                                if r['valid']:
                                    results.append({
                                        'Bias': r['params']['bias'],
                                        'Panic': r['params']['panic'],
                                        'LT Usine': r['params']['lt_usine'],
                                        'Stock Magasin': r['params']['stock_init_mag'],
                                        'Stock Entrepôt': r['params']['stock_init_ent'],
                                        'Stock Usine': r['params']['stock_init_usi'],
                                        'CA (K€)': round(r['kpis']['ca'], 1),
                                        'Marge Réelle (K€)': round(r['kpis']['marge'], 1),
                                        'Ventes Perdues (K€)': round(r['kpis']['ventes_perdues'], 1),
                                        'Stock Restant (€)': int(r['kpis']['stock_restant_euros']),
                                        'Bullwhip': round(r['kpis']['bullwhip'], 2),
                                        'Service (%)': round(r['kpis']['service_level'], 1)
                                    })
                                count += 1
                                progress.progress(count / total)
        
        st.session_state['test_results'] = pd.DataFrame(results)
        progress.empty()
        st.success(f"✅ {len(results)}/{total} tests valides!")

# AFFICHAGE SIMULATION
if 'result' in st.session_state and st.session_state['result'] is not None:
    r = st.session_state['result']
    
    if not r['data'].empty:
        df, kpis = r['data'], r['kpis']
        
        st.markdown("---")
        st.subheader("📊 Résultats de la Simulation Actuelle")
        
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        col1.metric("SERVICE LEVEL", f"{kpis['service_level']:.1f}%")
        col2.metric("VENTES PERDUES", f"${kpis['ventes_perdues']:.1f}K")
        col3.metric("MARGE RÉELLE", f"${kpis['marge']:.1f}K")
        col4.metric("BULLWHIP RATIO", f"{kpis['bullwhip']:.2f}x")
        col5.metric("STOCK FINAL", f"{kpis['stock_final']:,}")
        col6.metric("LT TOTAL", f"{r['params']['total_lt']} sem")
        col7.metric("MAX CMD USINE", f"{kpis['max_cmd_usine']}", 
                   delta="✅ OK" if kpis['max_cmd_usine'] <= 1000 else "⚠️ LIMITE",
                   delta_color="normal" if kpis['max_cmd_usine'] <= 1000 else "inverse")
        
        col_ca1, col_ca2, col_ca3 = st.columns(3)
        col_ca1.metric("💰 CHIFFRE D'AFFAIRES", f"${kpis['ca']:.1f}K")
        col_ca2.metric("📦 STOCK RESTANT (€)", f"{kpis['stock_restant_euros']:,.0f} €")
        col_ca3.metric("📊 TAUX DE MARGE", f"{(kpis['marge']/kpis['ca']*100) if kpis['ca'] > 0 else 0:.1f}%")
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df['t'], y=df['dem'], mode='lines', 
                                  name='Demande Réelle', line=dict(color='black', width=3)))
        fig1.add_trace(go.Scatter(x=df['t'], y=df['cmd_mag'], mode='lines', 
                                  name='Commandes Magasin', line=dict(color='#007bff', width=2)))
        fig1.add_trace(go.Scatter(x=df['t'], y=df['cmd_ent'], mode='lines', 
                                  name='Commandes Entrepôt', line=dict(color='#ffc107', width=2)))
        fig1.add_trace(go.Scatter(x=df['t'], y=df['cmd_usi'], mode='lines', 
                                  name='Commandes Usine', line=dict(color='#dc3545', width=2)))
        
        fig1.add_hline(y=1000, line_dash="dash", line_color="red", 
                      annotation_text="Limite Usine (1000)", 
                      annotation_position="right")
        
        fig1.update_layout(title="📈 Amplification de la Demande (Effet Bullwhip)",
                          xaxis_title="Semaine", yaxis_title="Quantité", height=450)
        st.plotly_chart(fig1, use_container_width=True)
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df['t'], y=df['s_mag'], mode='lines', 
                                      name='Magasin', fill='tozeroy', line=dict(color='#007bff')))
            fig2.add_trace(go.Scatter(x=df['t'], y=df['s_ent'], mode='lines', 
                                      name='Entrepôt', fill='tozeroy', line=dict(color='#ffc107')))
            fig2.add_trace(go.Scatter(x=df['t'], y=df['s_usi'], mode='lines', 
                                      name='Usine', fill='tozeroy', line=dict(color='#28a745')))
            fig2.update_layout(title="📦 Niveaux de Stock", 
                              xaxis_title="Semaine", yaxis_title="Stock", height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col_g2:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df['t'], y=df['svc'], mode='lines', 
                                      name='Service %', fill='tozeroy', line=dict(color='#28a745')))
            fig3.update_layout(title="✅ Taux de Service", 
                              xaxis_title="Semaine", yaxis_title="Service (%)", 
                              yaxis_range=[0, 100], height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔬 Validation des Calculs")
        if r['valid']:
            st.success(f"✅ Simulation valide - Bullwhip: {kpis['bullwhip']:.2f}x | Service: {kpis['service_level']:.1f}% | CA: ${kpis['ca']:.1f}K")
        else:
            st.error("❌ Problèmes détectés")

# ✅ AFFICHAGE TESTS AVEC TOUTES LES COMBINAISONS DE STOCKS
if 'test_results' in st.session_state and st.session_state['test_results'] is not None:
    df_test = st.session_state['test_results']
    
    if not df_test.empty:
        st.markdown("---")
        st.subheader(f"🧪 Résultats des Tests Complets ({len(df_test)} combinaisons)")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CA Moyen", f"${df_test['CA (K€)'].mean():.1f}K")
        col2.metric("Marge Moyenne", f"${df_test['Marge Réelle (K€)'].mean():.1f}K")
        col3.metric("Bullwhip Moyen", f"{df_test['Bullwhip'].mean():.2f}x")
        col4.metric("Service Moyen", f"{df_test['Service (%)'].mean():.1f}%")
        
        st.markdown("### 📋 Tableau des Résultats (triable par colonne)")
        
        st.dataframe(
            df_test,
            use_container_width=True,
            height=400
        )
        
        st.markdown("### 📥 Télécharger les Résultats")
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_test.to_excel(writer, index=False, sheet_name='Résultats')
        
        excel_data = output.getvalue()
        
        st.download_button(
            label="📊 Télécharger en Excel",
            data=excel_data,
            file_name="bullwhip_simulation_results_972.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        col_top, col_bottom = st.columns(2)
        
        with col_top:
            st.markdown("### 🏆 Top 10 Meilleure Marge")
            top10 = df_test.nlargest(10, 'Marge Réelle (K€)')[['Bias', 'Panic', 'LT Usine', 'Stock Magasin', 'Stock Entrepôt', 'Stock Usine', 'Marge Réelle (K€)', 'CA (K€)']]
            st.dataframe(top10, use_container_width=True)
        
        with col_bottom:
            st.markdown("### ⚠️ Top 10 Pire Marge")
            bottom10 = df_test.nsmallest(10, 'Marge Réelle (K€)')[['Bias', 'Panic', 'LT Usine', 'Stock Magasin', 'Stock Entrepôt', 'Stock Usine', 'Marge Réelle (K€)', 'Ventes Perdues (K€)']]
            st.dataframe(bottom10, use_container_width=True)
