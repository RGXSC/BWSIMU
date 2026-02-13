import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import time
from io import BytesIO

st.set_page_config(page_title="Simulateur Bullwhip - Mode Pas-à-Pas", layout="wide")

st.markdown("""
<div style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 30px; border-radius: 15px; margin-bottom: 30px;'>
    <h1 style='color: white; margin: 0;'>🔄 Simulateur Bullwhip - Mode Pas-à-Pas</h1>
    <p style='color: #f0f0f0; margin: 10px 0 0 0;'>Visualisation détaillée étape par étape de la Supply Chain</p>
</div>
""", unsafe_allow_html=True)

class BullwhipSimulator:
    def __init__(self, spikiness=1, spike_duration=9):
        self.W = 52
        
        # Calculer le pic de demande selon spikiness
        if spikiness == 1:
            spike_value = 500
        elif spikiness == 2:
            spike_value = 1000
        elif spikiness == 3:
            spike_value = 1500
        else:  # spikiness == 4
            spike_value = 2000
        
        # Capacité max fournisseur = 4x le pic de demande
        self.MAX_WEEKLY_CAPACITY = spike_value * 4
        
        # Demande avec spikiness paramétrable
        self.DEMAND_BASE = [0]
        for i in range(3):
            self.DEMAND_BASE.append(100)
        
        # Pic de demande avec durée paramétrable
        for i in range(spike_duration):
            self.DEMAND_BASE.append(spike_value)
        
        # Reste des semaines : demande normale
        while len(self.DEMAND_BASE) <= self.W:
            self.DEMAND_BASE.append(100)
        
        self.DEMAND_BASE = np.array(self.DEMAND_BASE, dtype=float)
        self.MAX_DEMAND = np.max(self.DEMAND_BASE)

 
    
    def simulate_full(self, panic, lt_usine, lt_fournisseur, 
                      freq_retail, freq_supply,
                      stock_mag, stock_ent, stock_usi, stock_four, 
                      pv=200, pc=30):
        
        demand = np.array(self.DEMAND_BASE[:self.W + 1], dtype=float)
        
        lt = {'mag': 1, 'ent': 1, 'usi': lt_usine, 'four': lt_fournisseur}
        
        s = {'mag': int(stock_mag), 'ent': int(stock_ent), 
             'usi': int(stock_usi), 'four': int(stock_four)}
        
        backlog = {'mag': 0.0, 'ent': 0.0, 'usi': 0.0, 'four': 0.0}
        pipe = {'mag': {}, 'ent': {}, 'usi': {}, 'four': {}}
        outstanding = {'mag': 0.0, 'ent': 0.0, 'usi': 0.0, 'four': 0.0}
        
        ORDER_CAP = self.MAX_WEEKLY_CAPACITY
        
        initial_fc = 100
        forecast = {'mag': initial_fc, 'ent': initial_fc, 'usi': initial_fc, 'four': initial_fc}
        history = {'mag': [initial_fc, initial_fc], 'ent': [initial_fc, initial_fc],
                   'usi': [initial_fc, initial_fc], 'four': [initial_fc, initial_fc]}
        
        all_weeks_data = []
        
        for t in range(self.W + 1):
            week_log = {'t': t, 'events': []}
            dem = int(demand[t])
            
            # 1. RÉCEPTIONS
            arrivals = {'mag': 0, 'ent': 0, 'usi': 0, 'four': 0}
            for level in ['mag', 'ent', 'usi', 'four']:
                arrival = int(pipe[level].get(t, 0))
                if arrival > 0:
                    s[level] += arrival
                    arrivals[level] = arrival
                    week_log['events'].append(f"📥 {level.upper()}: Réception de {arrival} pcs")
                    if t in pipe[level]:
                        del pipe[level][t]
            
            week_log['events'].append(f"📋 MAGASIN: Demande client = {dem} pcs")
            
            # 2. VENTES
            available = max(0, s['mag'])
            total_demand = dem + backlog['mag']
            sales = min(total_demand, available)
            s['mag'] -= sales
            
            if sales < total_demand:
                lost = total_demand - sales
                backlog['mag'] = 0
                week_log['events'].append(f"❌ MAGASIN: Ventes perdues = {int(lost)} pcs")
            else:
                backlog['mag'] = 0
            
            if sales > 0:
                week_log['events'].append(f"✅ MAGASIN: Ventes = {int(sales)} pcs")
            
            # 3. LIVRAISONS
            shipments = {'mag': 0, 'ent': 0, 'usi': 0, 'four': 0}
            
            # ENTREPÔT → MAGASIN
            if outstanding['mag'] > 1:
                available_stock = max(0, s['ent'])
                if available_stock > 0:
                    shipment = min(int(outstanding['mag']), available_stock, self.MAX_WEEKLY_CAPACITY)
                    if shipment > 0:
                        s['ent'] -= shipment
                        arrival_week = t + lt['mag']
                        pipe['mag'][arrival_week] = int(pipe['mag'].get(arrival_week, 0) + shipment)
                        outstanding['mag'] -= shipment
                        shipments['mag'] = shipment
                        week_log['events'].append(f"🚚 ENTREPÔT → MAGASIN: Expédition de {shipment} pcs (arrive sem {arrival_week})")
            
            # USINE → ENTREPÔT
            if outstanding['ent'] > 1:
                available_stock = max(0, s['usi'])
                if available_stock > 0:
                    shipment = min(int(outstanding['ent']), available_stock, self.MAX_WEEKLY_CAPACITY)
                    if shipment > 0:
                        s['usi'] -= shipment
                        arrival_week = t + lt['ent']
                        pipe['ent'][arrival_week] = int(pipe['ent'].get(arrival_week, 0) + shipment)
                        outstanding['ent'] -= shipment
                        shipments['ent'] = shipment
                        week_log['events'].append(f"🚚 USINE → ENTREPÔT: Expédition de {shipment} pcs (arrive sem {arrival_week})")
            
            # FOURNISSEUR → USINE
            if outstanding['usi'] > 1:
                available_stock = max(0, s['four'])
                if available_stock > 0:
                    shipment = min(int(outstanding['usi']), available_stock, self.MAX_WEEKLY_CAPACITY)
                    if shipment > 0:
                        s['four'] -= shipment
                        arrival_week = t + lt['usi']
                        pipe['usi'][arrival_week] = int(pipe['usi'].get(arrival_week, 0) + shipment)
                        outstanding['usi'] -= shipment
                        shipments['usi'] = shipment
                        week_log['events'].append(f"🚚 FOURNISSEUR → USINE: Expédition de {shipment} pcs (arrive sem {arrival_week})")
            
            # PRODUCTION
            if outstanding['four'] > 1:
                production = min(int(outstanding['four']), self.MAX_WEEKLY_CAPACITY)
                if production > 0:
                    arrival_week = t + lt['four']
                    pipe['four'][arrival_week] = int(pipe['four'].get(arrival_week, 0) + production)
                    outstanding['four'] -= production
                    shipments['four'] = production
                    week_log['events'].append(f"🏭 FOURNISSEUR: Production de {production} pcs (arrive sem {arrival_week})")
            
            # 4. FORECAST
            if t > 0:
                history['mag'].append(dem)
                history['mag'] = history['mag'][-2:]
                h0 = history['mag'][0] if len(history['mag']) > 0 else 0
                h1 = history['mag'][1] if len(history['mag']) > 1 else h0
                
                if panic == 1:
                    forecast['mag'] = h0 * 0.5 + h1 * 0.5
                elif panic == 2:
                    forecast['mag'] = h0 * 0.25 + h1 * 0.75
                else:
                    forecast['mag'] = h0 * 0.05 + h1 * 0.95
                
                week_log['events'].append(f"📊 MAGASIN: Nouveau forecast = {int(forecast['mag'])} pcs")
            
            # 5. COMMANDES
            cmd = {'mag': 0, 'ent': 0, 'usi': 0, 'four': 0}
            target_level = {'mag': 0, 'ent': 0, 'usi': 0, 'four': 0}
            
            # MAGASIN → ENTREPÔT
            if t % freq_retail == 0 and t > 0:
                L = lt['mag']
                fc = forecast['mag']
                R = freq_retail
                target = fc * (L + R)
                target_level['mag'] = target
                
                inv_pos = s['mag'] + sum(pipe['mag'].values()) + outstanding['mag'] - backlog['mag']
                order = max(0, target - inv_pos)
                order = min(int(order), ORDER_CAP)
                
                if order > 0:
                    cmd['mag'] = order
                    outstanding['mag'] += order
                    week_log['events'].append(f"🛒 MAGASIN → ENTREPÔT: Commande de {order} pcs")
            
            # ENTREPÔT → USINE
            if t % freq_retail == 0 and t > 0:
                observed = cmd['mag'] if cmd['mag'] > 0 else forecast['mag']
                history['ent'].append(observed)
                history['ent'] = history['ent'][-2:]
                h0 = history['ent'][0] if len(history['ent']) > 0 else 0
                h1 = history['ent'][1] if len(history['ent']) > 1 else h0
                
                if panic == 1:
                    forecast['ent'] = h0 * 0.5 + h1 * 0.5
                elif panic == 2:
                    forecast['ent'] = h0 * 0.25 + h1 * 0.75
                else:
                    forecast['ent'] = h0 * 0.05 + h1 * 0.95
                
                L = lt['ent']
                fc = forecast['ent']
                R = freq_retail
                target = fc * (L + R)
                target_level['ent'] = target
                
                inv_pos = s['ent'] + sum(pipe['ent'].values()) + outstanding['ent']
                order = max(0, target - inv_pos)
                order = min(int(order), ORDER_CAP)
                
                if order > 0:
                    cmd['ent'] = order
                    outstanding['ent'] += order
                    week_log['events'].append(f"🛒 ENTREPÔT → USINE: Commande de {order} pcs")
            
            # USINE → FOURNISSEUR
            if t % freq_supply == 0 and t > 0:
                observed = cmd['ent'] if cmd['ent'] > 0 else forecast['ent']
                history['usi'].append(observed)
                history['usi'] = history['usi'][-2:]
                h0 = history['usi'][0] if len(history['usi']) > 0 else 0
                h1 = history['usi'][1] if len(history['usi']) > 1 else h0
                
                if panic == 1:
                    forecast['usi'] = h0 * 0.5 + h1 * 0.5
                elif panic == 2:
                    forecast['usi'] = h0 * 0.25 + h1 * 0.75
                else:
                    forecast['usi'] = h0 * 0.05 + h1 * 0.95
                
                L = lt['usi']
                fc = forecast['usi']
                R = freq_supply
                target = fc * (L + R)
                target_level['usi'] = target
                
                inv_pos = s['usi'] + sum(pipe['usi'].values()) + outstanding['usi']
                order = max(0, target - inv_pos)
                order = min(int(order), ORDER_CAP)
                
                if order > 0:
                    cmd['usi'] = order
                    outstanding['usi'] += order
                    week_log['events'].append(f"🛒 USINE → FOURNISSEUR: Commande de {order} pcs")
            
            # FOURNISSEUR
            if t % freq_supply == 0 and t > 0:
                observed = cmd['usi'] if cmd['usi'] > 0 else forecast['usi']
                history['four'].append(observed)
                history['four'] = history['four'][-2:]
                h0 = history['four'][0] if len(history['four']) > 0 else 0
                h1 = history['four'][1] if len(history['four']) > 1 else h0
                
                if panic == 1:
                    forecast['four'] = h0 * 0.5 + h1 * 0.5
                elif panic == 2:
                    forecast['four'] = h0 * 0.25 + h1 * 0.75
                else:
                    forecast['four'] = h0 * 0.05 + h1 * 0.95
                
                L = lt['four']
                fc = forecast['four']
                R = freq_supply
                target = fc * (L + R)
                target_level['four'] = target
                
                inv_pos = s['four'] + sum(pipe['four'].values()) + outstanding['four']
                order = max(0, target - inv_pos)
                order = min(int(order), ORDER_CAP)
                
                if order > 0:
                    cmd['four'] = order
                    outstanding['four'] += order
                    week_log['events'].append(f"🛒 FOURNISSEUR: Planification production de {order} pcs")
            
            # Enregistrement
            week_data = {
                't': t,
                'dem': dem,
                's_mag': int(s['mag']),
                's_ent': int(s['ent']),
                's_usi': int(s['usi']),
                's_four': int(s['four']),
                'cmd_mag': int(cmd['mag']),
                'cmd_ent': int(cmd['ent']),
                'cmd_usi': int(cmd['usi']),
                'cmd_four': int(cmd['four']),
                'outstanding_mag': int(outstanding['mag']),
                'outstanding_ent': int(outstanding['ent']),
                'outstanding_usi': int(outstanding['usi']),
                'outstanding_four': int(outstanding['four']),
                'pipe_mag': {k: int(v) for k, v in pipe['mag'].items()},
                'pipe_ent': {k: int(v) for k, v in pipe['ent'].items()},
                'pipe_usi': {k: int(v) for k, v in pipe['usi'].items()},
                'pipe_four': {k: int(v) for k, v in pipe['four'].items()},
                'fc_mag': int(forecast['mag']),
                'fc_ent': int(forecast['ent']),
                'fc_usi': int(forecast['usi']),
                'fc_four': int(forecast['four']),
                'target_mag': int(target_level['mag']),
                'target_ent': int(target_level['ent']),
                'target_usi': int(target_level['usi']),
                'target_four': int(target_level['four']),
                'sales': int(sales),
                'backlog_mag': int(backlog['mag']),
                'events': week_log['events'],
                'lt': lt
            }
            
            all_weeks_data.append(week_data)
        
        return all_weeks_data


# ========== INTERFACE ==========

st.sidebar.header("⚙️ Paramètres de Simulation")

# PARAMÈTRES FINANCIERS
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Paramètres Financiers")
prix_vente = st.sidebar.select_slider("Prix de Vente (€)", options=list(range(50, 210, 10)), value=200)
cout_produit = st.sidebar.select_slider("Coût Produit (€)", options=list(range(10, 60, 10)), value=30)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Demande Client")
spikiness = st.sidebar.slider("Intensité du Pic de Demande", 1, 4, 1, 
                               help="1=500, 2=1000, 3=1500, 4=2000")
spike_duration = st.sidebar.slider("Durée du Pic (semaines)", 1, 20, 9,
                                    help="Nombre de semaines avec demande élevée")


st.sidebar.markdown("---")
panic = st.sidebar.select_slider("Niveau de Panique", options=[1, 2, 3], value=1)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Fréquence")
freq_retail = st.sidebar.selectbox("Magasin & Entrepôt", [1, 2, 4], index=0, format_func=lambda x: f"{x} sem")
freq_supply = st.sidebar.selectbox("Usine & Fournisseur", [1, 4, 12], index=0, format_func=lambda x: f"{x} sem")

st.sidebar.markdown("---")
lt_usine = st.sidebar.select_slider("Délai Production Usine (sem)", options=[2, 6, 10], value=2)
lt_fournisseur = st.sidebar.select_slider("Délai Fournisseur MP (sem)", options=[6, 12, 18, 24], value=6)

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Stocks Initiaux")
stock_mag = st.sidebar.slider("Stock Initial Magasin", 100, 2000, 500, 100)
stock_ent = st.sidebar.slider("Stock Initial Entrepôt", 100, 2000, 1100, 100)
stock_usi = st.sidebar.slider("Stock Initial Usine", 0, 5000, 3200, 200)
stock_four = st.sidebar.slider("Stock Initial Fournisseur", 0, 5000, 5000, 200)

if st.sidebar.button("▶ LANCER SIMULATION", type="primary"):
    sim = BullwhipSimulator(spikiness=spikiness, spike_duration=spike_duration)
    all_data = sim.simulate_full(panic, lt_usine, lt_fournisseur, freq_retail, freq_supply,
                                  stock_mag, stock_ent, stock_usi, stock_four, 
                                  pv=prix_vente, pc=cout_produit)
    st.session_state['simulation_data'] = all_data
    st.session_state['current_week'] = 0
    st.session_state['pv'] = prix_vente
    st.session_state['pc'] = cout_produit
    st.session_state['spikiness'] = spikiness
    st.session_state['spike_duration'] = spike_duration 

# ========== ANALYSE COMBINATOIRE OPTIMISÉE ==========
st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Analyse Combinatoire Complète")

# Options avancées
with st.sidebar.expander("⚙️ Options avancées"):
    test_all_combinations = st.checkbox("Tester TOUTES les combinaisons possibles", value=False)
    nb_simulations = st.number_input(
        "Nombre de simulations aléatoires", 
        min_value=1000, 
        max_value=100000,
        value=10000, 
        step=1000,
        disabled=test_all_combinations,
        help="Nombre de combinaisons aléatoires à tester"
    )
    
    # Pas de variation pour accélérer
    stock_step = st.select_slider("Pas de variation stocks", options=[200, 400, 500, 600, 1000], value=500)
    
    # Tester spike duration et hauteur
    test_spike_params = st.checkbox("Varier intensité et durée du spike", value=True)

if st.sidebar.button("🚀 LANCER ANALYSE COMPLÈTE", type="secondary"):
    
    with st.spinner("⏳ Analyse en cours..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Paramètres à tester
        panics = [1, 2, 3]
        lt_usines = [2, 6, 10]
        lt_fournisseurs = [6, 12, 18, 24]
        freq_retails = [1, 2, 4]
        freq_supplies = [1, 4, 12]
        stocks_mag = list(range(200, 2200, stock_step))
        stocks_ent = list(range(200, 2200, stock_step))
        stocks_usi = list(range(0, 5200, stock_step))
        stocks_four = list(range(0, 5200, stock_step))
        
        # Spike parameters
        if test_spike_params:
            spikiness_levels = [1, 2, 3, 4]
            spike_duration_levels = [4, 6, 9, 12, 15]
        else:
            spikiness_levels = [spikiness]
            spike_duration_levels = [spike_duration]
        
        # Calculer nombre total
        total_possible = (len(panics) * len(lt_usines) * len(lt_fournisseurs) * 
                         len(freq_retails) * len(freq_supplies) * len(stocks_mag) * 
                         len(stocks_ent) * len(stocks_usi) * len(stocks_four) *
                         len(spikiness_levels) * len(spike_duration_levels))
        
        if test_all_combinations:
            nb_to_test = total_possible
            status_text.info(f"📊 Test de TOUTES les {total_possible:,} combinaisons possibles")
        else:
            nb_to_test = min(nb_simulations, total_possible)
            status_text.info(f"📊 Test de {nb_to_test:,} combinaisons sur {total_possible:,} possibles")
        
        results = []
        np.random.seed(42)
        
        import time as time_module
        start_time = time_module.time()
        
        # Génération optimisée
        if test_all_combinations:
            # TOUTES les combinaisons
            from itertools import product as iter_product
            all_params = iter_product(
                panics, lt_usines, lt_fournisseurs, freq_retails, freq_supplies,
                stocks_mag, stocks_ent, stocks_usi, stocks_four,
                spikiness_levels, spike_duration_levels
            )
            params_list = list(all_params)
        else:
            # Échantillonnage aléatoire
            params_list = []
            for _ in range(nb_to_test):
                params_list.append((
                    np.random.choice(panics),
                    np.random.choice(lt_usines),
                    np.random.choice(lt_fournisseurs),
                    np.random.choice(freq_retails),
                    np.random.choice(freq_supplies),
                    np.random.choice(stocks_mag),
                    np.random.choice(stocks_ent),
                    np.random.choice(stocks_usi),
                    np.random.choice(stocks_four),
                    np.random.choice(spikiness_levels),
                    np.random.choice(spike_duration_levels)
                ))
        
        # Simulation en batch
        for idx, (p, lt_u, lt_f, fr_ret, fr_sup, s_m, s_e, s_u, s_f, spk, spike_dur) in enumerate(params_list):
            
            try:
                sim = BullwhipSimulator(spikiness=spk, spike_duration=spike_dur)
                data = sim.simulate_full(p, lt_u, lt_f, fr_ret, fr_sup, s_m, s_e, s_u, s_f, prix_vente, cout_produit)
                
                # Calcul rapide des métriques
                total_sales = sum(w['sales'] for w in data)
                total_demand = sum(w['dem'] for w in data)
                ca_total = total_sales * prix_vente / 1000
                cout_production = total_sales * cout_produit / 1000
                stock_final = data[-1]['s_mag'] + data[-1]['s_ent'] + data[-1]['s_usi'] + data[-1]['s_four']
                cout_stock_final = stock_final * cout_produit / 1000
                marge_absolue = ca_total - cout_production - cout_stock_final
                service_level = (total_sales / total_demand * 100) if total_demand > 0 else 0
                
                results.append({
                    'panic': p,
                    'lt_usine': lt_u,
                    'lt_fournisseur': lt_f,
                    'freq_retail': fr_ret,
                    'freq_supply': fr_sup,
                    'stock_mag': s_m,
                    'stock_ent': s_e,
                    'stock_usi': s_u,
                    'stock_four': s_f,
                    'spikiness': spk,
                    'spike_duration': spike_dur,
                    'marge_nette': marge_absolue,
                    'service_level': service_level,
                    'ca_total': ca_total,
                    'ventes_totales': total_sales,
                    'demande_totale': total_demand
                })
                
                del sim, data
                
            except:
                pass
            
            # Mise à jour tous les 100
            if idx % 100 == 0 and idx > 0:
                progress_bar.progress(idx / nb_to_test)
                elapsed = time_module.time() - start_time
                speed = idx / elapsed
                remaining = (nb_to_test - idx) / speed if speed > 0 else 0
                status_text.text(f"⏳ {idx:,}/{nb_to_test:,} | {speed:.0f} sim/s | Restant: ~{remaining/60:.1f}min")
        
        progress_bar.progress(1.0)
        elapsed_total = time_module.time() - start_time
        
        df_results = pd.DataFrame(results)
        st.session_state['combinatorial_results'] = df_results
        
        # ===== RÉGRESSION TREE (Plus rapide et performant) =====
        
        X = df_results[['panic', 'lt_usine', 'lt_fournisseur', 'freq_retail', 'freq_supply',
                        'stock_mag', 'stock_ent', 'stock_usi', 'stock_four', 
                        'spikiness', 'spike_duration']]
        y = df_results['marge_nette']
        
        # Decision Tree (pas besoin de normalisation)
        from sklearn.tree import DecisionTreeRegressor
        model_tree = DecisionTreeRegressor(max_depth=15, min_samples_split=20, random_state=42)
        model_tree.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'Paramètre': X.columns,
            'Importance': model_tree.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Random Forest pour comparaison
        from sklearn.ensemble import RandomForestRegressor
        model_rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        model_rf.fit(X, y)
        
        feature_importance_rf = pd.DataFrame({
            'Paramètre': X.columns,
            'Importance': model_rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.session_state['feature_importance_tree'] = feature_importance
        st.session_state['regression_score_tree'] = model_tree.score(X, y)
        st.session_state['feature_importance_rf'] = feature_importance_rf
        st.session_state['regression_score_rf'] = model_rf.score(X, y)
        
        st.success(f"✅ Analyse terminée ! {len(df_results):,} simulations en {elapsed_total/60:.1f} min | Vitesse: {len(df_results)/elapsed_total:.0f} sim/s")




# ========== AFFICHAGE RÉSULTATS ==========

if 'combinatorial_results' in st.session_state:
    st.markdown("---")
    st.markdown("## 🔬 Résultats de l'Analyse Combinatoire Complète")
    
    df_comb = st.session_state['combinatorial_results']
    feat_tree = st.session_state.get('feature_importance_tree')
    r2_tree = st.session_state.get('regression_score_tree')
    feat_rf = st.session_state.get('feature_importance_rf')
    r2_rf = st.session_state.get('regression_score_rf')
    
    # Export Excel
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Combinaisons')
        return output.getvalue()
    
    col_export1, col_export2, col_export3 = st.columns([2, 1, 1])
    with col_export1:
        st.download_button(
            label="📥 TÉLÉCHARGER TOUTES LES COMBINAISONS (Excel)",
            data=to_excel(df_comb),
            file_name="combinaisons_bullwhip_complete.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
    with col_export2:
        st.metric("Simulations", f"{len(df_comb):,}")
    with col_export3:
        nb_params = len([c for c in df_comb.columns if c not in ['marge_nette', 'service_level', 'ca_total', 'ventes_totales', 'demande_totale']])
        st.metric("Paramètres", str(nb_params))
    
    st.markdown("---")
    
    # ===== IMPORTANCE DES PARAMÈTRES =====
    st.markdown("### 🎯 Quels Paramètres Impactent le Plus la Marge ?")
    
    col_model1, col_model2 = st.columns(2)
    
    with col_model1:
        st.markdown(f"#### 🌲 Decision Tree - R² = **{r2_tree:.3f}**")
        st.dataframe(feat_tree, use_container_width=True, hide_index=True)
        
        fig_tree = px.bar(
            feat_tree, 
            x='Importance', 
            y='Paramètre',
            orientation='h',
            title="Decision Tree - Importance des Paramètres",
            color='Importance',
            color_continuous_scale='Greens'
        )
        fig_tree.update_layout(height=500)
        st.plotly_chart(fig_tree, use_container_width=True)
    
    with col_model2:
        st.markdown(f"#### 🌳 Random Forest - R² = **{r2_rf:.3f}**")
        st.dataframe(feat_rf, use_container_width=True, hide_index=True)
        
        fig_rf = px.bar(
            feat_rf, 
            x='Importance', 
            y='Paramètre',
            orientation='h',
            title="Random Forest - Importance des Paramètres",
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig_rf.update_layout(height=500)
        st.plotly_chart(fig_rf, use_container_width=True)
    
    # ===== STATISTIQUES =====
    st.markdown("---")
    st.markdown("### 📈 Distribution de la Marge Nette")
    
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    
    with col_d1:
        st.metric("Marge Moyenne", f"{df_comb['marge_nette'].mean():.1f} K€")
    with col_d2:
        st.metric("Marge Max", f"{df_comb['marge_nette'].max():.1f} K€")
    with col_d3:
        st.metric("Marge Min", f"{df_comb['marge_nette'].min():.1f} K€")
    with col_d4:
        st.metric("Écart-Type", f"{df_comb['marge_nette'].std():.1f} K€")
    
    # ===== GRAPHIQUES =====
    col_hist1, col_hist2 = st.columns(2)
    
    with col_hist1:
        fig_hist = px.histogram(
            df_comb, 
            x='marge_nette', 
            nbins=50,
            title="Distribution de la Marge Nette",
            labels={'marge_nette': 'Marge Nette (K€)'},
            color_discrete_sequence=['#667eea']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_hist2:
        fig_scatter = px.scatter(
            df_comb,
            x='service_level',
            y='marge_nette',
            color='spikiness',
            title="Marge vs Service Level (par Spikiness)",
            labels={'service_level': 'Service Level (%)', 'marge_nette': 'Marge (K€)'},
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # ===== TOP 10 =====
    st.markdown("---")
    st.markdown("### 🏆 Top 10 Meilleures Configurations")
    
    top10 = df_comb.nlargest(10, 'marge_nette')
    st.dataframe(top10, use_container_width=True, hide_index=True)
    
    # ===== MEILLEURE CONFIG =====
    st.markdown("---")
    st.markdown("### 🎯 Configuration Optimale")
    
    best = df_comb.loc[df_comb['marge_nette'].idxmax()]
    
    col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
    
    with col_b1:
        st.markdown(f"""
        **Stocks:**
        - Mag: {best['stock_mag']:.0f}
        - Ent: {best['stock_ent']:.0f}
        - Usine: {best['stock_usi']:.0f}
        - Frns: {best['stock_four']:.0f}
        """)
    with col_b2:
        st.markdown(f"""
        **Lead Times:**
        - Usine: {best['lt_usine']:.0f} sem
        - Frns: {best['lt_fournisseur']:.0f} sem
        """)
    with col_b3:
        st.markdown(f"""
        **Fréquences:**
        - Retail: {best['freq_retail']:.0f} sem
        - Supply: {best['freq_supply']:.0f} sem
        """)
    with col_b4:
        st.markdown(f"""
        **Demande:**
        - Spikiness: {best['spikiness']:.0f}
        - Durée: {best['spike_duration']:.0f} sem
        - Panique: {best['panic']:.0f}
        """)
    with col_b5:
        st.markdown(f"""
        **Performance:**
        - **Marge: {best['marge_nette']:.1f} K€**
        - Service: {best['service_level']:.1f}%
        - CA: {best['ca_total']:.1f} K€
        """)




# ========== VISUALISATION ==========

if 'simulation_data' in st.session_state:
    data = st.session_state['simulation_data']
    max_week = len(data) - 1
    pv = st.session_state.get('pv', 200)
    pc = st.session_state.get('pc', 30)
    spk = st.session_state.get('spikiness', 1)
    spike_dur = st.session_state.get('spike_duration', 9)


    
    st.markdown("---")
    st.subheader("🎬 Mode Pas-à-Pas - Navigation")
    
    # Initialize
    if 'current_week' not in st.session_state:
        st.session_state['current_week'] = 0

    # Callback functions
    def go_to_start():
        st.session_state['current_week'] = 0

    def go_previous():
        if st.session_state['current_week'] > 0:
            st.session_state['current_week'] -= 1

    def go_next():
        if st.session_state['current_week'] < max_week:
            st.session_state['current_week'] += 1

    def go_to_end():
        st.session_state['current_week'] = max_week

    # Buttons with callbacks
    col_nav1, col_nav2, col_nav3, col_nav4, col_nav5 = st.columns([1, 1, 3, 1, 1])

    with col_nav1:
        st.button("⏮ Début", on_click=go_to_start, use_container_width=True)

    with col_nav2:
        st.button("◀ Précédent", on_click=go_previous, use_container_width=True)

    with col_nav3:
        st.slider(
            "Semaine", 
            0, 
            max_week, 
            key='current_week',  # Bind directly to session state
            label_visibility="visible"
        )

    with col_nav4:
        st.button("Suivant ▶", on_click=go_next, use_container_width=True)

    with col_nav5:
        st.button("Fin ⏭", on_click=go_to_end, use_container_width=True)

    # Use the week
    week = data[st.session_state['current_week']]

    lt = week['lt']
    
    st.markdown(f"## 📅 Semaine {st.session_state['current_week']}")

    
    # ========== VISUALISATION SUPPLY CHAIN HORIZONTALE ==========
    
    st.markdown("### 🔄 Vue Supply Chain")
    
    # FLUX COMMANDES (EN HAUT - vers l'amont ←)
    st.markdown("#### ← ← ← FLUX INFORMATION (Commandes)")
    
    col_cmd1, col_cmd2, col_cmd3, col_cmd4 = st.columns(4)
    
    with col_cmd4:  # MAGASIN à droite
        if week['cmd_mag'] > 0:
            st.markdown(f"""
            <div style='background:#667eea; padding:10px; border-radius:5px; text-align:center; color:white;'>
                📝 Cmd: <strong>{week['cmd_mag']}</strong> pcs
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='padding:10px; text-align:center; color:#ccc;'>—</div>", unsafe_allow_html=True)
    
    with col_cmd3:  # ENTREPÔT
        if week['cmd_ent'] > 0:
            st.markdown(f"""
            <div style='background:#f093fb; padding:10px; border-radius:5px; text-align:center; color:white;'>
                📝 Cmd: <strong>{week['cmd_ent']}</strong> pcs
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='padding:10px; text-align:center; color:#ccc;'>—</div>", unsafe_allow_html=True)
    
    with col_cmd2:  # USINE
        if week['cmd_usi'] > 0:
            st.markdown(f"""
            <div style='background:#4facfe; padding:10px; border-radius:5px; text-align:center; color:white;'>
                📝 Cmd: <strong>{week['cmd_usi']}</strong> pcs
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='padding:10px; text-align:center; color:#ccc;'>—</div>", unsafe_allow_html=True)
    
    with col_cmd1:  # FOURNISSEUR à gauche
        if week['cmd_four'] > 0:
            st.markdown(f"""
            <div style='background:#43e97b; padding:10px; border-radius:5px; text-align:center; color:white;'>
                🏭 Prod: <strong>{week['cmd_four']}</strong> pcs
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='padding:10px; text-align:center; color:#ccc;'>—</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # LES 4 ÉTAGES (AMONT → AVAL = Gauche → Droite)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 20px; border-radius: 10px; color: white;'>
            <h3 style='margin:0; text-align:center;'>🚚 FOURNISSEUR</h3>
            <hr style='border-color: white;'>
            <p><strong>Stock:</strong> {week['s_four']} pcs</p>
            <p><strong>Outstanding:</strong> {week['outstanding_four']} pcs</p>
            <p><strong>Forecast:</strong> {week['fc_four']} pcs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 20px; border-radius: 10px; color: white;'>
            <h3 style='margin:0; text-align:center;'>🏗️ USINE</h3>
            <hr style='border-color: white;'>
            <p><strong>Stock:</strong> {week['s_usi']} pcs</p>
            <p><strong>Outstanding:</strong> {week['outstanding_usi']} pcs</p>
            <p><strong>Forecast:</strong> {week['fc_usi']} pcs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 10px; color: white;'>
            <h3 style='margin:0; text-align:center;'>🏭 ENTREPÔT</h3>
            <hr style='border-color: white;'>
            <p><strong>Stock:</strong> {week['s_ent']} pcs</p>
            <p><strong>Outstanding:</strong> {week['outstanding_ent']} pcs</p>
            <p><strong>Forecast:</strong> {week['fc_ent']} pcs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white;'>
            <h3 style='margin:0; text-align:center;'>🏪 MAGASIN</h3>
            <hr style='border-color: white;'>
            <p><strong>Stock:</strong> {week['s_mag']} pcs</p>
            <p><strong>Demande:</strong> {week['dem']} pcs</p>
            <p><strong>Ventes:</strong> {week['sales']} pcs</p>
            <p><strong>Outstanding:</strong> {week['outstanding_mag']} pcs</p>
            <p><strong>Forecast:</strong> {week['fc_mag']} pcs</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # FLUX PRODUITS (EN BAS - vers l'aval →)
    st.markdown("#### → → → FLUX PRODUITS (Livraisons)")
    
    # FOURNISSEUR → USINE
    st.markdown(f"**FOURNISSEUR → USINE** (LT = {lt['usi']} sem)")
    pipe_four = week['pipe_four']
    if pipe_four:
        sorted_items = sorted(pipe_four.items())[:lt['usi']]
        if sorted_items:
            cols = st.columns(max(len(sorted_items), 1))
            for idx, (arrival_week, qty) in enumerate(sorted_items):
                with cols[idx]:
                    st.markdown(f"""
                    <div style='background:#43e97b; padding:8px; border-radius:5px; text-align:center; margin:2px;'>
                        <small>Sem {arrival_week}</small><br><strong>{qty}</strong>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Aucun transit")
    
    # USINE → ENTREPÔT
    st.markdown(f"**USINE → ENTREPÔT** (LT = {lt['ent']} sem)")
    pipe_usi = week['pipe_usi']
    if pipe_usi:
        sorted_items = sorted(pipe_usi.items())[:lt['ent']]
        if sorted_items:
            cols = st.columns(max(len(sorted_items), 1))
            for idx, (arrival_week, qty) in enumerate(sorted_items):
                with cols[idx]:
                    st.markdown(f"""
                    <div style='background:#4facfe; padding:8px; border-radius:5px; text-align:center; margin:2px;'>
                        <small>Sem {arrival_week}</small><br><strong>{qty}</strong>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Aucun transit")
    
    # ENTREPÔT → MAGASIN
    st.markdown(f"**ENTREPÔT → MAGASIN** (LT = {lt['mag']} sem)")
    pipe_ent = week['pipe_ent']
    if pipe_ent:
        sorted_items = sorted(pipe_ent.items())[:lt['mag']]
        if sorted_items:
            cols = st.columns(max(len(sorted_items), 1))
            for idx, (arrival_week, qty) in enumerate(sorted_items):
                with cols[idx]:
                    st.markdown(f"""
                    <div style='background:#f093fb; padding:8px; border-radius:5px; text-align:center; margin:2px;'>
                        <small>Sem {arrival_week}</small><br><strong>{qty}</strong>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Aucun transit")
    
    # ========== LOG DES ÉVÉNEMENTS ==========
    
    st.markdown("---")
    st.subheader(f"📋 Événements de la Semaine {st.session_state['current_week']}")
    
    if week['events']:
        for event in week['events']:
            st.markdown(f"- {event}")
    else:
        st.info("Aucun événement cette semaine")
    
    # ========== GRAPHIQUES AGRÉGÉS ==========
    
    st.markdown("---")
    st.subheader("📊 Analyse Globale de la Simulation")
    
    # Préparer les données agrégées
    all_weeks = []
    for w in data:
        all_weeks.append({
            't': w['t'],
            'dem': w['dem'],
            's_mag': w['s_mag'],
            's_ent': w['s_ent'],
            's_usi': w['s_usi'],
            's_four': w['s_four'],
            'cmd_mag': w['cmd_mag'],
            'cmd_ent': w['cmd_ent'],
            'cmd_usi': w['cmd_usi'],
            'cmd_four': w['cmd_four'],
            'sales': w['sales']
        })
    
    df_analysis = pd.DataFrame(all_weeks)
    
    # Calcul Service Level par semaine
    df_analysis['service_level'] = df_analysis.apply(
        lambda row: 100 * row['sales'] / row['dem'] if row['dem'] > 0 else 100,
        axis=1
    )
    
    # ========== KPIs FINANCIERS ==========
    
    total_sales = df_analysis['sales'].sum()
    total_demand = df_analysis['dem'].sum()
    total_lost = total_demand - total_sales
    
    ca_total = total_sales * pv / 1000  # en K€
    cout_production = total_sales * pc / 1000  # en K€
    
    stock_final_total = df_analysis.iloc[-1][['s_mag', 's_ent', 's_usi', 's_four']].sum()
    cout_stock_final = stock_final_total * pc / 1000  # en K€
    
    marge_absolue = ca_total - cout_production - cout_stock_final  # en K€
    marge_pct = (marge_absolue / ca_total * 100) if ca_total > 0 else 0
    
    service_global = (total_sales / total_demand * 100) if total_demand > 0 else 0
    
    # Affichage KPIs
    st.markdown("### 💰 Indicateurs Financiers")
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    
    with col_kpi1:
        st.metric("CA Total", f"{ca_total:.1f} K€")
    
    with col_kpi2:
        st.metric("Coût Production", f"{cout_production:.1f} K€")
    
    with col_kpi3:
        st.metric("Marge Absolue", f"{marge_absolue:.1f} K€")
    
    with col_kpi4:
        st.metric("Marge %", f"{marge_pct:.1f}%")
    
    with col_kpi5:
        st.metric("Service Level Global", f"{service_global:.1f}%")
    
    st.markdown("---")
    
    # ========== GRAPHIQUES ==========
    
    # Graphique 1: Taux de Service Client
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        fig_service = go.Figure()
        fig_service.add_trace(go.Scatter(
            x=df_analysis['t'],
            y=df_analysis['service_level'],
            mode='lines+markers',
            name='Service Level',
            fill='tozeroy',
            line=dict(color='#28a745', width=2),
            marker=dict(size=4)
        ))
        
        fig_service.add_hline(y=95, line_dash="dash", line_color="orange", 
                             annotation_text="Cible 95%")
        fig_service.add_hline(y=100, line_dash="dot", line_color="green")
        
        fig_service.update_layout(
            title="📈 Taux de Service Client (par semaine)",
            xaxis_title="Semaine",
            yaxis_title="Service Level (%)",
            yaxis_range=[0, 105],
            height=400
        )
        st.plotly_chart(fig_service, use_container_width=True, key='fig_service_global')
    
    # Graphique 2: Commandes par Étage + DEMANDE CLIENT
    with col_g2:
        fig_orders = go.Figure()
        
        # DEMANDE CLIENT en premier
        fig_orders.add_trace(go.Scatter(
            x=df_analysis['t'],
            y=df_analysis['dem'],
            mode='lines+markers',
            name='Demande Client',
            line=dict(color='#FF6B6B', width=3, dash='dot'),
            marker=dict(size=6, symbol='star')
        ))
        
        fig_orders.add_trace(go.Scatter(
            x=df_analysis['t'],
            y=df_analysis['cmd_mag'],
            mode='lines+markers',
            name='Magasin',
            line=dict(color='#667eea', width=2),
            marker=dict(size=4)
        ))
        
        fig_orders.add_trace(go.Scatter(
            x=df_analysis['t'],
            y=df_analysis['cmd_ent'],
            mode='lines+markers',
            name='Entrepôt',
            line=dict(color='#f093fb', width=2),
            marker=dict(size=4)
        ))
        
        fig_orders.add_trace(go.Scatter(
            x=df_analysis['t'],
            y=df_analysis['cmd_usi'],
            mode='lines+markers',
            name='Usine',
            line=dict(color='#4facfe', width=2),
            marker=dict(size=4)
        ))
        
        fig_orders.add_trace(go.Scatter(
            x=df_analysis['t'],
            y=df_analysis['cmd_four'],
            mode='lines+markers',
            name='Fournisseur',
            line=dict(color='#43e97b', width=2),
            marker=dict(size=4)
        ))
        
        fig_orders.update_layout(
            title="📦 Demande Client & Commandes par Étage",
            xaxis_title="Semaine",
            yaxis_title="Quantité",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_orders, use_container_width=True, key='fig_orders_global')
    
    # Graphique 3: Stocks par Étage
    st.markdown("### 📊 Évolution des Stocks par Étage")
    
    fig_stocks = go.Figure()
    
    fig_stocks.add_trace(go.Scatter(
        x=df_analysis['t'],
        y=df_analysis['s_mag'],
        mode='lines',
        name='Magasin',
        fill='tonexty',
        line=dict(color='#667eea', width=2)
    ))
    
    fig_stocks.add_trace(go.Scatter(
        x=df_analysis['t'],
        y=df_analysis['s_ent'],
        mode='lines',
        name='Entrepôt',
        fill='tonexty',
        line=dict(color='#f093fb', width=2)
    ))
    
    fig_stocks.add_trace(go.Scatter(
        x=df_analysis['t'],
        y=df_analysis['s_usi'],
        mode='lines',
        name='Usine',
        fill='tonexty',
        line=dict(color='#4facfe', width=2)
    ))
    
    fig_stocks.add_trace(go.Scatter(
        x=df_analysis['t'],
        y=df_analysis['s_four'],
        mode='lines',
        name='Fournisseur',
        fill='tonexty',
        line=dict(color='#43e97b', width=2)
    ))
    
    fig_stocks.update_layout(
        title="Niveaux de Stock - Tous Étages",
        xaxis_title="Semaine",
        yaxis_title="Stock (unités)",
        height=450
    )
    st.plotly_chart(fig_stocks, use_container_width=True, key='fig_stocks_global')
    
    # ========== TABLEAU RÉCAPITULATIF ==========
    
    st.markdown("---")
    st.markdown("### 📋 Résumé de la Simulation")
    
    summary_data = {
        'Indicateur': [
            'Demande Totale',
            'Ventes Totales',
            'Ventes Perdues',
            'Service Level Global',
            'CA Total',
            'Coût Production',
            'Stock Final (unités)',
            'Coût Stock Final',
            'Marge Absolue',
            'Marge %',
            'Spikiness Demande'
        ],
        'Valeur': [
            f"{int(total_demand):,} pcs",
            f"{int(total_sales):,} pcs",
            f"{int(total_lost):,} pcs",
            f"{service_global:.1f}%",
            f"{ca_total:.1f} K€",
            f"{cout_production:.1f} K€",
            f"{int(stock_final_total):,} pcs",
            f"{cout_stock_final:.1f} K€",
            f"{marge_absolue:.1f} K€",
            f"{marge_pct:.1f}%",
            f"{spk} / 4"
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    
    st.dataframe(
        df_summary.style.set_properties(**{
            'background-color': '#f0f2f6',
            'color': '#262730',
            'border-color': 'white'
        }),
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("👈 Configurez les paramètres et cliquez sur 'LANCER SIMULATION' pour commencer")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Simulateur Bullwhip - Mode Pas-à-Pas v4.0</strong></p>
    <p>✅ Spikiness Demande | ✅ Demande Client Visible | ✅ Export Excel | ✅ Régression Linéaire + Tree | ✅ Feature Importance</p>
</div>
""", unsafe_allow_html=True)

