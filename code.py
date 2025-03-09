import GEOparse
import numpy as np
import pandas as pd
import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import scipy as sc
import sklearn as sk
import sklearn.feature_selection
from statsmodels.stats.anova import AnovaRM
from tqdm import tqdm
import scipy.stats as stats
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.multitest import multipletests

#### **Funzione per la costruzione del dataframe con i dati della serie**
def get_series(series, class_col='title', gene_name_col=None):
    if gene_name_col is None:
        raise(ValueError('Il parametro gene_name_col deve essere specificato'))
    gse = GEOparse.get_GEO(geo=series, silent=True)
    platform_ids = gse.metadata['platform_id']
    samples = gse.phenotype_data.index  # ['geo_accession']
    classes = gse.phenotype_data['title']
    columns = gse.gsms[samples[0]].table['ID_REF']
    ns = len(samples)
    ng = len(columns)
    ls = []
    for i in range(ns):
        s = gse.gsms[samples[i]].table
        c_drop = s.columns[s.columns != "VALUE"]
        s = s.drop(c_drop, axis=1).transpose()
        ls.append(s)
    df = pd.concat(ls)
    df.index = samples
    df.columns = columns
    df.insert(ng, 'CLASS', classes)
    gpls = []
    for pl in platform_ids:
        table = gse.gpls[pl].table
        table.index = table.ID
        gpls.append(table.drop('ID', axis=1))
    if len(gpls) == 1:
        gpls = gpls[0]
    else:
        warnings.warn("Sono presenti piattaforme multiple, viene restituita una lista di tabelle gpls")
    if gene_name_col not in gpls.columns:
        raise(ValueError(f'La colonna {gene_name_col} non è presente nella tabella gpls'))
    return df, gse, gpls.loc[:, gene_name_col]

def get_gene_names(cell_ids, gene_symbols):
    return gene_symbols.loc[cell_ids].dropna().unique()

def sam_test(data1, data2, alpha=0.001, num_permutations=750, seed=3):
    # Calcola statistiche di base
    n1 = len(data1)
    n2 = len(data2)

    # Calcola medie per ogni gene
    mean1 = np.mean(data1, axis=0)
    mean2 = np.mean(data2, axis=0)

    # Calcola la deviazione standard
    s1 = np.std(data1, axis=0, ddof=1)
    s2 = np.std(data2, axis=0, ddof=1)

    # Calcola s0 (small positive constant)
    s = np.sqrt(1/n1 + 1/n2) * np.sqrt((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2)
    s0 = np.percentile(s, 5)

    # Calcola d statistic
    d = ((mean1 - mean2) / (s + s0)).to_numpy()

    # Permutazioni per stimare FDR
    d_perm = np.zeros((num_permutations, len(d)))
    combined = np.vstack((data1, data2))
    np.random.seed(seed)

    for i in tqdm(range(num_permutations), bar_format='|{bar}|', colour='cyan'):
        np.random.shuffle(combined)
        perm1 = combined[:n1]
        perm2 = combined[n1:]
        mean1_perm = np.mean(perm1, axis=0)
        mean2_perm = np.mean(perm2, axis=0)
        s1_perm = np.std(perm1, axis=0, ddof=1)
        s2_perm = np.std(perm2, axis=0, ddof=1)
        s_perm = np.sqrt(1/n1 + 1/n2) * np.sqrt((n1-1)*s1_perm**2 + (n2-1)*s2_perm**2) / (n1+n2-2)
        d_perm[i] = (mean1_perm - mean2_perm) / (s_perm + s0)

    # Calcola p-values
    p_values = np.zeros_like(d)
    for i in tqdm(range(len(d)), bar_format='|{bar}|', colour='cyan'):
        p_values[i] = np.mean(np.abs(d_perm[:, i]) >= np.abs(d[i]))

    return p_values

def mutual_information(data1, data2, thres):
    x = pd.concat([data1, data2], axis=0)
    y = np.zeros(len(data1) + len(data2))
    y[len(data1):len(data1)+len(data2)] = 1
    scores = sk.feature_selection.mutual_info_regression(x, y, discrete_features=False)
    return scores

def anova_test_dependent(*groups, subjects, alpha):
    lengths = []
    for g in groups:
        lengths.append(len(g))
    ntot = np.sum(lengths)
    pvalues = np.zeros(groups[0].shape[0])
    y = np.zeros(ntot)
    n = 0
    for i, l in enumerate(lengths):
        y[n:n+l] = i
        n = n + l
    df = pd.DataFrame({
        'subject': subjects,
        'dep_var': y,
        'data': x
    })
    x = np.zeros(ntot)
    for i in range(groups[0].shape[0]):
        for g in enumerate(groups):
            x[l:l + len(g)] = g[i]
            l = l + len(g)
        df.loc[:, 'data'] = x[:]
        aovrm = AnovaRM(df, 'data', 'subject', ['dep_var'])
        aores = aovrm.fit()
        pvalues.append(float(aores.anova_table["Pr > F"]["dep_var"]))
    return pvalues

def volcano_plot(log2fc, log10pv, thres=[1, 1.5, -1, 1.7]):
    thy1, thy2, thx1, thx2 = thres
    if thy1 <= 0:
        raise Exception("ERRORE: la soglia 1 deve essere positiva")
    if thy2 <= 0:
        raise Exception("ERRORE: la soglia 2 deve essere positiva")
    if thx1 >= 0:
        raise Exception("ERRORE: la soglia 3 deve essere negativa")
    if thx2 <= 0:
        raise Exception("ERRORE: la soglia 4 deve essere positiva")
    down = (log2fc < thx1) & (log10pv > thy1)
    up = (log2fc > thx2) & (log10pv > thy2)
    seaborn.scatterplot(x=log2fc, y=log10pv)
    seaborn.scatterplot(x=log2fc[down], y=log10pv[down])
    seaborn.scatterplot(x=log2fc[up], y=log10pv[up])
    plt.axvline(thx1, color='gray', linestyle='--')
    plt.axvline(thx2, color='gray', linestyle='--')
    plt.hlines(thy1, xmin=log2fc.min(), xmax=0, color='gray', linestyle='--')
    plt.hlines(thy2, xmin=0, xmax=log2fc.max(), color='gray', linestyle='--')
    plt.title('Volcano Plot')
    plt.xlabel('$log_2 fc$')
    plt.ylabel('$-log_{10} pv$')
    plt.grid()
    plt.show()
    # ordino per distanza radiale
    x_down = log2fc[down]
    y_down = log10pv[down]
    down_idx = down.index[down]
    dist = (x_down - thx1)**2 + (y_down - thy1)**2
    down_idx = down_idx[np.argsort(-1 * dist)]

    x_up = log2fc[up]
    y_up = log10pv[up]
    up_idx = up.index[up]
    dist = (x_up - thx2)**2 + (y_up - thy2)**2
    up_idx = up_idx[np.argsort(-1 * dist)]

    return (down_idx, up_idx)

def disambigua(vals, df):
    means = df.groupby('CLASS').mean()  # assumendo DUE gruppi
    dm = means.iloc[0] / means.iloc[1]
    dm = np.abs(np.log(dm))
    uniq_val, uniq_count = np.unique(vals, return_counts=True)
    for u, c in zip(uniq_val, uniq_count):
        ids = np.argsort(vals)
        pred = vals[vals < u]
        if len(pred) == 0:
            pred = u
        else:
            pred = (pred.max() + u) / 2
        succ = vals[vals > u]
        if len(succ) == 0:
            succ = u
        else:
            succ = (succ.min() + u) / 2
        ord_idx = np.argsort(-1 * dm[vals == u])
        new_vals = np.linspace(pred, succ, c)
        new_vals = new_vals[ord_idx]
        vals[vals == u] = new_vals
    return vals

# --------------------------
# FUNZIONI CHE INCAPSULANO I VARI TEST
# --------------------------

def run_ttest(G1, G2, gene_symbols, means, alpha=0.05, csv_filename='ttest_results.csv'):
    print("=============================== T-TEST ===============================")
    stat, pv_tt = sc.stats.ttest_ind(G1, G2)
    # Correzione del p-value (opzionale)
    pv_tt = sc.stats.false_discovery_control(pv_tt)
    # Selezione dei geni con p-value < alpha
    sig_mask = pv_tt < alpha
    sig_genes = G1.columns[sig_mask]
    # Ordinamento per p-value
    sorted_idx = np.argsort(pv_tt[sig_mask])
    sig_genes = sig_genes[sorted_idx]
    # Separazione in geni sotto- e sovra-espressi
    under = sig_genes[means[sig_genes].iloc[0] < means[sig_genes].iloc[1]]
    over = sig_genes[means[sig_genes].iloc[0] >= means[sig_genes].iloc[1]]
    gene_names_under = get_gene_names(under, gene_symbols)
    gene_names_over = get_gene_names(over, gene_symbols)
    gene_names_all = get_gene_names(sig_genes, gene_symbols)

    # Creazione del DataFrame dei risultati
    df_out = pd.DataFrame({
        'GeneID': list(sig_genes),
        'GeneSymbol': [gene_symbols.loc[g] for g in sig_genes],
        'P_value': pv_tt[sig_mask],
        'Expression': ['Under' if g in under else 'Over' for g in sig_genes]
    })
    df_out.to_csv(csv_filename, index=False)

    print(f"Con soglia {alpha} attraverso il t-test sono stati individuati {len(gene_names_under)} geni sotto-espressi")
    print(gene_names_under)
    print(f"Con soglia {alpha} attraverso il t-test sono stati individuati {len(gene_names_over)} geni sovra-espressi")
    print(gene_names_over)
    print(f"Con soglia {alpha} attraverso il t-test sono stati individuati {len(gene_names_all)} geni significativi")
    print(gene_names_all)
    return gene_names_under, gene_names_over, gene_names_all

def run_utest(G1, G2, gene_symbols, means, alpha=0.012, csv_filename='utest_results.csv'):
    print("=============================== U-TEST ===============================")
    stat, pv_ut = sc.stats.mannwhitneyu(G1, G2)
    sig_mask = pv_ut < alpha
    sig_genes = G1.columns[sig_mask]
    sorted_idx = np.argsort(pv_ut[sig_mask])
    sig_genes = sig_genes[sorted_idx]
    under = sig_genes[means[sig_genes].iloc[0] < means[sig_genes].iloc[1]]
    over = sig_genes[means[sig_genes].iloc[0] >= means[sig_genes].iloc[1]]
    gene_names_under = get_gene_names(under, gene_symbols)
    gene_names_over = get_gene_names(over, gene_symbols)
    gene_names_all = get_gene_names(sig_genes, gene_symbols)

    df_out = pd.DataFrame({
        'GeneID': list(sig_genes),
        'GeneSymbol': [gene_symbols.loc[g] for g in sig_genes],
        'P_value': pv_ut[sig_mask],
        'Expression': ['Under' if g in under else 'Over' for g in sig_genes]
    })
    df_out.to_csv(csv_filename, index=False)

    print(f"Con soglia {alpha} attraverso l'u-test sono stati individuati {len(gene_names_under)} geni sotto-espressi")
    print(gene_names_under)
    print(f"Con soglia {alpha} attraverso l'u-test sono stati individuati {len(gene_names_over)} geni sovra-espressi")
    print(gene_names_over)
    print(f"Con soglia {alpha} attraverso l'u-test sono stati individuati {len(gene_names_all)} geni significativi")
    print(gene_names_all)
    return gene_names_under, gene_names_over, gene_names_all

def run_mutual_information_test(G1, G2, gene_symbols, means, thres=0.7, csv_filename='mutual_information_results.csv'):
    print("========================= MUTUAL INFORMATION =========================")
    scores = mutual_information(G1, G2, thres)
    sig_mask = scores >= thres
    sig_genes = G1.columns[sig_mask]
    sorted_idx = np.argsort(-scores[sig_mask])
    sig_genes = sig_genes[sorted_idx]
    under = sig_genes[means[sig_genes].iloc[0] < means[sig_genes].iloc[1]]
    over = sig_genes[means[sig_genes].iloc[0] >= means[sig_genes].iloc[1]]
    gene_names_under = get_gene_names(under, gene_symbols)
    gene_names_over = get_gene_names(over, gene_symbols)
    gene_names_all = get_gene_names(sig_genes, gene_symbols)

    df_out = pd.DataFrame({
        'GeneID': list(sig_genes),
        'GeneSymbol': [gene_symbols.loc[g] for g in sig_genes],
        'Score': scores[sig_mask],
        'Expression': ['Under' if g in under else 'Over' for g in sig_genes]
    })
    df_out.to_csv(csv_filename, index=False)

    print(f"Con soglia {thres} attraverso la mutua informazione sono stati individuati {len(gene_names_under)} geni sotto-espressi")
    print(gene_names_under)
    print(f"Con soglia {thres} attraverso la mutua informazione sono stati individuati {len(gene_names_over)} geni sovra-espressi")
    print(gene_names_over)
    print(f"Con soglia {thres} attraverso la mutua informazione sono stati individuati {len(gene_names_all)} geni significativi")
    print(gene_names_all)
    return gene_names_under, gene_names_over, gene_names_all

def run_sam_test(G1, G2, gene_symbols, means, alpha , csv_filename='sam_test_results.csv'):
    print("============================== SAM TEST ==============================")
    pv_st = sam_test(G1, G2)
    pv_st = sc.stats.false_discovery_control(pv_st)
    sig_mask = pv_st < alpha
    sig_genes = G1.columns[sig_mask]
    sorted_idx = np.argsort(pv_st[sig_mask])
    sig_genes = sig_genes[sorted_idx]
    under = sig_genes[means[sig_genes].iloc[0] < means[sig_genes].iloc[1]]
    over = sig_genes[means[sig_genes].iloc[0] >= means[sig_genes].iloc[1]]
    gene_names_under = get_gene_names(under, gene_symbols)
    gene_names_over = get_gene_names(over, gene_symbols)
    gene_names_all = get_gene_names(sig_genes, gene_symbols)

    df_out = pd.DataFrame({
        'GeneID': list(sig_genes),
        'GeneSymbol': [gene_symbols.loc[g] for g in sig_genes],
        'P_value': pv_st[sig_mask],
        'Expression': ['Under' if g in under else 'Over' for g in sig_genes]
    })
    df_out.to_csv(csv_filename, index=False)

    print(f"Con soglia {alpha} attraverso il sam test sono stati individuati {len(gene_names_under)} geni sotto-espressi")
    print(gene_names_under)
    print(f"Con soglia {alpha} attraverso il sam test sono stati individuati {len(gene_names_over)} geni sovra-espressi")
    print(gene_names_over)
    print(f"Con soglia {alpha} attraverso il sam test sono stati individuati {len(gene_names_all)} geni significativi")
    print(gene_names_all)
    return gene_names_under, gene_names_over, gene_names_all

def run_volcano_plot(df, gene_symbols, pv_vp, csv_filename='volcano_plot_results.csv', thres=[1, 1.5, -1, 1.7]):
    print("============================ VOLCANO PLOT ============================")
    # Utilizza i p-value (ad esempio quelli del t-test) e il rapporto tra medie
    means = df.groupby('CLASS').mean()  # assumendo DUE gruppi
    dm = means.iloc[0] / means.iloc[1]
    log10pv = -1 * np.log10(pv_vp)
    log2fc = np.log2(dm)
    down_idx, up_idx = volcano_plot(log2fc, log10pv, thres=thres)
    gene_names_under = get_gene_names(down_idx, gene_symbols)
    gene_names_over = get_gene_names(up_idx, gene_symbols)

    # Creazione di un DataFrame con i geni sotto- e sovra-espressi
    df_under = pd.DataFrame({
        'GeneID': list(down_idx),
        'GeneSymbol': [gene_symbols.loc[g] for g in down_idx],
        'Expression': 'Under'
    })
    df_over = pd.DataFrame({
        'GeneID': list(up_idx),
        'GeneSymbol': [gene_symbols.loc[g] for g in up_idx],
        'Expression': 'Over'
    })
    df_volcano = pd.concat([df_under, df_over], ignore_index=True)
    df_volcano.to_csv(csv_filename, index=False)

    print(f"Con le soglie specificate attraverso il volcano plot sono stati individuati {len(gene_names_under)} geni sotto-espressi")
    print(gene_names_under)
    print(f"Con le soglie specificate attraverso il volcano plot sono stati individuati {len(gene_names_over)} geni sovra-espressi")
    print(gene_names_over)
    return gene_names_under, gene_names_over


if __name__ == "__main__":
    
    df, gse, gene_symbols = get_series('GSE10072', gene_name_col = 'Gene Symbol') # 
    # Lettura di righe e colonne del data frame:
    ns, ng = df.shape

    df.iloc[ :53 , ng - 1] = 'Tumore' #  
    df.iloc [54:, ng -1 ] = 'Normale' # 
    # Separazione dei gruppi
    G1 = df.loc[df.CLASS == "Normale", df.columns != "CLASS"]#nornale
    G2= df.loc[df.CLASS == "Tumore", df.columns != "CLASS"] #tumore

    # Calcolo delle medie per ogni gene nei due gruppi
    means = df.groupby('CLASS').mean()

    '''
    # Esecuzione del t-test e salvataggio dei risultati in CSV
    gene_names_tt_under, gene_names_tt_over, gene_names_tt_all = run_ttest(
        G1, G2, gene_symbols, means, alpha= 0.2, csv_filename='ttest_results.csv'  #73
    )
    
    # Esecuzione dell'u-test e salvataggio dei risultati in CSV
    gene_names_ut_under, gene_names_ut_over, gene_names_ut_all = run_utest(
        G1, G2, gene_symbols, means, alpha=0.001, csv_filename='utest_results.csv'
    ) 
    
    # Esecuzione del test di mutual information e salvataggio dei risultati in CSV
    gene_names_mi_under, gene_names_mi_over, gene_names_mi_all = run_mutual_information_test(
    G1, G2, gene_symbols, means, thres=0.15, csv_filename='mutual_information_results.csv'
    )
    
    
    # Esecuzione del SAM test e salvataggio dei risultati in CSV
    gene_names_st_under, gene_names_st_over, gene_names_st_all = run_sam_test(
    G1, G2, gene_symbols, means, alpha = 0.05 , csv_filename='sam_test_results.csv'
    )    
    
    # Per il volcano plot, ad esempio, utilizziamo i p-value del t-test
    # (puoi scegliere quelli che preferisci)
   
    stat, pv_tt = sc.stats.ttest_ind(G1, G2)
    pv_tt = sc.stats.false_discovery_control(pv_tt)
    gene_names_vp_under, gene_names_vp_over = run_volcano_plot(
    df, gene_symbols, pv_tt, csv_filename='volcano_plot_results.csv', thres=[1,1.5,-1,1.7])
    print("============================ VOLCANO PLOT ============================")'''
    print("=============================== T-TEST ===============================")
    stat, pv_tt = sc.stats.ttest_ind(G1, G2)
    # disambiguazione, qualora troppi valori uguali
    # pv_tt = disambigua(pv_tt, df)
    # correzione del p-value (OPZIONALE):
    pv_tt = sc.stats.false_discovery_control(pv_tt)
    # scelta della soglia di significatività
    alpha = 0.2
    # selezione delle colonne con p-value inferiore alla soglia:
    col_test_tt = G1.columns[pv_tt < alpha]
    # geni associati ordinati per p-value:
    ids_sorted = np.argsort(pv_tt[pv_tt < alpha])
    col_test_tt = col_test_tt[ids_sorted]
    # separo geni sotto-espressi e geni sovra espressi, SOLO PER DUE GRUPPI
    gene_names_tt_ste = get_gene_names(col_test_tt[means[col_test_tt].iloc[0] < means[col_test_tt].iloc[1]], gene_symbols)
    gene_names_tt_sve = get_gene_names(col_test_tt[means[col_test_tt].iloc[0] >= means[col_test_tt].iloc[1]], gene_symbols)
    print(f"Con soglia {alpha} attraverso il t-test sono stati individuati {len(gene_names_tt_ste)} geni sotto-espressi")
    print(gene_names_tt_ste)
    print(f"Con soglia {alpha} attraverso il t-test sono stati individuati {len(gene_names_tt_sve)} geni sovra-espressi")
    print(gene_names_tt_sve)
    # geni complessivi, ANCHE PER MOLTI GRUPPI
    gene_names_tt = get_gene_names(col_test_tt, gene_symbols)
    print(f"Con soglia {alpha} attraverso il t-test sono stati individuati {len(gene_names_tt)} geni significativi")
    print(gene_names_tt)
    print("======================================================================")
    print("============================ VOLCANO PLOT ============================")

    ## AD ESEMPIO, PER OTTENRE IL VOLCANO PLOT USANDO IL RISULTATO DEL T-TEST
    ## IMPOSTO I P-VALUE DA USARE PER I VOLCANO PLOT A QUELLI DEL T-TEST
    pv_vp = pv_tt

    # calcolo il rapporto tra le medie:
    means = df.groupby('CLASS').mean() # assumendo DUE gruppi
    dm = means.iloc[0] / means.iloc[1]
    # calcolo il negativo del log10 del pvalue
    log10pv = -1*np.log10(pv_vp)
    # calcolo il log2 del rapporto tra le medie
    log2fc = np.log2(dm)
    #
    # disegno il grafico:
    # thres = [x1, x2, y] è la posizione desiderata delle rette che delimitano le regioni di interesse
    down_vp, up_vp = volcano_plot(log2fc, log10pv, thres = [1,1.5,-1,1.7]) # y1,y2,x1,x2
    #
    #nomi dei geni sotto espressi:
    gene_names_vp_ste = get_gene_names(down_vp, gene_symbols)
    print(f"Con le soglie specificate attraverso il volcano plot sono stati individuati {len(gene_names_vp_ste)} geni sotto-espressi")
    print(gene_names_vp_ste)
    #nomi dei geni sovra espressi:
    gene_names_vp_sve = get_gene_names(up_vp, gene_symbols)
    print(f"Con le soglie specificate attraverso il volcano plot sono stati individuati {len(gene_names_vp_sve)} geni sovra-espressi")
    print(gene_names_vp_sve)
    print("======================================================================")


   