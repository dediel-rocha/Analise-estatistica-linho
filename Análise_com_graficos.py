import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# FUNÇÕES ESTATÍSTICAS MANUAIS (Trata Desbalanceamento)
# ==========================================
def teste_lsd_fisher(df, val_col, group_col):
    """Teste LSD de Fisher exato para dados DESBALANCEADOS."""
    grupos = df[group_col].unique()
    n_groups = len(grupos)
    p_matrix = pd.DataFrame(np.ones((n_groups, n_groups)), index=grupos, columns=grupos)
    
    df_clean = df.dropna(subset=[val_col, group_col]).copy()
    N = len(df_clean)
    k = n_groups
    df_err = N - k
    
    group_data = [df_clean[df_clean[group_col] == g][val_col].values for g in grupos]
    variances = [np.var(g, ddof=1) if len(g) > 1 else 0 for g in group_data]
    ns = [len(g) for g in group_data]
    
    if df_err <= 0 or sum(ns) == 0:
        return p_matrix
        
    mse = sum((n - 1) * v for n, v in zip(ns, variances)) / df_err
    
    for i in range(n_groups):
        for j in range(i+1, n_groups):
            if ns[i] == 0 or ns[j] == 0: continue
            
            mean_diff = np.mean(group_data[i]) - np.mean(group_data[j])
            se = np.sqrt(mse * (1/ns[i] + 1/ns[j]))
            if se == 0: continue
            
            t_stat = abs(mean_diff) / se
            p_val = 2 * (1 - stats.t.cdf(t_stat, df_err))
            p_matrix.iloc[i, j] = p_matrix.iloc[j, i] = p_val
            
    return p_matrix

def teste_dunn(df, val_col, group_col):
    """Teste Não-Paramétrico de Dunn para Acama."""
    df_clean = df.dropna(subset=[val_col, group_col]).copy()
    grupos = df_clean[group_col].unique()
    n_groups = len(grupos)
    p_matrix = pd.DataFrame(np.ones((n_groups, n_groups)), index=grupos, columns=grupos)
    
    df_clean['rank'] = stats.rankdata(df_clean[val_col])
    N = len(df_clean)
    
    ties = df_clean[val_col].value_counts()
    tie_term = sum(t**3 - t for t in ties)
    variance = (N * (N + 1) / 12) - (tie_term / (12 * (N - 1)))
    
    rank_means = df_clean.groupby(group_col)['rank'].mean()
    ns = df_clean.groupby(group_col)['rank'].count()
    
    for i in range(n_groups):
        for j in range(i+1, n_groups):
            g1, g2 = grupos[i], grupos[j]
            diff = abs(rank_means[g1] - rank_means[g2])
            se = np.sqrt(variance * (1/ns[g1] + 1/ns[g2]))
            
            if se > 0:
                z = diff / se
                p_val = 2 * (1 - stats.norm.cdf(z))
                p_matrix.loc[g1, g2] = p_matrix.loc[g2, g1] = p_val
                
    num_comparisons = n_groups * (n_groups - 1) / 2
    p_matrix = p_matrix * num_comparisons
    p_matrix[p_matrix > 1] = 1.0
    return p_matrix

def gerar_letras(p_matrix, medias_series, alpha=0.05):
    """Gera letras a partir da matriz de p-valores."""
    grupos = p_matrix.index.values
    n = len(grupos)
    adj = np.eye(n, dtype=bool)
    
    for i in range(n):
        for j in range(i+1, n):
            if p_matrix.iloc[i, j] >= alpha:
                adj[i, j] = adj[j, i] = True
                
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]: G.add_edge(i, j)
            
    cliques = list(nx.find_cliques(G))
    medias_map = medias_series.to_dict()
    cliques.sort(key=lambda c: max([medias_map[grupos[x]] for x in c if grupos[x] in medias_map]), reverse=True)
    
    letras = [""] * n
    alfabeto = "abcdefghijklmnopqrstuvwxyz"
    for i, clq in enumerate(cliques):
        let = alfabeto[i] if i < len(alfabeto) else "z"+str(i)
        for idx in clq:
            letras[idx] += let
            
    res = {}
    for i in range(n):
        res[grupos[i]] = "".join(sorted(set(letras[i])))
    return res

# ==========================================
# 1. CARREGAMENTO DOS DADOS
# ==========================================
def carregar_dados(nome_arquivo):
    print(f"Lendo o arquivo: {nome_arquivo}...")
    try:
        df = pd.read_excel(nome_arquivo, engine='openpyxl')
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        exit()
        
    fatores = ['Gen', 'Rep', 'Ano', 'Local']
    for fat in fatores:
        df[fat] = df[fat].astype(str).str.strip()
        
    var_continuas = ['Rend', 'Altura', 'N_ramos', 'N_capsulas', 'Grao_capsula', 'Massa_1000']
    var_discreta = ['Acama']
    
    for col in var_continuas + var_discreta:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            
    df = df.dropna(subset=fatores)
    return df, var_continuas, var_discreta

# ==========================================
# 2. REMOÇÃO DE OUTLIERS (MÉTODO IQR)
# ==========================================
def remover_outliers_iqr(df, variaveis):
    print("\n--- Removendo Outliers (Método IQR) ---")
    df_clean = pd.DataFrame()
    
    for nome, grupo in df.groupby(['Gen', 'Local', 'Ano']):
        for var in variaveis:
            if grupo[var].isnull().all():
                continue
            Q1 = grupo[var].quantile(0.25)
            Q3 = grupo[var].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            
            grupo.loc[(grupo[var] < limite_inferior) | (grupo[var] > limite_superior), var] = np.nan
        df_clean = pd.concat([df_clean, grupo])
    return df_clean

# ==========================================
# 3. TABELAS DE MÉDIAS POR LOCAL
# ==========================================
def gerar_tabelas_finais(df, var_continuas, var_discreta):
    print("\n" + "="*50)
    print("GERANDO TABELAS DE MÉDIAS POR LOCAL (FISHER LSD / DUNN)")
    print("="*50)
    
    locais = df['Local'].unique()
    genotipos = df['Gen'].unique()
    todas_vars = var_continuas + var_discreta
    tabelas = {}
    
    for loc in locais:
        print(f"Processando Local: {loc}...")
        df_loc = df[df['Local'] == loc].copy()
        df_resumo = pd.DataFrame(index=genotipos)
        
        for var in todas_vars:
            df_var = df_loc.dropna(subset=[var, 'Gen', 'Ano', 'Rep'])
            
            if df_var.empty:
                df_resumo[var] = "-"
                continue
                
            medias = df_var.groupby('Gen')[var].mean()
            letras_dict = {g: "a" for g in genotipos} 
            
            if var in var_continuas:
                if df_var['Ano'].nunique() > 1:
                    modelo = ols(f'{var} ~ C(Gen) + C(Ano) + C(Rep):C(Ano)', data=df_var).fit()
                else:
                    modelo = ols(f'{var} ~ C(Gen) + C(Rep)', data=df_var).fit()
                
                try:
                    aov = sm.stats.anova_lm(modelo, typ=2)
                    p_val = aov.loc['C(Gen)', 'PR(>F)']
                    
                    if p_val < 0.05:
                        lsd_matrix = teste_lsd_fisher(df_var, var, 'Gen')
                        letras_dict = gerar_letras(lsd_matrix, medias)
                except Exception as e:
                    pass
            
            elif var in var_discreta:
                grupos_kw = [g[var].values for n, g in df_var.groupby('Gen')]
                if len(grupos_kw) > 1:
                    try:
                        _, p_kw = stats.kruskal(*grupos_kw)
                        if p_kw < 0.05:
                            dunn_matrix = teste_dunn(df_var, var, 'Gen')
                            letras_dict = gerar_letras(dunn_matrix, medias)
                    except ValueError:
                        pass
            
            coluna_formatada = []
            for g in genotipos:
                if g in medias.index:
                    texto = f"{medias[g]:.2f} {letras_dict.get(g, '')}".strip()
                else:
                    texto = "-"
                coluna_formatada.append(texto)
                
            df_resumo[var] = coluna_formatada
            
        tabelas[loc] = df_resumo
    
    nome_saida = 'Tabelas_Medias_Conjunta_Desbalanceada.xlsx'
    with pd.ExcelWriter(nome_saida) as writer:
        for loc, tabela in tabelas.items():
            if 'Rend' in tabela.columns and not tabela['Rend'].eq("-").all():
                ordem = tabela['Rend'].str.extract(r'([\d\.]+)').astype(float).sort_values(by=0, ascending=False).index
                tabela = tabela.loc[ordem]
            
            aba_nome = str(loc).replace("/", "-")[:31]
            tabela.to_excel(writer, sheet_name=aba_nome)
            
            print(f"\n--- Resumo para o Local: {loc} ---")
            try:
                print(tabela.to_markdown())
            except ImportError:
                print(tabela.to_string())
            
    print(f"\nTabelas (por local) geradas e salvas no arquivo: {nome_saida}")

# ==========================================
# 4. GRÁFICOS (CORRELAÇÃO E BOXPLOT)
# ==========================================
def analise_grafica(df, var_continuas, var_discreta):
    print("\nGerando Gráficos...")
    todas_vars = var_continuas + var_discreta
    
    # 1. Matriz de Correlação
    corr = df[todas_vars].corr(method='spearman')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Matriz de Correlação (Spearman)')
    plt.tight_layout()
    plt.savefig('correlacao_heatmap.png')
    plt.close() # Fecha a figura para não sobrepor
    print(" - Gráfico 'correlacao_heatmap.png' salvo.")
    
    # 2. Boxplot da Interação (Local x Genótipo para Rendimento)
    if 'Rend' in df.columns:
        plt.figure(figsize=(14, 6))
        # O hue='Gen' faz com que cada barra no local seja de um genótipo
        sns.boxplot(x='Local', y='Rend', hue='Gen', data=df, palette='Set3')
        plt.title('Interação Genótipo x Local para Rendimento de Grãos (kg/ha)')
        plt.ylabel('Rendimento (kg/ha)')
        plt.xlabel('Local')
        plt.xticks(rotation=45)
        # Move a legenda para fora do gráfico
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Genótipo")
        plt.tight_layout()
        plt.savefig('boxplot_interacao_rend.png')
        plt.close()
        print(" - Gráfico 'boxplot_interacao_rend.png' salvo.")

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # NOME DO NOVO ARQUIVO DE DADOS
    arquivo_excel = 'Conjunta.xlsx'
    
    df_raw, continuas, discretas = carregar_dados(arquivo_excel)
    df_limpo = remover_outliers_iqr(df_raw, continuas)
    gerar_tabelas_finais(df_limpo, continuas, discretas)
    
    # Chama a função corrigida que gera o Boxplot e a Correlação
    analise_grafica(df_limpo, continuas, discretas)
    
    print("\nAnálise concluída com sucesso! Verifique as imagens na sua pasta.")

    EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # NOME DO NOVO ARQUIVO DE DADOS
    arquivo_excel = 'Conjunta.xlsx'
    
    df_raw, continuas, discretas = carregar_dados(arquivo_excel)
    df_limpo = remover_outliers_iqr(df_raw, continuas)
    gerar_tabelas_finais(df_limpo, continuas, discretas)
    
    # Chama a função corrigida que gera o Boxplot e a Correlação
    analise_grafica(df_limpo, continuas, discretas)
    
    print("\nAnálise concluída com sucesso! Verifique as imagens na sua pasta.")