
# %%
import numpy as np
import pandas as pd
from datetime import datetime
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller


# %%
# HACK:
def wareki_parser(date):
    """
    Parameters
    ----------
    date: str

    Return
    ------
    dt: datetime.datetime
    """

    wareki = {
        'R': 2018,
        'H': 1988,
        'S': 1925,
    }

    _Y, _m, _d = date[1:].split('.')

    # dt = datetime.strptime(str(int(_Y) + wareki[date[0]]) + '/' + _m + '/' + _d, '%Y/%m/%d')
    dt = datetime.strptime(str(int(_Y) + wareki[date[0]]) + '/' + _m + '/' + _d, '%Y/%m/%d').strftime('%Y%m%d')
    return dt


# %%
def get_input_data_jgb() -> pd.DataFrame:
    df_input = pd.read_csv("jgbcm_all.csv", encoding='shift_jis', skiprows=1)
    df_input['基準日'] = df_input['基準日'].map(wareki_parser)
    df_input = df_input.replace('-', np.nan)
    df_input = df_input.dropna(how='any')
    df_input = df_input.rename(columns={'基準日': 'base_date'})
    df_input = df_input.set_index('base_date')
    df_input = df_input.astype(float)
    return df_input


# %%
# 最小二乗法の解析解を算出
def predict_simple_lstsq(X, t):
    """simple least square solution"""
    w = (np.linalg.inv(X.T @ X).T @ (X.T @ t).T).T
    return w


# %%
# インプットのファクター情報の計算
lvl_fac_fnc = lambda _: lambda _: 1
slp_fac_fnc = lambda lmd: lambda tenor: (1 - np.exp(- lmd * tenor)) / (lmd * tenor)
cvt_fac_fnc = lambda lmd: lambda tenor: (1 - np.exp(- lmd * tenor)) / (lmd * tenor) - np.exp(- lmd * tenor)

make_fnc_input_fac \
    = lambda func, lmd, tenor_list: np.matrix(np.array([x for x in map(func(lmd), tenor_list)]))

def calc_input_factor_values(lmd, tenor_list) -> tuple:
    level_fac_input = make_fnc_input_fac(lvl_fac_fnc, lmd, tenor_list)
    slope_fac_input = make_fnc_input_fac(slp_fac_fnc, lmd, tenor_list)
    corvature_fac_input = make_fnc_input_fac(cvt_fac_fnc, lmd, tenor_list)
    return (level_fac_input, slope_fac_input, corvature_fac_input)

# LSTの係数を最小二乗法で推定
def estimate_w_dns(level_input: np.matrix, 
                   slope_input: np.matrix, 
                   corvature_input: np.matrix, 
                   correct_rate_list: np.matrix) -> np.matrix:
    X = np.concatenate([level_input, slope_input, corvature_input], 0).T
    result_lstsq = predict_simple_lstsq(X, correct_rate_list)
    return np.ravel(result_lstsq) # 結果をベクトル化して返す

# 基準日ごとのLSTの係数の算出を実行
def execute_estimate_w(df_input_rate, level_input, slope_input, corvature_input):
    factor_lst = {}
    for base_date, correct_rate_list in df_input_rate.iterrows():
        factor_lst[base_date] = estimate_w_dns(level_input, slope_input, corvature_input, correct_rate_list)
    df_factor_lst = pd.DataFrame.from_dict(factor_lst, orient='index', columns=['w_L', 'w_S', 'w_C'])
    return df_factor_lst

def get_df_w_lsc(df_input, lmd, tenor_list):
    level_input, slope_input, corvature_input = calc_input_factor_values(lmd, tenor_list)
    df_w_lst = execute_estimate_w(df_input, level_input, slope_input, corvature_input)
    return df_w_lst 


# %%
# statsmodels

def get_armodel_fitted_by_statsmodels(df, target_colname, lagnum):
    # ARモデルの作成
    ar = AutoReg(df[target_colname], lags=lagnum, old_names=False)
    ar_fit = ar.fit()
    return ar_fit

def results_summary_by_sm_to_dataframe(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    stderr = results.bse
    z = results.tvalues
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "stderr": stderr,
                               "z": z,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })

    #Reordering...
    results_df = results_df[["coeff","pvals", "stderr", "z", "conf_lower","conf_higher"]]
    return results_df


# %%
# パラメータ推定
df_input = get_input_data_jgb()
tenor_list= np.array([int(col[:-1]) for col in df_input.columns.values])
df_input.columns = tenor_list

lmd = 0.206 

df_w_lsc = get_df_w_lsc(df_input, lmd, tenor_list)
df_w_lsc.to_pickle("pddf_w_lsc_with_" + str(lmd) + ".pkl")

adf_L = adfuller(df_w_lsc["w_L"])
print('ADF Statistic: %f' % adf_L[0])
print('p-value: %f' % adf_L[1])
adf_S = adfuller(df_w_lsc["w_S"])
print('ADF Statistic: %f' % adf_S[0])
print('p-value: %f' % adf_S[1])
adf_C = adfuller(df_w_lsc["w_C"])
print('ADF Statistic: %f' % adf_C[0])
print('p-value: %f' % adf_C[1])

ar_L_fitted_model = get_armodel_fitted_by_statsmodels(df_w_lsc, "w_L", 1)
ar_S_fitted_model = get_armodel_fitted_by_statsmodels(df_w_lsc, "w_S", 1)
ar_C_fitted_model = get_armodel_fitted_by_statsmodels(df_w_lsc, "w_C", 1)

res_summary_ar_L_model = results_summary_by_sm_to_dataframe(ar_L_fitted_model)
res_summary_ar_S_model = results_summary_by_sm_to_dataframe(ar_S_fitted_model)
res_summary_ar_C_model = results_summary_by_sm_to_dataframe(ar_C_fitted_model)
ar_L_fitted_model.summary()
ar_S_fitted_model.summary()
ar_C_fitted_model.summary()

# 定数項と係数
ar_L_const = res_summary_ar_L_model['coeff'].iloc[0]
ar_S_const = res_summary_ar_S_model['coeff'].iloc[0]
ar_C_const = res_summary_ar_C_model['coeff'].iloc[0]
ar_L_w = res_summary_ar_L_model['coeff'].iloc[1]
ar_S_w = res_summary_ar_S_model['coeff'].iloc[1]
ar_C_w = res_summary_ar_C_model['coeff'].iloc[1]

# 乱数生成時に使用するsigmaを算出

# ノイズ項を含めず、定数項と係数でLSCを計算
ar_test_ex_noise = {}
for idx, row in df_w_lsc.iterrows():
    l = ar_L_const + ar_L_w * row["w_L"]    # t+1 の Lを計算
    s = ar_S_const + ar_S_w * row["w_S"]    # t+1 の Sを計算
    c = ar_C_const + ar_C_w * row["w_C"]    # t+1 の Cを計算
    ar_test_ex_noise[idx] = {"test_w_l":l, "test_w_s":s, "test_w_c":c}

df_test_w_lsc =  pd.DataFrame.from_dict(ar_test_ex_noise, orient='index')
df_test_w_lsc = df_test_w_lsc.iloc[:-1, :]  # 実データのLSC（正解）との比較diffのため、余分な最終行を除外
df_test_w_lsc = df_test_w_lsc.reset_index()

# shift
# 実データのLSC（正解）との比較diffのため、t+1以降の値を抽出
df_shift_one = df_w_lsc.copy().iloc[1:, :]
df_shift_one = df_shift_one.reset_index()
df_shift_one = df_shift_one[["w_L", "w_S", "w_C"]]
# 比較diffのため、実データのLSC（正解）列を追加
df_test_w_lsc["w_L"] = df_shift_one["w_L"]
df_test_w_lsc["w_S"] = df_shift_one["w_S"]
df_test_w_lsc["w_C"] = df_shift_one["w_C"]
# 比較diffのため、
df_test_w_lsc["diff_l"] = df_test_w_lsc["test_w_l"]-df_test_w_lsc["w_L"]
df_test_w_lsc["diff_s"] = df_test_w_lsc["test_w_s"]-df_test_w_lsc["w_S"]
df_test_w_lsc["diff_c"] = df_test_w_lsc["test_w_c"]-df_test_w_lsc["w_C"]

lagnum = 1  # AR(1)なので１
diff_2_sum_L = np.sum(df_test_w_lsc["diff_l"].values ** 2)
sigma_L_2 = (1/(len(df_test_w_lsc) - 2 - lagnum)) * diff_2_sum_L
sigma_L = np.sqrt(sigma_L_2)

diff_2_sum_S = np.sum(df_test_w_lsc["diff_s"].values ** 2)
sigma_S_2 = (1/(len(df_test_w_lsc) - 2 - lagnum)) * diff_2_sum_S
sigma_S = np.sqrt(sigma_S_2)

diff_2_sum_C = np.sum(df_test_w_lsc["diff_c"].values ** 2)
sigma_C_2 = (1/(len(df_test_w_lsc) - 2 - lagnum)) * diff_2_sum_C
sigma_C = np.sqrt(sigma_C_2)


# コレスキー分解（乱数生成用）
r_in = np.matrix(df_w_lsc.corr().values)
chol = np.linalg.cholesky(r_in)


print('--------定数項------')
print(ar_L_const)
print(ar_S_const)
print(ar_C_const)
print('-------------------')
print('--------係数------')
print(ar_L_w)
print(ar_S_w)
print(ar_C_w)
print('------------------')
print('--------sigma------')
print(sigma_L)
print(sigma_S)
print(sigma_C)
print('-------------------')
print('--------相関係数------')
print(df_w_lsc.corr().values)
print('------------------')

print('-------------コレスキー分解-----------')
print(chol)
print('-------------------------------------')


# %%
# lambda探索

def calc_rate(df_factor_lst, tenor_list, level_input, slope_input, corvature_input) -> pd.DataFrame:
    sim_rate_rows = []
    for _, fac_row in df_factor_lst.iterrows():
        row_dict = {}
        for tenor_index in range(0, len(tenor_list)):
            tenor = tenor_list[tenor_index]
            row_dict[tenor] = fac_row["w_L"] * level_input[0, tenor_index] \
                                + fac_row["w_S"] * slope_input[0, tenor_index] \
                                + fac_row["w_C"] * corvature_input[0, tenor_index]
        sim_rate_rows.append(row_dict)

    df_ = pd.DataFrame(sim_rate_rows, columns=tenor_list, index=df_factor_lst.index)
    return df_


def sim_rate_approximate_lsc(df_input, lmd, tenor_list) -> pd.DataFrame:

    level_input, slope_input, corvature_input = calc_input_factor_values(lmd, tenor_list)

    df_w_lst = execute_estimate_w(df_input, level_input, slope_input, corvature_input)

    df_sim = calc_rate(df_w_lst, tenor_list, level_input, slope_input, corvature_input)
    return df_sim


df_input = get_input_data_jgb()
tenor_list = np.array([int(col[:-1]) for col in df_input.columns.values])
df_input.columns = tenor_list
lmd_list = [round(l*0.001, 3) for l in range(0, 1000)]
lmd_list = lmd_list[1:]  # 先頭のゼロは除外

# 直近のデータのみ
df_input = df_input[df_input.index >= '20230401']

res_mean_squares = {}
for lmd in lmd_list:
    df_sim = sim_rate_approximate_lsc(df_input, lmd, tenor_list)
    df_diff = (df_input - df_sim)
    res_mean_square = df_diff.map(lambda x: x**2).sum().sum()
    res_mean_squares[lmd] = res_mean_square

pd.DataFrame.from_dict(res_mean_squares, orient='index').to_csv("test_lmd_search_jgb_2years.csv")
pd.DataFrame.from_dict(res_mean_squares, orient='index')
