import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def prod_ratio_vars(df):
    # how to fillna?
    #     df_copy = df.fillna(0)
    #     print(df_copy.oancf)
    df_copy = df.copy()
    # liquidity ratios
    df_copy['current'] = df_copy.act / df_copy.lct
    df_copy['quick'] = (df_copy.che + df_copy.rect) / df_copy.lct
    df_copy['cash'] = df_copy.che / df_copy.lct

    # solvency ratios
    df_copy['debt2asset'] = df_copy.dltt / df_copy['at']
    df_copy['debt2capital'] = df_copy.dltt / (df_copy.dltt + df_copy.seq)
    df_copy['debt2equity'] = df_copy.dltt / df_copy.seq
    df_copy['avg_at'] = (df_copy.groupby('gvkey').at.shift(1) + df_copy['at']) / 2
    df_copy['financial_lev'] = df_copy.avg_at / df_copy.seq

    # df_copy['interest_cov'] = df_copy.ebit / df_copy.intpn

    # profitability ratios
    df_copy['gross_profit_margin'] = df_copy.gp / df_copy.revt
    df_copy['pretax_margin'] = df_copy.ebitda / df_copy.revt
    df_copy['net_profit_margin'] = df_copy.ni / df_copy.revt

    # df_copy['operating_ROA'] = df_copy.opiti / df_copy.avg_at
    df_copy['ROA'] = df_copy.ni / df_copy.avg_at
    df_copy['ROTC'] = df_copy.ebit / (df_copy.dltt + df_copy.seq)
    df_copy['avg_seq'] = (df_copy.groupby('gvkey').at.shift(1) + df_copy['seq']) / 2
    df_copy['ROE'] = df_copy.ni / df_copy.avg_seq
    df_copy['ROCE'] = (df_copy.ni - df_copy.dvp) / \
        ((df_copy.groupby('gvkey').ceq.shift(1) + df_copy['ceq']) / 2)

    # valuation ratios
    df_copy['p2e'] = df_copy.prcc_f / df_copy.epspi
    df_copy['p2bv'] = df_copy.prcc_f / df_copy.bkvlps

    # credit ratios
    df_copy['ebit2interest'] = df_copy.ebit / (df_copy.xint - df_copy.idit)
    df_copy['ebitda2interest'] = df_copy.ebitda / (df_copy.xint - df_copy.idit)
    df_copy['ROC'] = df_copy.ebit / (df_copy.seq + df_copy.tdc + df_copy.dltt)
    df_copy['cash_flow2debt'] = (df_copy.oancf - df_copy.capx - df_copy.dvpd) / df_copy.dltt

    # leverage ratios
#     df_copy['debt2asset'] = df_copy.dltt / df_copy['at']
#     df_copy['debt2capital'] = df_copy.dltt / (df_copy.dltt + df_copy.seq)
#     df_copy['debt2equity'] = df_copy.dltt / df_copy.seq
    # df_copy['financial_lev']

    # performance ratios
    df_copy['cash_flow2rev'] = df_copy.oancf / df_copy.revt
    df_copy['CROA'] = df_copy.oancf / df_copy.avg_at
    df_copy['CROE'] = df_copy.oancf / df_copy.avg_seq
    # df_copy['dividend_pay'] = df_copy.oancf / df_copy.dvpd
    # df_copy['inv_fin'] = df_copy.oancf / (df_copy.fincf + df_copy.ivncf)
    df_copy['debt_cov'] = df_copy.oancf / df_copy.dltt
    # df_copy['interest_cov'] = (df_copy.oancf + df_copy.intpn + df_copy.txpd) / df_copy.intpn

    # dechow
    # accruals quality related variables
    df_copy['WC_acc'] = (((df_copy.act - df_copy.groupby('gvkey').act.shift(1)) - (df_copy.che - df_copy.groupby('gvkey').che.shift(1)))
                         - ((df_copy.lct - df_copy.groupby('gvkey').lct.shift(1)) - (df_copy.dlc - df_copy.groupby('gvkey').dlc.shift(1))
                            - (df_copy.txp - df_copy.groupby('gvkey').act.shift(1)))) / df_copy.avg_at
    df_copy['WC'] = (df_copy.act - df_copy.che) - (df_copy.lct - df_copy.dlc)
    df_copy['NCO'] = (df_copy['at'] - df_copy.act - df_copy.ivaeq - df_copy.ivao) - \
        (df_copy['lt'] - df_copy.lct - df_copy.dltt)
    df_copy['FIN'] = (df_copy.ivst + df_copy.ivpt) - (df_copy.dltt + df_copy.dlc + df_copy.pstk)
    df_copy['rsst_acc'] = ((df_copy.WC - df_copy.groupby('gvkey').WC.shift(1)) + (df_copy.NCO - df_copy.groupby('gvkey').NCO.shift(1))
                           + (df_copy.FIN - df_copy.groupby('gvkey').FIN.shift(1))) / df_copy.avg_at
    df_copy['ch_res'] = (df_copy.rect - df_copy.groupby('gvkey').rect.shift(1)) / df_copy.avg_at
    df_copy['ch_inv'] = (df_copy.invt - df_copy.groupby('gvkey').invt.shift(1)) / df_copy.avg_at
    df_copy['soft_assets'] = (df_copy['at'] - df_copy.ppegt - df_copy.che) / df_copy['at']

    # performance variables
    df_copy['cs'] = df_copy.sale - (df_copy.rect - df_copy.groupby('gvkey').rect.shift(1))
    df_copy['ch_cs'] = (df_copy.cs - df_copy.groupby('gvkey').cs.shift(1)) / df_copy.cs
    df_copy['cm'] = 1 - (df_copy.cogs - (df_copy.invt - df_copy.groupby('gvkey').invt.shift(1))
                         + (df_copy.ap - df_copy.groupby('gvkey').ap.shift(1))) / df_copy.cs
    df_copy['ch_cm'] = (df_copy.cm - df_copy.groupby('gvkey').cm.shift(1))
    df_copy['ch_roa'] = df_copy.ebit / df_copy.avg_at - \
        df_copy.groupby('gvkey').ebit.shift(1) / df_copy.groupby('gvkey').avg_at.shift(1)
    df_copy['ch_fcf'] = ((df_copy.ebit - df_copy.rsst_acc) - (df_copy.groupby('gvkey').ebit.shift(1) -
                                                              df_copy.groupby('gvkey').rsst_acc.shift(1))) / df_copy.avg_at
    # df_copy['tax'] = df_copy.tdc / df_copy.groupby('gvkey').at.shift(1)
    # nonfinancial variables
    df_copy['ch_emp'] = (df_copy.emp - df_copy.groupby('gvkey').emp.shift(1)) / df_copy.emp \
        - (df_copy['at'] - df_copy.groupby('gvkey')['at'].shift(1)) / df_copy['at']
#     df_copy['ch_bocklog'] = (df_copy.ob - df_copy.groupby('gvkey').ob.shift(1)) / df_copy.ob \
#         - (df_copy.sale - df_copy.groupby('gvkey').sale.shift(1)) / df_copy.sale
#     print(df_copy.oancf)
    # market-related incentives
    df_copy['exfin'] = (df_copy.oancf - (df_copy.capx + df_copy.groupby('gvkey').capx.shift(1)
                                         + df_copy.groupby('gvkey').capx.shift(2)) / 3) / df_copy.act
    mask = df_copy.exfin.isnull()
    df_copy['exfin'] = df_copy.exfin < -0.5
    df_copy.loc[mask, 'exfin'] = np.nan
    df_copy['bm'] = df_copy.seq / df_copy.mkvalt
    df_copy['ep'] = df_copy.ebit / df_copy.mkvalt

    # return df_copy.replace([np.inf, -np.inf, 0], np.nan)
    return df_copy.replace([np.inf, -np.inf, 0], np.nan)


def main():
    # ctr_vars = ['au', 'curncd', 'fic', 'idbflag', 'exchg', 'fyr', 'naics',
    #               'sic', 'stko', 'fyear']
    ctr_vars = ['fyear', 'sic', 'curncd']
    # restatement_vars = ['rea', 'acchg', 'aqc']
    imp1_vars = ['acominc', 'act', 'at', 'auop', 'ceq', 'che', 'ci',
                 'cogs', 'dltt', 'dv', 'ebit', 'ebitda', 'epspi', 'fincf',
                 'gdwl', 'gp', 'idit', 'invt', 'ivncf', 'lct', 'llrci',
                 'llwoci', 'lt', 'mkvalt', 'ni', 'nopio', 'oancf', 'pi',
                 'pll', 'ppegt', 'ppenb', 'prcc_f', 'prch_f', 'prcl_f',
                 'rdip', 're', 'rect', 'reuna', 'revt', 'rmum', 'sale',
                 'seq', 'tdc', 'teq', 'tfva', 'txc', 'txt', 'wcap', 'wda',
                 'xad', 'xagt', 'xdp', 'xi', 'xint', 'xopr', 'xpr', 'xrd',
                 'xsga'
                 ]
    other_vars = ['opiti', 'csho', 'dvp', 'dvpd', 'txpd', 'intpn', 'txp', 'ivst',
                  'ivpt', 'dlc', 'pstk', 'emp', 'ob', 'capx', 'bkvlps', 'ivaeq', 'ivao', 'ap']
    # imp2_vars = ['aco', 'amgw', 'ao', 'ap', 'arc', 'auopic', 'bkvlps',
    #              'capx', 'cdvc', 'cga', 'chech']
    # imp3_vars =
    # imp4_vars =

    all_vars = ['gvkey'] + ctr_vars + imp1_vars + other_vars
    df = pd.read_csv("data/annual_compustat.zip", usecols=all_vars)
    df = df.groupby(['gvkey'] + ctr_vars)[imp1_vars + other_vars].mean().reset_index()
    df.sic = df.sic // 100
    # df.fyear = df.fyear.astype(int)

    # produce misstated column
    dgls = pd.read_excel('data/DGLS_20160930_D.xlsx', sheet_name='ann')
    match = pd.read_csv('data/match_gvkey_cik.csv')
    df_m = df.merge(match[['gvkey', 'cik']].drop_duplicates(), on=['gvkey'], how='left').dropna(subset=['cik']) \
        .merge(dgls[['CIK', 'YEARA']].drop_duplicates(), left_on=['cik', 'fyear'], right_on=['CIK', 'YEARA'], how='left')
    df_m = df_m[(df_m.curncd == 'USD') & (df_m.fyear <= 2013)]
    df_m['misstated'] = np.where(df_m['YEARA'].notnull(), True, False)
    # df_m.fyear = df_m.fyear.astype(int).astype(str) + '-12-31'
    df_m = df_m.drop(['curncd', 'cik', 'CIK', 'YEARA'], axis=1).fillna(0)
    # df_m.to_csv('annual_compustat_preprocessed.csv', index=False)

    df_ratios = prod_ratio_vars(df_m)
    ratio_cols = ['current', 'quick', 'cash', 'debt2asset', 'debt2capital', 'debt2equity',
                  'financial_lev', 'gross_profit_margin', 'pretax_margin', 'net_profit_margin',
                  'ROA', 'ROTC', 'ROE', 'ROCE', 'p2e', 'p2bv', 'ebit2interest', 'ebitda2interest',
                  'ROC', 'cash_flow2debt', 'cash_flow2rev', 'CROA', 'CROE', 'debt_cov', 'WC_acc',
                  'rsst_acc', 'ch_res', 'ch_inv', 'soft_assets', 'ch_cs', 'ch_cm', 'ch_roa',
                  'ch_fcf', 'ch_emp', 'exfin', 'bm', 'ep']

    results = pd.read_csv('results.csv')
    df_ratios_only = df_ratios[['gvkey', 'fyear', 'sic'] +
                               ratio_cols + ['misstated', 'misstated_prob']]
    df_ratios_only = df_ratios_only[(df_ratios_only.isnull().sum(
        axis=1) / df_ratios_only.shape[1]) < 0.8]
    df_ratios_only = df_ratios_only.merge(
        results[['fyear', 'gvkey', 'pred_prob']], on=['gvkey', 'fyear'])
    df_ratios_only.to_csv('data/annual_compustat_ratios.zip', index=False)


if __name__ == '__main__':
    if os.path.isfile('data/annual_compustat_ratios.zip'):
        print("annual_compustat_ratios.zip already exists")
    else:
        main()
