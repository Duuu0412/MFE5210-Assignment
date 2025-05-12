#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 19:41:00 2025

@author: dujiayu
"""
#%%
import pandas as pd
import numpy as np
import os

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

from typing import Tuple
#%% 数据导入和预处理
os.chdir(r'/Users/dujiayu/Desktop/mfe/mfe5210/assignments/行业有效量价因子与行业轮动策略/STK_IndustryClassAnl')
df_Ind = pd.read_csv('STK_IndustryClassAnl.csv', low_memory = False)
df_Ind['Symbol'] = [str(x).zfill(6) for x in df_Ind.Symbol] #str(x).zfill(6) 将股票代码转为字符串并自动补充为6位，不足6位在其左侧添0
df_Ind = df_Ind.rename(columns = {'Symbol' : 'Stkcd'})
df_Ind = df_Ind[df_Ind.IndustryClassificationID == 'P0217'] #P0217 中信证券行业分类 根据研报使用中信证券行业分类
df_Ind = df_Ind[['IndustryCode1','IndustryName1']]
### 按研报要求剔除综合和综合金融两个行业分类
df_Ind = df_Ind[df_Ind.IndustryName1 != '综合']
df_Ind = df_Ind[df_Ind.IndustryName1 != '综合金融']

# 确定一级行业
df_Ind = df_Ind.drop_duplicates()

# 匹配行业指数 
os.chdir(r'/Users/dujiayu/Desktop/mfe/mfe5210/assignments/行业有效量价因子与行业轮动策略')
df_Fund = pd.read_csv('FUND_MKT_Quotation.csv')
df_Fund = df_Fund.rename(columns = {'ReturnDaily':'Return'})
## 计算换手率
df_Fund['Turnover'] = df_Fund['Amount'] / df_Fund['MarketValue']

## 根据数据完整性保留有效时间段的样本（原文为2010-2022.7.31）
df_Fund = df_Fund[df_Fund.TradingDate >= '2018-11-23']

# 提取相应数据
df_FVolume = df_Fund.pivot(index = 'TradingDate',columns = 'Symbol',values = 'Volume')
# df_FVolume = df_FVolume.fillna(0)
df_FAmount = df_Fund.pivot(index = 'TradingDate',columns = 'Symbol',values = 'Amount')
# df_FAmount = df_FAmount.fillna(0)
df_FOpen = df_Fund.pivot(index = 'TradingDate',columns = 'Symbol',values = 'OpenPrice')
# df_FOpen = df_FOpen.fillna(0)
df_FClose = df_Fund.pivot(index = 'TradingDate',columns = 'Symbol',values = 'ClosePrice')
#df_FClose = df_FClose.fillna(0)
df_FHigh = df_Fund.pivot(index = 'TradingDate',columns = 'Symbol',values = 'HighPrice')
#df_FHigh = df_FHigh.fillna(0)
df_FLow = df_Fund.pivot(index = 'TradingDate',columns = 'Symbol',values = 'LowPrice')
#df_FLow = df_FLow.fillna(0)
df_FTurnover = df_Fund.pivot(index = 'TradingDate',columns = 'Symbol',values = 'Turnover')
#df_FTurnover = df_FTurnover.fillna(0)

#%% 因子构建及IC值
# 定义因子IC值计算函数
def Factor_IC(df, factor_name, base_ret):
    factor = df[factor_name]
    corr = factor.corr(base_ret)
    IC = np.sqrt(factor.shape[0])*corr
    return IC

## 因子构建及IC值检验
df_Factor = pd.DataFrame()
df_IC = pd.DataFrame()
df_Factor = df_Fund[['TradingDate','Symbol','Return']]

#->1. 动量
## 简单动量 过去20日累计收益
df_Mom = df_Fund.pivot(index = 'TradingDate', columns = 'Symbol', values = 'Return')
df_Mom = df_Mom.fillna(0)
df_Mom = df_Mom * 0.01 + 1
df_Mom = df_Mom.rolling(window = 20).apply(np.prod, raw = True) - 1

##->1.1 二阶动量
## 差分
df_Mom_diff = df_Mom - df_Mom.shift(1) 

## 计算指数加权移动的平均值
df_Mom_diff = df_Mom_diff.ewm(alpha = 0.3, adjust = False).mean() 
df_Mom_diff = df_Mom_diff.shift(1).unstack().dropna().reset_index().rename(columns={0: 'Mom_diff'})

## 归一化
df_Mom_diff['Mom_diff'] = (df_Mom_diff.groupby(['Symbol'])['Mom_diff']
                           .apply(lambda x:(x - np.min(x)) / (np.max(x) - np.min(x)))
                           .reset_index(level=0, drop=True))

## 计算因子IC值
df_IC['Mom_diff'] = df_Mom_diff.groupby(['TradingDate']).agg(Factor_IC,'Mom_diff',df_Factor['Return'])['Mom_diff']

## 绘制因子IC值时序图
plt.figure(figsize=(10, 5))  # 创建图形窗口
df_IC['Mom_diff'].plot(title='IC value of the second-order momentum', color = '#4C72B0')
plt.xlabel('Date')
plt.ylabel('IC Value')
plt.show()

## 数据合并
df_Factor = pd.merge(df_Factor,df_Mom_diff,on = ['Symbol', 'TradingDate'], how = 'left')

##->1.2 动量期限差 
## 长期动量减去短期动量。该因子值越高可以理解为在长期向上趋势明显的行业中剔除了近日较为拥挤的行业
df_Mom_term = df_FClose.copy()
df_Mom_term = ((df_Mom_term - df_Mom_term.shift(40)) / df_Mom_term.shift(40)
               - (df_Mom_term - df_Mom_term.shift(10)) / df_Mom_term.shift(10))
df_Mom_term = df_Mom_term.replace([np.inf,-np.inf],np.nan)
df_Mom_term = df_Mom_term.shift(1).unstack().dropna().reset_index().rename(columns={0: 'Mom_term'})
df_Mom_term['Mom_term'] = (df_Mom_term.groupby(['Symbol'])['Mom_term']
                           .apply(lambda x:(x - np.min(x)) / (np.max(x) - np.min(x)))
                           .reset_index(level=0, drop=True))

## 计算因子IC值
df_IC['Mom_term'] = df_Mom_term.groupby(['TradingDate']).agg(Factor_IC,'Mom_term',df_Factor['Return'])['Mom_term']

## 绘制因子IC值时序图
plt.figure(figsize=(10, 5))  # 创建图形窗口
df_IC['Mom_term'].plot(title='IC value of momentum term difference', color = '#4C72B0')
plt.xlabel('Date')
plt.ylabel('IC Value')
plt.show()

## 数据合并
df_Factor = pd.merge(df_Factor,df_Mom_term,on = ['Symbol', 'TradingDate'], how = 'left')

#%%
#->2. 交易波动
##->2.1 成交金额波动（用标准差来衡量）
## 用过去一段时间的成交金额标准差来衡量行业交易情况的稳定程度,时间窗口设定为10日。
df_AVol = df_FAmount.rolling(window = 10).std().shift(1).unstack().dropna().reset_index().rename(columns={0: 'AVol'})
df_AVol['AVol'] = (df_AVol.groupby(['Symbol'])['AVol']
                   .apply(lambda x:(x - np.min(x)) / (np.max(x) - np.min(x)))
                   .reset_index(level=0, drop=True))

## 计算因子IC值
df_IC['AVol'] = df_AVol.groupby(['TradingDate']).agg(Factor_IC,'AVol',df_Factor['Return'])['AVol']

## 绘制因子IC值时序图
plt.figure(figsize=(10, 5))  # 创建图形窗口
df_IC['AVol'].plot(title='IC value of transaction amount volatility', color = '#4C72B0')
plt.xlabel('Date')
plt.ylabel('IC Value')
plt.show()

## 数据合并
df_Factor = pd.merge(df_Factor,df_AVol,on = ['Symbol', 'TradingDate'], how = 'left')

##->2.2 成交量波动（用标准差来衡量）
## 过去一段时间成交量标准差，时间窗口设定为10日。
df_VVol = df_FVolume.rolling(window = 10).std().shift(1).unstack().dropna().reset_index().rename(columns={0: 'VVol'})
df_VVol['VVol'] = (df_VVol.groupby(['Symbol'])['VVol']
                   .apply(lambda x:(x - np.min(x)) / (np.max(x) - np.min(x)))
                   .reset_index(level=0, drop=True))

## 计算因子IC值
df_IC['VVol'] = df_VVol.groupby(['TradingDate']).agg(Factor_IC,'VVol',df_Factor['Return'])['VVol']

## 绘制因子IC值时序图
plt.figure(figsize=(10, 5))  # 创建图形窗口
df_IC['VVol'].plot(title='IC value of volume volatility', color = '#4C72B0')
plt.xlabel('Date')
plt.ylabel('IC Value')
plt.show()

## 数据合并
df_Factor = pd.merge(df_Factor,df_VVol,on = ['Symbol', 'TradingDate'], how = 'left')
#del(df_VVol)

#%%
#->3.换手率
## 长期换手率均值/短期换手率均值
df_Turnover = df_FTurnover.copy()
df_Turnover = df_Turnover.rolling(window=40).mean() / df_Turnover.rolling(window=10).mean()
df_Turnover = df_Turnover.replace([np.inf,-np.inf],np.nan)
df_Turnover = df_Turnover.shift(1).unstack().dropna().reset_index().rename(columns={0: 'Turnover'})
## 归一化
df_Turnover['Turnover'] = (df_Turnover.groupby(['Symbol'])['Turnover']
                           .apply(lambda x:(x - np.min(x)) / (np.max(x) - np.min(x)))
                           .reset_index(level=0, drop=True))

## 计算因子IC值
df_IC['Turnover'] = df_Turnover.groupby(['TradingDate']).agg(Factor_IC,'Turnover',df_Factor['Return'])['Turnover']

## 绘制因子IC值时序图
plt.figure(figsize=(10, 5))  # 创建图形窗口
df_IC['Turnover'].plot(title='IC value of turnover', color = '#4C72B0')
plt.xlabel('Date')
plt.ylabel('IC Value')
plt.show()

## 数据合并
df_Factor = pd.merge(df_Factor,df_Turnover,on = ['Symbol', 'TradingDate'], how = 'left')

#%%
#->4.多空对比
# 多头力量 = 行业指数每日收盘价 - 最低价；
# 空头力量 = 每日最高价 - 收盘价；
# 根据指数日频的价格信息构建多空对比因子

##->4.1多空对比总量
##一段时间内多头力量与空头力量的比值之和，窗口期为20日
df_LSAmount = (df_FClose - df_FLow) / (df_FHigh - df_FClose)
df_LSAmount = df_LSAmount.fillna(0)
df_LSAmount = df_LSAmount.rolling(window=20).sum()
df_LSAmount = df_LSAmount.shift(1).unstack().dropna().reset_index().rename(columns={0: 'LSAmount'})
df_LSAmount['LSAmount'] = (df_LSAmount.groupby(['Symbol'])['LSAmount']
                           .apply(lambda x:(x - np.min(x)) / (np.max(x) - np.min(x)))
                           .reset_index(level=0, drop=True))

## 计算因子IC值
df_IC['LSAmount'] = df_LSAmount.groupby(['TradingDate']).agg(Factor_IC,'LSAmount',df_Factor['Return'])['LSAmount']

## 绘制因子IC时序图
plt.figure(figsize=(10, 5))  # 创建图形窗口
df_IC['LSAmount'].plot(title='IC value of long-short amount factor', color = '#4C72B0')
plt.xlabel('Date')
plt.ylabel('IC Value')
plt.show()

df_Factor = pd.merge(df_Factor,df_LSAmount,on = ['Symbol', 'TradingDate'], how = 'left')

##->4.2多空对比变化
## 分子：多头力量 - 空头力量，即(𝐶𝑙𝑜𝑠𝑒 − 𝐿𝑜𝑤) − (𝐻𝑖𝑔ℎ − 𝐶𝑙𝑜𝑠𝑒)；
## 分母：最高价 - 最低价，日内价格区间的极值。
## 当日多空力量对比的金额绝对值 = 多空力量对比 * 当日行业成交量
## 多空对比变化因子 = 长期每日多空力量对比的指数加权平均值 - 短期每日多空力量对比的指数加权平均值

## 多空力量对比的金额绝对值
df_LSChange = (df_FClose - df_FLow) - (df_FHigh - df_FClose)
df_LSChange = df_LSChange / (df_FHigh - df_FLow)
df_LSChange = df_LSChange * df_FAmount
    
   
## 定义长短期alpha值
short_window = 10
short_alpha = 2 / (short_window + 1)
long_window = 40
long_alpha = 2 / (long_window + 1)

## 计算长短期指数加权平均值
short_ewma = df_LSChange.ewm(alpha = short_alpha, adjust = False).mean()
long_ewma  = df_LSChange.ewm(alpha = long_alpha,  adjust = False).mean()

## 计算长短期指数加权平均值之差
df_LSChange = long_ewma - short_ewma

df_LSChange = df_LSChange.shift(1).unstack().dropna().reset_index().rename(columns={0: 'LSChange'})
df_LSChange['LSChange'] = (df_LSChange.groupby(['Symbol'])['LSChange']
                           .apply(lambda x:(x - np.min(x)) / (np.max(x) - np.min(x)))
                           .reset_index(level=0, drop=True))

## 计算因子IC值
df_IC['LSChange'] = df_LSChange.groupby(['TradingDate']).agg(Factor_IC,'LSChange',df_Factor['Return'])['LSChange']

## 绘制因子IC时序图
plt.figure(figsize=(10, 5))  # 创建图形窗口
df_IC['LSChange'].plot(title='IC value of long-short change factor', color = '#4C72B0')
plt.xlabel('Date')
plt.ylabel('IC Value')
plt.show()

## 合并数据
df_Factor = pd.merge(df_Factor,df_LSChange,on = ['Symbol', 'TradingDate'], how = 'left')

del short_window, long_window, short_alpha, long_alpha, short_ewma, long_ewma

#%% 因子相关系数
# df_Factor_copy = df_Factor.copy()

## 各因子IC均值
print(df_IC.mean())

## 因子间IC序列相关系数
factor_list = ['Mom_diff','Mom_term','AVol','VVol', 'Turnover', 'LSAmount','LSChange']
relations = df_Factor[factor_list].corr()
# relations = relations[relations.index]
print(relations)

## 绘制热力图
fontsize = 14
plt.rc('font', weight='light', family='Times New Roman', style='normal', size=str(fontsize)) #设置字体
plt.tick_params(labelsize=fontsize) #设置坐标轴
sns.heatmap(relations, cmap='Blues', annot = True) #设置热力图，annot=True表示在热力图的每个单元格中显示具体的数值
plt.show()

## 因子合成
### AVol和VVol的相关性较高0.67，因此考虑使用等权法将二者合成一个因子
df_Factor['Vol'] = (df_Factor['AVol'] + df_Factor['VVol']) / 2
#df_Factor_copy = df_Factor.copy()

## 因子间IC值相关系数
factor_list = ['Mom_diff','Mom_term','Vol', 'Turnover','LSAmount','LSChange']
relations = df_Factor[factor_list].corr()
print(relations)

## 绘制热力图
fontsize = 14
plt.rc('font', weight='light', family='Times New Roman', style='normal', size=str(fontsize)) #设置字体
plt.tick_params(labelsize=fontsize) #设置坐标轴
sns.heatmap(relations, cmap='Blues', annot = True) #设置热力图，annot=True表示在热力图的每个单元格中显示具体的数值
plt.show()

del fontsize, factor_list

#%% 单因子分组分层回测
# 计算分组
def get_groups(series, n_groups):
    """
    输入: series - 因子值序列, n_groups - 分组数
    输出: 分组标签 (1到n_groups)
    """
    try:
        labels = pd.qcut(series, q =n_groups, labels = False, duplicates = "drop") + 1  # 转为1-based标签
    except ValueError:  # 处理全相同值或数据不足的情况
        labels = pd.Series(1, index=series.index)
    # 强制转换为整数并填充缺失值
    return labels.fillna(1).astype(int)

# 清理数据
df_Factor_clean = df_Factor.dropna(axis=0).copy()  # 删除缺失值

# 定义因子列表
factor_columns = ['Mom_diff','Mom_term','Vol','Turnover','LSAmount','LSChange']

# 计算分组累计收益
dfs = []
for factor in factor_columns:
    tag_col = f'Tag_{factor}'
    ret_col = f'Ret_{factor}'
    
    # 动态生成临时数据（避免污染原始数据）
    df_temp = df_Factor_clean[['TradingDate', 'Symbol', 'Return', factor]].copy()
    
    # 删除可能残留的同名分组列
    if tag_col in df_temp.columns:
        df_temp = df_temp.drop(columns=tag_col)
    
    # 生成分组标签
    df_temp[tag_col] = (
        df_temp.groupby('TradingDate', group_keys=False)[factor] #按天分组，根据每个因子的因子值分箱
        .apply(lambda x: get_groups(x, n_groups=5))
        )
    
    # 计算分组累计收益
    df_ret = (
        df_temp.groupby(['TradingDate', tag_col])['Return'].mean() #因子 每天 每个分组 收益均值
        .groupby(level=1, group_keys = False)  # level = 1表示按照第二层（tag_col）分组
        .apply(lambda x: (1 + x).cumprod() - 1) # 计算因子每个分组的累计收益
        .rename(ret_col) # 重命名累计收益
        .reset_index()
        .rename(columns={tag_col: 'Group'})  # 统一列名
    )
    df_ret['factor'] = factor  # 标记因子名称 用factor+group代替tag_factor
    dfs.append(df_ret)

## 合并所有结果
df_group_ret = pd.concat(dfs).reset_index(drop=True)

del factor, ret_col, tag_col, dfs, df_temp, df_ret

#%% 绘制分组累计收益曲线
# 设置绘图样式
sns.set_style("whitegrid")
sns.set_palette("husl")  # 使用更鲜明的颜色

# 遍历每个因子绘图
for factor in factor_columns:
    # 筛选当前因子数据
    df_plot = df_group_ret[df_group_ret['factor'] == factor]
    
    # 转换为宽表格式
    df_pivot = df_plot.pivot(
        index ='TradingDate', 
        columns ='Group', 
        values = f'Ret_{factor}'
    )
    
    # 创建画布
    plt.figure(figsize=(12, 6))
    
    # 绘制所有分组曲线
    for group in sorted(df_pivot.columns):
        plt.plot(
            pd.to_datetime(df_pivot.index), 
            df_pivot[group], 
            label = f'Group {group}',
            linewidth=2
        )

    # 美化图表
    plt.title(f'{factor} Cumulative Returns', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Returns', fontsize=12)
    plt.legend(title='Group', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))  # 显示百分比
    
    # 保存图片
    plt.show()

del factor, df_plot, df_pivot
#%% 计算多空组合夏普比率
# 计算日度分组平均收益
dfs = []

for factor in factor_columns:
    tag_col = f'Tag_{factor}'
    ret_col = f'Ret_{factor}'
    
    # 动态生成临时数据
    df_temp = df_Factor_clean[['TradingDate', 'Symbol', 'Return', factor]].copy()
    
    # 删除可能残留的同名分组列
    if tag_col in df_temp.columns:
        df_temp = df_temp.drop(columns = tag_col)
    
    # 生成分组标签
    df_temp[tag_col] = (
        df_temp.groupby('TradingDate', group_keys = False)[factor]
        .apply(lambda x: get_groups(x, n_groups=5))
    )
    
    # 计算每日各分组平均收益（未累计）
    df_ret = (
        df_temp.groupby(['TradingDate', tag_col])['Return'].mean()
        .rename(ret_col)
        .reset_index()
        .rename(columns={tag_col: 'Group'})
    )
    
    df_ret['factor'] = factor
    
    dfs.append(df_ret)

# 合并日度平均收益数据
df_group_retAve = pd.concat(dfs).reset_index(drop=True)

# 计算夏普比率
def calculate_sharpe(returns, risk_free_rate = 0.0, annualize_factor=252):
    """计算年化夏普比率"""
    excess_returns = returns - risk_free_rate / annualize_factor
    if len(excess_returns) < 2:  # 至少需要2个观测值
        return np.nan
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()
    return mean_return / std_return * np.sqrt(annualize_factor)

# 存储各因子夏普比率
sharpe_results = []

for factor in factor_columns:
    ret_col = f'Ret_{factor}'
    
    # 提取当前因子的日度收益数据
    df_factor = df_group_retAve[df_group_retAve['factor'] == factor]
    
    # 获取最高组和最低组
    max_group = df_factor['Group'].max()
    min_group = df_factor['Group'].min()
    
    # 提取最高组和最低组日度收益
    returns_high = df_factor[df_factor['Group'] == max_group].set_index('TradingDate')[ret_col]
    returns_low = df_factor[df_factor['Group'] == min_group].set_index('TradingDate')[ret_col]
    
    # 对齐日期
    common_dates = returns_high.index.intersection(returns_low.index)
    returns_high = returns_high.loc[common_dates]
    returns_low = returns_low.loc[common_dates]
    
    # 计算多空组合收益
    ls_returns = returns_high - returns_low
    
    # 计算夏普比率
    sharpe = calculate_sharpe(ls_returns)
    sharpe_results.append(sharpe)

# 转换为DataFrame
df_sharpe = pd.DataFrame({
    'Factor': factor_columns,
    'Sharpe Ratio': sharpe_results
})

# 计算平均夏普比率
average_sharpe = df_sharpe['Sharpe Ratio'].mean()
print(f"所有Alpha因子的平均夏普比率: {average_sharpe}")
print("\n各因子夏普比率详情:")
print(df_sharpe)

del dfs, factor, tag_col, ret_col, df_temp, df_ret, sharpe_results, df_factor, max_group, min_group, returns_high, returns_low, common_dates, ls_returns, sharpe