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
df_Fund = pd.read_csv('IDX_Idxtrd.csv')
df_Fund = df_Fund.rename(columns = {
    'Indexcd' : 'Symbol',
    'Idxtrd01' : 'TradingDate',
    'Idxtrd02' : 'Open',
    'Idxtrd03' : 'High',
    'Idxtrd04' : 'Low',
    'Idxtrd05' : 'Close',
    'Idxtrd06' : 'Volume',
    'Idxtrd07' : 'Amount',
    'Idxtrd08' : 'Return'
    }) 

# 提取相应数据
df_FVolume = df_Fund.pivot(index = 'TradingDate',columns = 'Symbol',values = 'Volume')
df_FAmount = df_Fund.pivot(index = 'TradingDate',columns = 'Symbol',values = 'Amount')
df_FOpen = df_Fund.pivot(index = 'TradingDate',columns = 'Symbol',values = 'Open')
df_FClose = df_Fund.pivot(index = 'TradingDate',columns = 'Symbol',values = 'Close')
df_FHigh = df_Fund.pivot(index = 'TradingDate',columns = 'Symbol',values = 'High')
df_FLow = df_Fund.pivot(index = 'TradingDate',columns = 'Symbol',values = 'Low')


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
#df_Mom_diff['Mom_diff'] = (df_Mom_diff.groupby(['Symbol'])['Mom_diff']
#                           .apply(lambda x:(x - np.min(x)) / (np.max(x) - np.min(x)))
#                           .reset_index(level=0, drop=True))

## 计算因子IC值
df_IC['Mom_diff'] = df_Mom_diff.groupby(['TradingDate']).agg(Factor_IC,'Mom_diff',df_Factor['Return'])['Mom_diff']

## 绘制因子IC值时序图
plt.figure(figsize=(10, 5))  # 创建图形窗口
df_IC['Mom_diff'].plot(title='IC value of the second-order momentum factor')
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
#df_Mom_term['Mom_term'] = (df_Mom_term.groupby(['Symbol'])['Mom_term']
#                           .apply(lambda x:(x - np.min(x)) / (np.max(x) - np.min(x)))
#                           .reset_index(level=0, drop=True))

## 计算因子IC值
df_IC['Mom_term'] = df_Mom_term.groupby(['TradingDate']).agg(Factor_IC,'Mom_term',df_Factor['Return'])['Mom_term']

### 绘制因子IC值时序图
plt.figure(figsize=(10, 5))  # 创建图形窗口
df_IC['Mom_term'].plot(title='IC value of momentum term difference factor')
plt.xlabel('Date')
plt.ylabel('IC Value')
plt.show()

## 数据合并
df_Factor = pd.merge(df_Factor,df_Mom_term,on = ['Symbol', 'TradingDate'], how = 'left')

#->2. 交易波动
##->2.1 成交金额波动（用标准差来衡量）
## 用过去一段时间的成交金额标准差来衡量行业交易情况的稳定程度，并取相反数，波动率最小组为因子值最大组， 波动率最大组为因子值最小组。 时间窗口设定为10日。
df_AVol = df_FAmount.rolling(window = 10).std().shift(1).unstack().dropna().reset_index().rename(columns={0: 'AVol'})
df_AVol['AVol'] = (-1) * df_AVol['AVol']
#df_AVol['AVol'] = (df_AVol.groupby(['Symbol'])['AVol']
#                   .apply(lambda x:(x - np.min(x)) / (np.max(x) - np.min(x)))
#                   .reset_index(level=0, drop=True))

## 计算因子IC值
df_IC['AVol'] = df_AVol.groupby(['TradingDate']).agg(Factor_IC,'AVol',df_Factor['Return'])['AVol']

## 绘制因子IC值时序图
plt.figure(figsize=(10, 5))  # 创建图形窗口
df_IC['AVol'].plot(title='IC value of transaction amount volatility factor')
plt.xlabel('Date')
plt.ylabel('IC Value')
plt.show()

## 数据合并
df_Factor = pd.merge(df_Factor,df_AVol,on = ['Symbol', 'TradingDate'], how = 'left')

##->2.2 成交量波动（用标准差来衡量）
## 过去一段时间成交量标准差的相反数，波动率最小组为因子值最大组，波动率最大组为因子值最小组。代表做多市场情绪稳定的行业。时间窗口设定为10日。
df_VVol = df_FVolume.rolling(window = 10).std().shift(1).unstack().dropna().reset_index().rename(columns={0: 'VVol'})
df_VVol['VVol'] = (-1) * df_VVol['VVol']
#df_VVol['VVol'] = (df_VVol.groupby(['Symbol'])['VVol']
#                   .apply(lambda x:(x - np.min(x)) / (np.max(x) - np.min(x)))
#                   .reset_index(level=0, drop=True))

## 计算因子IC值
df_IC['VVol'] = df_VVol.groupby(['TradingDate']).agg(Factor_IC,'VVol',df_Factor['Return'])['VVol']

## 绘制因子IC值时序图
plt.figure(figsize=(10, 5))  # 创建图形窗口
df_IC['VVol'].plot(title='IC value of volume volatility factor')
plt.xlabel('Date')
plt.ylabel('IC Value')
plt.show()

## 数据合并
df_Factor = pd.merge(df_Factor,df_VVol,on = ['Symbol', 'TradingDate'], how = 'left')
#del(df_VVol)

#->3.多空对比
# 多头力量 = 行业指数每日收盘价 - 最低价；
# 空头力量 = 每日最高价 - 收盘价；
# 根据指数日频的价格信息构建多空对比因子

##->3.1多空对比总量
##一段时间内多头力量与空头力量的比值之和的相反数，窗口期为20日
df_LSAmount = (df_FClose - df_FLow) / (df_FHigh - df_FClose)
df_LSAmount = df_LSAmount.fillna(0)
df_LSAmount = (-1) * df_LSAmount.rolling(window=20).sum()
df_LSAmount = df_LSAmount.shift(1).unstack().dropna().reset_index().rename(columns={0: 'LSAmount'})
#df_LSAmount['LSAmount'] = (df_LSAmount.groupby(['Symbol'])['LSAmount']
#                           .apply(lambda x:(x - np.min(x)) / (np.max(x) - np.min(x)))
#                           .reset_index(level=0, drop=True))

## 计算因子IC值
df_IC['LSAmount'] = df_LSAmount.groupby(['TradingDate']).agg(Factor_IC,'LSAmount',df_Factor['Return'])['LSAmount']

## 绘制因子IC时序图
plt.figure(figsize=(10, 5))  # 创建图形窗口
df_IC['LSAmount'].plot(title='IC value of long-short amount factor')
plt.xlabel('Date')
plt.ylabel('IC Value')
plt.show()

df_Factor = pd.merge(df_Factor,df_LSAmount,on = ['Symbol', 'TradingDate'], how = 'left')

##->3.2多空对比变化
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
#df_LSChange['LSChange'] = (df_LSChange.groupby(['Symbol'])['LSChange']
#                           .apply(lambda x:(x - np.min(x)) / (np.max(x) - np.min(x)))
#                           .reset_index(level=0, drop=True))

## 计算因子IC值
df_IC['LSChange'] = df_LSChange.groupby(['TradingDate']).agg(Factor_IC,'LSChange',df_Factor['Return'])['LSChange']

## 绘制因子IC时序图
plt.figure(figsize=(10, 5))  # 创建图形窗口
df_IC['LSChange'].plot(title='IC value of long-short change factor')
plt.xlabel('Date')
plt.ylabel('IC Value')
plt.show()

## 合并数据
df_Factor = pd.merge(df_Factor,df_LSChange,on = ['Symbol', 'TradingDate'], how = 'left')

del short_window, long_window, short_alpha, long_alpha, short_ewma, long_ewma

#->4.因子合并
df_Factor_copy = df_Factor.copy()

## 各因子IC均值
print(df_IC.mean())

## 因子间IC序列相关系数
factor_list = ['Mom_diff','Mom_term','AVol','VVol','LSAmount','LSChange']
relations = df_Factor[factor_list].corr()
# relations = relations[relations.index]

## 绘制热力图
fontsize = 14
plt.rc('font', weight='light', family='Times New Roman', style='normal', size=str(fontsize)) #设置字体
plt.tick_params(labelsize=fontsize) #设置坐标轴
sns.heatmap(relations, cmap='Blues', annot = True) #设置热力图，annot=True表示在热力图的每个单元格中显示具体的数值
plt.show()

## 因子合成
### AVol和VVol的相关性较高0.67，因此考虑使用等权法将二者合成一个因子
df_Factor['Vol'] = (df_Factor['AVol'] + df_Factor['VVol']) / 2

## 因子间IC值相关系数
factor_list = ['Mom_diff','Mom_term','Vol','LSAmount','LSChange']
relations = df_Factor[factor_list].corr()

## 绘制热力图
fontsize = 14
plt.rc('font', weight='light', family='Times New Roman', style='normal', size=str(fontsize)) #设置字体
plt.tick_params(labelsize=fontsize) #设置坐标轴
sns.heatmap(relations, cmap='Blues', annot = True) #设置热力图，annot=True表示在热力图的每个单元格中显示具体的数值
plt.show()

del fontsize

#%% 单因子分组分层回测
# 计算分组
def get_groups(group, col_name, n_groups):
    """
    对每个分组按指定列的值进行分位数分组，生成标签（1到n_groups）。
    例如，n_groups=5 表示将数据分为5个等分（五分位）。
    """
    try:
        # 使用 pd.qcut 分位数切割，处理可能的重复值
        labels = pd.qcut(
            group[col_name], 
            q=n_groups, 
            labels=False, 
            duplicates='drop'  # 如果数据不足n_groups，自动调整
        ) + 1  # 将标签从0-based转为1-based
    except ValueError as e:
        # 处理无法分组的极端情况（例如所有值相同）
        labels = pd.Series([1] * len(group), index=group.index)
    return pd.Series(labels, index=group.index)


df_Factor = df_Factor.dropna(axis=0)  # 明确指定按行删除
# 对各行业进行分组
df_group = df_Factor[['TradingDate','Symbol','Return']]
df_group['Return'] = df_group['Return'] * 0.01

## 参照研报做法计算行业等权收益
df_group['Tag_Mom_diff'] = df_Factor.groupby(['TradingDate'], group_keys = False).apply(get_groups, 'Mom_diff', 5)
df_group['Tag_Mom_term'] = df_Factor.groupby(['TradingDate'], group_keys = False).apply(get_groups, 'Mom_term', 5)
df_group['Tag_Vol'] = df_Factor.groupby(['TradingDate'], group_keys = False).apply(get_groups, 'Vol', 5)
df_group['Tag_LSAmount'] = df_Factor.groupby(['TradingDate'], group_keys = False).apply(get_groups, 'LSAmount', 5)
df_group['Tag_LSChange'] = df_Factor.groupby(['TradingDate'], group_keys = False).apply(get_groups, 'LSChange', 5)
df_group['Avg'] = df_Factor.groupby(['TradingDate']).Return.transform('mean')

# 计算分组收益（累计，等权）
dfs = []
for i in ['Mom_diff','Mom_term','Vol','LSAmount','LSChange']:
    tag_col = f'Tag_{i}'
    ret_col = f'Ret_{i}'
    
    # 计算每组累计收益
    df_temp = (
        (1 + df_group.groupby(['TradingDate', tag_col])['Return'].mean())
        .cumprod().sub(1)
        .rename(ret_col)
        .reset_index()
    )
    dfs.append(df_temp)

## 合并所有结果（横向拼接）
df_group_ret = pd.concat(dfs, axis=1)
df_group_ret = df_group_ret.loc[:, ~df_group_ret.columns.duplicated()]  # 去重列
del dfs, df_temp, i, tag_col, ret_col

# 画图
## 设置绘图样式
sns.set_style("whitegrid")

## 遍历每个因子标签
for factor in ['Mom_diff','Mom_term','Vol','LSAmount','LSChange']:
    ### 提取对应因子数据
    df_plot = df_group_ret[['TradingDate', f'Tag_{factor}', f'Ret_{factor}']].copy()
    df_plot.columns = ['Date', 'Group', 'CumReturn']  # 统一列名
    df_plot['Group'] = df_plot['Group'].astype(str)
    
    ### 转换为透视表格式（日期为索引，分组为列）
    df_pivot = df_plot.pivot(index='Date', columns='Group', values='CumReturn')
    
    ### 创建画布
    plt.figure(figsize=(12, 6))
    
    ### 绘制所有分组的累计收益曲线
    for group in df_pivot.columns:
        plt.plot(pd.to_datetime(df_pivot.index), 
                 df_pivot[group], 
                 label=f'Group {group}')
    
    # 计算纵轴范围（例如取所有组的最小值和最大值）
    y_min = df_pivot.min().min()  # 所有组的最小累计收益
    y_max = df_pivot.max().max()  # 所有组的最大累计收益
    margin = 0.1 * (y_max - y_min)  # 添加10%的边距
    plt.ylim(y_min - margin, y_max + margin)  # 设置纵轴范围
    
    ### 添加图表元素
    plt.title(f'{factor} factor: Cumulative Returns by Group')
    plt.xlabel('Trading Date')
    plt.ylabel('Cumulative Returns')
    plt.legend(title='Group', loc='upper left')
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签
    
    ### 显示图形
    plt.show() 
    
del factor, group, margin, y_max, y_min
 
# 计算Sharpe ratio
def calculate_sharpe(returns, risk_free_rate = 0.0, annualize_factor=252):
    """
    计算夏普比率
    :param returns: 日收益率序列（列表或数组）
    :param risk_free_rate: 年化无风险利率（默认0）
    :param annualize_factor: 年化因子（日数据=252，月数据=12）
    :return: 年化夏普比率
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return np.nan
    # 计算年化收益率和波动率
    annualized_return = np.mean(returns) * annualize_factor
    annualized_volatility = np.std(returns) * np.sqrt(annualize_factor)
    # 计算夏普比率
    sharpe = (annualized_return - risk_free_rate) / annualized_volatility
    return sharpe


# 分组Sharpe
sharpe_ratios = pd.DataFrame()
rf = 0.02 

for factor in factor_list:
    tag_col = f'Tag_{factor}'  # 分组标签列名（例如 Tag_Mom_diff）

    # 按分组标签计算夏普比率
    sharpe = (
        df_group.groupby(tag_col)['Return']  # 按分组标签分组
        .apply(calculate_sharpe, risk_free_rate = rf)         # 应用夏普比率函数
        .sort_index()                        # 按分组标签升序排列（Group1到Group5）
        .rename(lambda x: f'Group{x}')       # 将分组标签转为Group1, Group2...
    )

    # 将结果添加到sharpe_df中（行名为因子名称）
    sharpe_ratios = pd.concat([
        sharpe_ratios,
        pd.DataFrame([sharpe], index=[factor])
    ])

column_order = [f'Group{i}' for i in range(1, 6)]
sharpe_ratios = sharpe_ratios.reindex( columns = column_order)

# long-short Sharpe
sharpe_df = pd.DataFrame(columns=['factor', 'long_short'])

for factor in factor_list:
    tag_col = f'Tag_{factor}'  # 分组标签列
    
    try:
        # 先对同一交易日和分组的收益率取均值
        df_pivot = (
            df_group.groupby(['TradingDate', tag_col])['Return']
            .mean()  # 聚合重复值（取均值）
            .unstack()  # 转换为宽表
        )
        # 构建多空组合收益
        df_pivot['Long_Short'] = df_pivot[5] - df_pivot[1]
        # 计算夏普比率
        sharpe_ls = calculate_sharpe(
            df_pivot['Long_Short'].dropna(),
            risk_free_rate=rf,
        )
    except KeyError:
        sharpe_ls = np.nan
    
    # 创建当前因子的结果行（DataFrame格式）
    temp_df = pd.DataFrame({
        'factor': [factor],
        'long_short': [sharpe_ls]
    })
    
    # 添加到总结果中
    sharpe_df = pd.concat([sharpe_df, temp_df], ignore_index=True)
