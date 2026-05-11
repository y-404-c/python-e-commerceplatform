import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt
from mlxtend.frequent_patterns import apriori, association_rules


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1.读入数据
df = pd.read_excel(r"E:\Git项目\Online Retail.xlsx")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 2.数据清洗
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df = df.dropna(subset=['CustomerID'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
print(f"清洗后数据量: {len(df)} 行")

# RFM用户分层
snapshot = df['InvoiceDate'].max() + dt.timedelta(days=1)
rfm = df.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalPrice', 'sum')
)

rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
rfm['M_score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])
rfm['RFM_Score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)

def segment(score):
    r, f, m = score[0], score[1], score[2]
    if r>='3' and f>='3' and m>='3': return '重要价值客户'
    elif r>='3' and f<'3' and m>='3': return '重要保持客户'
    elif r<'3' and f>='3' and m>='3': return '重要发展客户'
    elif r<'3' and f<'3' and m>='3': return '重要挽留客户'
    elif r>='3' and f>='3' and m<'3': return '一般价值客户'
    elif r>='3' and f<'3' and m<'3': return '一般发展客户'
    elif r<'3' and f>='3' and m<'3': return '一般保持客户'
    else: return '流失客户'

rfm['Segment'] = rfm['RFM_Score'].apply(segment)

# 4.商品二八定律
product_sales = df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False)
cum_pct = product_sales.cumsum() / product_sales.sum() * 100
top80 = (cum_pct <= 80).sum()
print(f"\n帕累托分析：{top80}/{len(product_sales)} 种商品贡献了80%销售额，占比 {top80/len(product_sales)*100:.1f}%")

# 5.购物篮深度分析
basket_stats = df.groupby('InvoiceNo').agg(
    商品种类数=('Description', 'nunique'),
    商品数量=('Quantity', 'sum'),
    订单金额=('TotalPrice', 'sum')
)
print(f"平均客单价: £{basket_stats['订单金额'].mean():.2f}，每单平均 {basket_stats['商品种类数'].mean():.1f} 种商品")

# 6. Apriori 关联规则
# 取英国数据举例
basket_df = df[df['Country']=='United Kingdom'].groupby(['InvoiceNo','Description'])['Quantity'].sum().unstack().fillna(0)
basket_df = basket_df.applymap(lambda x: 1 if x>0 else 0)

freq_items = apriori(basket_df, min_support=0.02, use_colnames=True)
rules = association_rules(freq_items, metric='lift', min_threshold=1)
rules = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)]
rules = rules[['antecedents','consequents','support','confidence','lift']].sort_values('lift', ascending=False)
print(f"\n关联规则数量: {len(rules)}")
print(rules.head(10))

# 7. 可视化
fig, axes = plt.subplots(2,3, figsize=(18,10))

# 7.1 RFM柱状图
rfm['Segment'].value_counts().plot(kind='bar', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('RFM用户分层人数')
axes[0,0].tick_params(axis='x', rotation=45)

# 7.2 销售趋势（按月）
df['Month'] = df['InvoiceDate'].dt.to_period('M')
monthly_sales = df.groupby('Month')['TotalPrice'].sum()
monthly_sales.plot(ax=axes[0,1], marker='o', color='#e74c3c')
axes[0,1].set_title('月度销售额趋势')

# 7.3 帕累托图（前30商品）
axes[0,2].bar(range(30), product_sales.head(30).values, color='#3498db', alpha=0.7)
ax2 = axes[0,2].twinx()
ax2.plot(range(30), cum_pct.head(30).values, color='red', marker='o')
ax2.axhline(80, color='gray', linestyle='--')
axes[0,2].set_title('商品帕累托图（前30）')

# 7.4 订单金额分布
axes[1,0].hist(basket_stats['订单金额'], bins=50, color='#2ecc71', alpha=0.7)
axes[1,0].axvline(basket_stats['订单金额'].mean(), color='red', linestyle='--', label='均值')
axes[1,0].set_title('订单金额分布')
axes[1,0].legend()

# 7.5 国家销售额TOP10
top_countries = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
top_countries.plot(kind='barh', ax=axes[1,1], color='#9b59b6')
axes[1,1].set_title('销售额TOP10国家')
axes[1,1].invert_yaxis()

# 7.6 关联规则置信度-提升度散点
if len(rules)>0:
    axes[1,2].scatter(rules['confidence'], rules['lift'], alpha=0.5, color='orange')
    axes[1,2].set_xlabel('Confidence')
    axes[1,2].set_ylabel('Lift')
    axes[1,2].set_title('关联规则分布')
else:
    axes[1,2].text(0.5,0.5, '无关联规则', ha='center')
    axes[1,2].set_title('关联规则（无）')

plt.tight_layout()
plt.savefig('full_analysis.png', dpi=150)
plt.show()


rfm.to_csv('rfm_result.csv', index=False)
product_sales.to_csv('product_sales.csv')
rules.to_csv('association_rules.csv', index=False)
print("\n 分析完成，图表已保存。")