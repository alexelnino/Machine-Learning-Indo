import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel(
    'indo_12_1.xls',
    header=3,
    skipfooter=3,
    na_values=['-']     #untuk membuat pengecualian pada tanda "-" di data
)

# print(df)
# print(df[2010].max())
# print(df[df[2010]==df[2010].max()])
# print(df.index)     #menampilkan index semuanya

# mencari daerah dengan populasi terbanyak di th. 2010
dfMax2010=df[df[2010]==df[2010].max()]
namaMax2010=dfMax2010.index[0]
print(dfMax2010)
print(dfMax2010.index[0])
print(df.loc[dfMax2010.index[0]])
print()

# df prov 1971 jumlah penduduk min
df.dropna(subset=[1971])    #hapus data NuN di kolom 1971
dfMin1971 = df[df[1971]==df[1971].min()]
namaMin1971=dfMin1971.index[0]
print(dfMin1971)
print(dfMin1971.columns.values)
print(dfMin1971.iloc[0])
print()

# cari Indonesia
df=pd.read_excel(
    'indo_12_1.xls',
    header=3,
    skipfooter=2,
    # index_col=0,        
    na_values=['-']     #untuk membuat pengecualian pada tanda "-" di data
)

dfIndo=df[df[2010]==df[2010].max()]
namaIndo=dfIndo.index[0]
print(dfIndo)
print(dfIndo.index[0])
print(df.loc[dfIndo.index[0]])
print()

# linear regression
from sklearn.linear_model import LinearRegression
modelMax2010 = LinearRegression()
modelMin1971 = LinearRegression()
modelIndo=LinearRegression()



#training           buat prediksi
x=dfMax2010.columns.values.reshape(-1,1)
y=dfMax2010.values[0]
modelMax2010.fit(x,y)

x=dfMin1971.columns.values.reshape(-1,1)
y=dfMin1971.values[0]
modelMin1971.fit(x,y)

x=dfIndo.columns.values.reshape(-1,1)
y=dfIndo.values[0]
modelIndo.fit(x,y)


#prediksi jabar 2050
max1050=int(round(modelMax2010.predict([[2050]])[0]))
min7150=int(round(modelMin1971.predict([[2050]])[0]))
indo50=int(round(modelIndo.predict([[2050]])[0]))

print('prediksi jmlh penduduk {} th 2050 ='.format(namaMax2010),max1050)
print('prediksi jmlh penduduk {} th 2050 ='.format(namaMin1971),min7150)
print('prediksi jmlh penduduk {} th 2050 ='.format(namaIndo),indo50)


plt.plot(
    dfMax2010.columns.values,
    dfMax2010.iloc[0],
    'm-',
)
plt.plot(    
    dfMin1971.columns.values,
    dfMin1971.iloc[0],
    'g-'
    
)
plt.plot(    
    dfIndo.columns.values,
    dfIndo.iloc[0],
    'r-'
    
)
plt.scatter(
    dfMax2010.columns.values,
    dfMax2010.iloc[0],
    color='m',
    s=80
)

plt.scatter(
    dfMin1971.columns.values,
    dfMin1971.iloc[0],
    color='g',
    s=80
)

plt.scatter(
    dfIndo.columns.values,
    dfIndo.iloc[0],
    color='r',
    s=80
)

# best fit line
plt.plot(
    dfMax2010.columns.values,
    modelMax2010.coef_*dfMax2010.columns.values + modelMax2010.intercept_,'y-'
)
plt.plot(
    dfMin1971.columns.values,
    modelMin1971.coef_*dfMin1971.columns.values + modelMin1971.intercept_,'y-'
)
plt.plot(
    dfIndo.columns.values,
    modelIndo.coef_*dfIndo.columns.values + modelIndo.intercept_,'y-'
)


plt.legend([namaMax2010,namaMin1971,namaIndo])
plt.title('jumlah penduduk {} (1971-2010)'.format(namaIndo))
plt.grid(True)
plt.show()

