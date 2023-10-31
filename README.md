# Laporan Proyek Machine Learning
### Nama : Achmad Fauzan Nabil
### Nim : 211351002
### Kelas : Malam B

## Domain Proyek

Populasi yang kian tahun semakin meningkat membuat padatnya populasi dibumi ini, sehingga diperlukannya sistem yang dapat membantu dalam memprediksi total populasi sekian tahun kedepannya.

## Business Understanding

Dapat memberikan informasi terkait perubahan dan pertambahan populasi di india dengan index waktu pertahun.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Populasi yang selalu meningkat sehingga d perlukannya sistem yang dapat memberikan informasi terkait jumlah populasi di India kedepannya

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Dapat membantu dalam memberikan informasi terkait populasi di India beberapa tahun kedepannya


    ### Solution statements
    - Pembuatan sistem yang dapat melakukan forecasting jumlah populasi di India berbasis web sehingga dapat diakses oleh siapapun yang membutuhkan informasi tersebut
    - Model yang dibuat menggunakan 3 algoritma yang selanjutnya akan di bandingan dengan nilai RSME terkecil baru dipakai untuk sistem tersebut

## Data Understanding

Dataset yang digunakan diambil dari kaggle yang berisi informasi terkait jumlah populasi di india, perubahan jumlahnya dan lain lainnya terkait populasi di india

dataset: [Population of India (2050-1955)](https://www.kaggle.com/datasets/anandhuh/population-data-india).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
Informasi Atribut:

- Year - Tahun dari 2020-1955
- Population - Populasi pada tahun yang bersangkutan
- Yearly % Change - Persentase Perubahan Tahunan dalam Populasi
- Yearly Change - Perubahan Tahunan Populasi
- Migrants (net) -  Jumlah total migran
- Median Age - Usia rata-rata penduduk
- Fertility Rate - Tingkat kesuburan
- Density (P/Km²)- Kepadatan penduduk (populasi per km persegi)
- Urban Pop %- Persentase populasi perkotaan
- Urban Population- Populasi perkotaan
- Country's Share of World Pop - Pangsa populasi
- World Population - Populasi Dunia pada tahun yang bersangkutan
- India Global Rank - Peringkat Global dalam Populasi

cek tipe data:
<img width="332" alt="image" src="https://github.com/fauzann221/forecasting-tugas-uts/assets/149223860/c5600507-bc55-4b5f-bec0-d97c99c60275">

Visualisasi kolom populasi
```
df.plot(grid=True)
```
![image](https://github.com/fauzann221/forecasting-tugas-uts/assets/149223860/ae793d87-49d6-4dc0-bf2d-0391a806b5c7)


## Data Preparation
Hapus kolom yang tidak dipakai:
```
df = df.drop(['Yearly % Change', 'Yearly Change','Migrants (net)', 'Median Age',
              'Fertility Rate', 'Density (P/Km²)','Urban Pop %', 'Urban Population',
              "Country's Share of World Pop",'World Population', 'India Global Rank'], axis=1)
```
convert kolom tahun ke tipe data datetime:
```
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
```
setting index data
```
df.set_index(['Year'], inplace=True)
```
resample dan setting frequency tanggal
```
df = df.resample('Y').sum()
```
setting data train dan test
```
train_df = df.iloc[:50]
test_df = df.iloc[51:]
```
testing ad fuller
```
def adf_test(timeseries):
    print ('Hasil testing Dickey-Fuller')
    print ('----------------------------------')
    adftest = adfuller(timeseries)
    adf_output = pd.Series(adftest[0:4], index=['Test statistic','p-value','Lags Used','Number of Observation Used'])
    for key, Value in adftest[4].items() :
        adf_output['Critical Value (%s)' %key] = Value
    print (adf_output)

adf_test(df.values)
```
<img width="218" alt="image" src="https://github.com/fauzann221/forecasting-tugas-uts/assets/149223860/3a85e228-8855-415a-8086-58413fa2a53e">

cek selisih antar dua poin data
```
diff_df = df.diff()
diff_df.head()
```
hapus data yang kosong
```
diff_df.dropna(inplace=True)
```
cek korelasi dari deret waktu
```
plot_acf(diff_df)
plot_pacf(diff_df)
```
![image](https://github.com/fauzann221/forecasting-tugas-uts/assets/149223860/5358a5f9-010c-4fb2-9ab1-bf946f283c41)

## Modeling
Pembuatan model untuk perbandingan, coba untuk prediksi 15 tahun kedepan
**Single Exponential Smoothing**
```
single_exp = SimpleExpSmoothing(train_df).fit()
single_exp_train_pred = single_exp.fittedvalues
single_exp_test_pred = single_exp.forecast(15)
```
```
train_df['Population'].plot(style='--', color='gray', legend=True, label='train_df')
test_df['Population'].plot(style='--', color='r', legend=True, label='test_df')
single_exp_test_pred.plot(color='b', legend=True, label='Prediction')
```
![image](https://github.com/fauzann221/forecasting-tugas-uts/assets/149223860/ba6c8ac1-dcd5-4efd-95ee-42ee9c455750)

Evaluasi single exponential smoothing:

Train RMSE : 320453411.9592464<br>
Test RMSE : 812873731.388359<br>
Train MAPE : 7.038215074382666e+23<br>
Test MAPE : 3.6395949304555325e+23<br>

**ARIMA**
```
ar = ARIMA(train_df, order=(15,1,15)).fit()
ar_train_pred = ar.fittedvalues
ar_test_pred = ar.forecast(15)
```
```
train_df['Population'].plot(style='--', color='gray', legend=True, label='train_df')
test_df['Population'].plot(style='--', color='r', legend=True, label='test_df')
ar_test_pred.plot(color='b', legend=True, label='Prediction')
```
![image](https://github.com/fauzann221/forecasting-tugas-uts/assets/149223860/80437ccb-a88b-4491-a74a-a5b812a2d12e)

Evaluasi model ARIMA:

Train RMSE : 116132410.34184496<br>
Test RMSE : 869433547.2106411<br>
Train MAPE : 1.6624039948213573e+23<br>
Test MAPE : 5.627361926236723e+23<br>


## Evaluation
Mari bandingkan hasil 2 model tersebut dengan ketentuan mencari nilai RMSE terkecil
```
comparision_df = pd.DataFrame(data=[
    ['Single Exp Smoothing', Test_RMSE_SES, Test_MAPE_SES],
    ['ARIMA', Test_RMSE_AR, Test_MAPE_AR]
    ],
    columns=['Model','RMSE','MAPE'])
comparision_df.set_index('Model', inplace=True)
```
```
comparision_df.sort_values(by='RMSE')
```
<img width="262" alt="image" src="https://github.com/fauzann221/forecasting-tugas-uts/assets/149223860/130534c2-d0cb-4dd8-8635-3e957344ec60">

berdasarkan nilai RMSE terkecil didapatkan oleh algoritma Single Exponential Smoothing maka dari itu, algoritma tersebut yang akan dipakai untuk sistem yang dibuat

## Deployment
[Forecasting Populasi di India](https://forecast-uts-fauzan.streamlit.app/)

