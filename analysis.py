import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Incarcam setul de date
df=pd.read_csv("C:\\Users\\elian\\Desktop\\teme\\day.csv")

# Afisam primele 5 randuri din setul de date ca sa intelegem cu ce fel de date lucram
print("Primele 5 randuri din setul de date sunt:")
print(df.head())

# Afisam numarul de inregistrari si de coloane
print("Numarul de inregistrari si de coloane este:",df.shape)
#%%
# Verificam daca exista valori null
print("Valorile lipsa de pe fiecare coloana sunt:")
print(df.isnull().sum())

# A doua varianta pentru valori null ca sa fim 100% siguri
print("Exista valori lipsa in setul de date?:")
print(df.isnull().values.any())
#%%
# Selectam doar coloanele cu valori numerice relevante pentru a calcula media,mediana,deviatia standard si cvartiile acestora
cols_numerice = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']

# Calculam media pentru valorile numerice
print("Media pentru valorile numerice este:")
print(df[cols_numerice].mean())

# Calculam mediana pentru valorile numerice
print("Mediana pentru valorile numerice este:")
print(df[cols_numerice].median())

# Calculam deviata standard pentru valorile numerice
print("Deviatia standard pentru valorile numerice este:")
print(df[cols_numerice].std())

# Calculam cvartiile pentru valorile numerice
print("Cvartiile pentru valorile numerice sunt:")
print(df[cols_numerice].quantile([0.25,0.5,0.75]))

# Adaugam media,mediana,deviatia standard si cvartiile intr-un singur tabel pentru o vizualizare mai buna
from IPython import display
print("Tabel cu media, mediana, deviata standard si cvartiile pentru valorile numerice:")
display.display(df[cols_numerice].describe().loc[['mean', '50%', 'std', '25%', '75%']])
#%%
# Identificam variabilele categoriale
# Pasul 1: Vedem câte valori unice are fiecare coloană
print("Număr de valori unice per coloană:")
print(df.nunique())

# Pasul 2: Filtrăm coloanele care au mai puțin de 15 valori (acestea sunt categoriile)
categorii = [col for col in df.columns if df[col].nunique() < 15]

# Pasul 3: Afișăm variabilele categoriale identificate
print("Variabilele categoriale identificate sunt:")
print(categorii)
#%%
# Calculăm media închirierilor (cnt) pentru diferite grupuri
medie_sezon = df.groupby('season')['cnt'].mean()
medie_vreme = df.groupby('weathersit')['cnt'].mean()
medie_lucru = df.groupby('workingday')['cnt'].mean()

print("Media pe Sezoane (1:Iarna, 2:Primavara, 3:Vara, 4:Toamna):")
print(medie_sezon)

print("Media pe Vreme (1:Limpede, 2:Nori/Ceață, 3:Ploaie/Zăpadă):")
print(medie_vreme)

print("Weekend/Sărbătoare (0) vs Media Zi Lucrătoare (1):")
print(medie_lucru)

# Realizam un boxplot cu 3 grafice pentru a intelege mai bine diferentele intre medii
plt.figure(figsize=(15, 5))

# Boxplot pentru Sezon
plt.subplot(1, 3, 1)
sns.boxplot(x='season', y='cnt', data=df, color='#79F8F8')
plt.title('Boxplot pentru Sezon')
plt.xlabel('Sezon (1:Iarna, 2:Primavara, 3:Vara, 4:Toamna)')
plt.ylabel('Număr Închirieri (cnt)')

# Boxplot pentru Vreme
plt.subplot(1, 3, 2)
sns.boxplot(x='weathersit', y='cnt', data=df, color='#7FDD4C')
plt.title('Boxplot pentru Vreme')
plt.xlabel('Vreme (1:Limpede, 2:Nori/Ceață, 3:Ploaie/Zăpadă)')
plt.ylabel('Număr Închirieri (cnt)')

# Boxplot pentru Zi Lucrătoare
plt.subplot(1, 3, 3)
sns.boxplot(x='workingday', y='cnt', data=df, color='#88421D')
plt.title('Boxplot pentru Zi Lucrătoare')
plt.xlabel('Weekend/Sărbătoare (0) vs Media Zi Lucrătoare (1)')
plt.ylabel('Număr Închirieri (cnt)')

# Afisare boxplot-urilor
plt.tight_layout()
plt.show()
#%%
#Excludem coloanele care nu sunt variabile independente (instant, dteday, casual, registered)
# și calculăm corelația cu 'cnt'
corelatii_importante = df.drop(columns=['instant', 'dteday', 'casual', 'registered']).corr()['cnt'].sort_values(ascending=False)

print("Corelația variabilelor independente cu numărul de închirieri (cnt):")
print(corelatii_importante)

# Calculăm și diferența de medie pe ani pentru a susține importanța variabilei 'yr'
print("Media închirierilor pe ani (0=2018, 1=2019):")
print(df.groupby('yr')['cnt'].mean())

#%%
# Selectăm doar coloanele numerice pentru corelație
df_numerice = df.select_dtypes(include=['int64', 'float64'])

# Calculam matricea de corelatie
corelatie = df_numerice.corr()

# Afișăm harta de corelații
plt.figure(figsize=(12,6))
sns.heatmap(corelatie, annot=True, cmap="coolwarm")
plt.title("Matricea de corelații între variabile")
plt.show()
# %%

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Selectăm variabile pentru exemplu (cele mai importante)
X = df[['temp', 'atemp', 'hum', 'windspeed']]

# Calculăm VIF pentru fiecare variabilă
# Dacă VIF > 10 → coliniaritate puternică
vif = pd.DataFrame()
vif["Variabila"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print("VIF înainte de eliminare:")
print(vif)

# Temp si atemp au VIF mare, deci atemp poate fi eliminata din model.
# Eliminăm variabila atemp (are VIF foarte mare)
X = df[['temp', 'hum', 'windspeed']]

# Recalculăm VIF după eliminare
vif2 = pd.DataFrame()
vif2["Variabila"] = X.columns
vif2["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print("\nVIF după eliminarea variabilei 'atemp':")
print(vif2)
# %%
# Creăm o copie a dataframe-ului original
df_dummies = df.copy()

# Eliminăm coloanele care nu sunt necesare pentru model
df_dummies = df_dummies.drop(columns=['instant', 'dteday', 'casual', 'registered', 'atemp'])
df_dummies = pd.get_dummies(df_dummies, columns=['season', 'weathersit', 'mnth'], drop_first=True)

#  Convertim tot tabelul la tip numeric 
df_dummies = df_dummies.astype(float)

# Verificam noul set de date
print("Verificare tipuri de date (trebuie să fie toate float):")
print(df_dummies.dtypes.head())
# %%

# Efectuam Regresia Liniară 
import statsmodels.api as sm

df_model = df_dummies # Acum df_model este definit corect

X_clean = df_model.drop(columns=['cnt'])
y_clean = df_model['cnt']

# Adăugăm constanta
X_clean = sm.add_constant(X_clean)
model = sm.OLS(y_clean, X_clean).fit()
print(model.summary())
# %%
# %%
# Verifica distribuția variabilei dependente 'cnt'
plt.figure(figsize=(7,5))
sns.histplot(df['cnt'], bins=30, kde=True)
plt.title("Distribuția variabilei dependente (cnt)")
plt.xlabel("Număr închirieri")
plt.ylabel("Frecvență")
plt.show()

# %%
# Scatterplot pentru variabilele importante din regresie
plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
sns.scatterplot(x='temp', y='cnt', data=df)
plt.title("Temp vs cnt")

plt.subplot(1,3,2)
sns.scatterplot(x='hum', y='cnt', data=df)
plt.title("Hum vs cnt")

plt.subplot(1,3,3)
sns.scatterplot(x='windspeed', y='cnt', data=df)
plt.title("Windspeed vs cnt")

plt.tight_layout()
plt.show()
# %%
# Verificăm reziduurile modelului
residuurile = model.resid
plt.figure(figsize=(7,5))
sns.histplot(residuurile, bins=30, kde=True)
plt.title("Distribuția reziduurilor")
plt.xlabel("Reziduuri")
plt.ylabel("Frecvență")
plt.show()
# %%
# Intrebari si raspunsuri specifice setului nostru de date 
# Întrebarea 1: Care este luna cu cel mai mare număr mediu de închirieri?
top_luna = df.groupby('mnth')['cnt'].mean().idxmax()
valoare_luna = df.groupby('mnth')['cnt'].mean().max()
print(f"1. Luna cu cele mai multe închirieri medii: Luna {top_luna} (Media: {valoare_luna:.0f})")

# Întrebarea 2: Cât de mult scade cererea în zilele cu vreme nefavorabilă (3) față de cele cu vreme bună (1)?
vreme_buna = df[df['weathersit'] == 1]['cnt'].mean()
vreme_rea = df[df['weathersit'] == 3]['cnt'].mean()
scadere_procent = ((vreme_buna - vreme_rea) / vreme_buna) * 100
print(f"2. Scăderea cererii în zilele cu vreme rea: {scadere_procent:.2f}%")

# Întrebarea 3: Cine închiriază mai mult în weekend: utilizatorii ocazionali (casual) sau cei înregistrați (registered)?
weekend_data = df[df['workingday'] == 0]
media_casual = weekend_data['casual'].mean()
media_registered = weekend_data['registered'].mean()
print(f"3. În weekend, media utilizatorilor: Casual ({media_casual:.0f}) vs Registered ({media_registered:.0f})")

# Întrebarea 4: Care a fost ziua record (maximă) din întregul set de date?
zi_record = df.loc[df['cnt'].idxmax()]
print(f"4. Recordul absolut de închirieri: {zi_record['cnt']} unități în data {zi_record['dteday']}")

# Întrebarea 5: Cum a crescut segmentul de utilizatori 'Casual' față de 'Registered' între cei doi ani?
crestere_utilizatori = df.groupby('yr')[['casual', 'registered']].mean()
print("5. Media utilizatorilor pe ani (2018 vs 2019):")
print(crestere_utilizatori)

# Întrebarea 6: Care este impactul vântului puternic asupra numărului de închirieri?
# Considerăm vânt puternic peste 18 mph și vânt slab sub 10 mph
vant_puternic = df[df['windspeed'] > 18]['cnt'].mean()
vant_slab = df[df['windspeed'] < 10]['cnt'].mean()
print(f"6. Media închirierilor: Vânt puternic (>18): {vant_puternic:.0f} vs Vânt slab (<10): {vant_slab:.0f}")