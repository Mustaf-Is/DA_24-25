import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


file_path = 'Accident_Information.csv'
df = pd.read_csv(file_path, dtype={'Accident_Index': str}, low_memory=False)

df = df.drop(['Carriageway_Hazards', 'Did_Police_Officer_Attend_Scene_of_Accident','Police_Force',
              'Special_Conditions_at_Site','Year','InScotland', 'Accident_Index', 'Date'], axis=1)

# Selektimi i rreshtave 800000 - 900000 (inclusive)
df = df.iloc[800000:900000]

df['Pedestrian_Crossing'] = df['Pedestrian_Crossing-Human_Control'] + df['Pedestrian_Crossing-Physical_Facilities']

# Zëvendësimi i vlerave që mungojnë (NULL) me modën e kolonës pasi që janë të dhëna kategorike
lsoa_mode = df['LSOA_of_Accident_Location'].mode()[0]
df['LSOA_of_Accident_Location'] = df['LSOA_of_Accident_Location'].fillna(lsoa_mode)

# Konvertimi i vlerave negative në pozitive
df['Longitude'] = df['Longitude'].abs()

# Largimi i numrave dhjetorë duke lënë vetëm 1 shifër pas presjes dhjetore
df['Longitude'] = df['Longitude'].round(1)
df['Latitude'] = df['Latitude'].round(1)

# Scale i të dhënave numerike në intervalin [0,1]
scaler = MinMaxScaler()
df['Location_Easting_OSGR'] = scaler.fit_transform(df['Location_Easting_OSGR'].values.reshape(-1, 1))
df['Location_Northing_OSGR'] = scaler.fit_transform(df['Location_Northing_OSGR'].values.reshape(-1, 1))

# Kategorizimi i çmimeve në bazë të shkallës së vlerave
if 'Speed_limit' in df.columns:
    scale_speed = df['Speed_limit']
    intervals = [0, 30, 60, 90]
    labels = [1, 2, 3]  # 1: low, 2: medium, 3: high

    # Bashkimi ne kategori
    df['Speed_limit'] = pd.cut(scale_speed, bins=intervals, labels=labels, include_lowest=True)

    #  Mbushja e vlerave qe mungojne me moden e kategorive te bashkuara
    mode_speed = df['Speed_limit'].mode()[0]
    df['Speed_limit'] = df['Speed_limit'].fillna(mode_speed).astype(int)


# Konvertimi i kolonave kategorike në vlera numerike
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))



# Ruaj rezultatin në një fajll të ri CSV
df.to_csv('Processed_Accidents.csv', index=False)
print(df.head())
