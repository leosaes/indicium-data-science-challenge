import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def clean_data(df):

  df['Certificate'] = df['Certificate'].apply(lambda x: str(x).strip().upper() if pd.notna(x) else 'Unknown')


  df.loc[df['Released_Year'] == 'PG', 'Released_Year'] = 1995
  
  df['Released_Year'] = df['Released_Year'].apply(lambda x: int(str(x)) if pd.notna(x) else np.nan)


  df['Gross'] = df['Gross'].apply(lambda x: float(str(x).replace(',','')) if pd.notna(x) else np.nan)


  df['Runtime'] = df['Runtime'].str.extract(r'(\d+)').astype(float)


  top_directors = df['Director'].value_counts().head(50).index.tolist()

  df['Director'] = df['Director'].apply(lambda x: x if x in top_directors else 'Other_Directors')


  df['Stars'] = df[["Star1", "Star2", "Star3", "Star4"]].values.tolist()

  all_stars = pd.Series([s for stars in df['Stars'] for s in stars]).value_counts().head(100).index.tolist()
  
  df['Stars'] = df['Stars'].apply(lambda x: [s if s in all_stars else 'Other_Stars' for s in x])

  mlb_star = MultiLabelBinarizer()

  stars = pd.DataFrame(mlb_star.fit_transform(df['Stars']), columns=mlb_star.classes_,index=df.index)

  
  df['Genre'] = df['Genre'].str.split(', ')

  all_genres = pd.Series([g for genres in df['Genre'] for g in genres]).value_counts().head(15).index.tolist()
  
  df['Genre'] = df['Genre'].apply(lambda x: [g if g in all_genres else 'Other_Genres' for g in x])

  mlb_genre = MultiLabelBinarizer()

  genres = pd.DataFrame(mlb_genre.fit_transform(df['Genre']), columns=mlb_genre.classes_,index=df.index)

 
  df = pd.concat([df,genres,stars],axis=1)


  df["Overview"] = df["Overview"].str.lower()

  df.drop(columns=["Genre","Star1", "Star2", "Star3", "Star4"],axis=1,inplace=True)


  df.drop(columns=["Series_Title","Stars"],axis=1,inplace=True)

  return df


def fix_dataframe(new_df, expected_columns, fill_value=0):
  
  for col in expected_columns:
    if col not in new_df.columns:
      new_df[col] = fill_value
  
  new_df = new_df[expected_columns]

  return new_df  