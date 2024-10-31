import pandas as pd
def load_data(file_path):
    return pd.read_json(file_path)

def extract_ingredients(ingredients):
    list_of_ingredients = []
    for ingredient in ingredients:
        list_of_ingredients.append(ingredient['name'])
    return list_of_ingredients

def preprocess_data(df):
    df['ingredients'] = df['ingredients'].apply(extract_ingredients)    
    ingredients_columns = df['ingredients'].str.join("|").str.get_dummies()
    tags_columns = df['tags'].str.join("|").str.get_dummies()
    # glass_columns = df['glass'].str.join("|").str.get_dummies()  Ahh
    glass_columns = pd.get_dummies(df['glass'])
    df_encoded = pd.concat([df,ingredients_columns,glass_columns],axis=1)

    df_encoded.drop(['id','name','category','glass','instructions','imageUrl','createdAt','updatedAt','tags','ingredients','alcoholic'],axis=1,inplace=True)

    return df_encoded



