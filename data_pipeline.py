import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.compose import ColumnTransformer

def read_data():
    df = pd.read_csv(r'H:\Term 5\ML\Students depression\data\raw\student_depression_dataset.csv')
    print(f"The data has been fully collected ✅. \nShape: {df.shape}\n")
    return df



def show_data(df):
    # First 10 rows
    print("\n🚩First 10 rows : \n")
    print(df.head(10))
    print()

    # Last 10 rows 
    print("\n🚩last 10 rows : \n")
    print(df.tail(10))
    print()

    # Data frame info
    print('\n📌Data fram info : \n')
    print(df.info())
    print()

def describe_df(df):
    # Describe of numerical columns
    print('\n📌Describe of numerical columns : \n')
    print(df.describe().T)
    print()

    # Describe of catgorical columns
    print('\n📌Describe of catgorical columns : \n')
    print(df.describe(include = "O").T)
    print()



def missing_values(df):
    print("\n🔍 Null values ratio per column:\n\n")
    for col in df.columns :
        print(f"Column : {col}")
        print(f"Missing values count = {df[col].isna().sum()}")
        print(f"Missing % = {(df[col].isna().sum()/len(df))*100}")
        print("="*50 + '\n')



def check_uniques(df):        
# Check the unique values in  columns  
    print('\n📌Check the unique values in  columns : \n')
    for col in df.columns :
        print(f'column : {col}')
        print(df[col].value_counts().head(30)) # preview top 10 unique values with count 
        print(f"\ncount of values :{len(df[col].unique())}") # preview the count of unique values
        print("="*50 + "\n")








def outliers_Checking(df):
    # check numerical columns to know how i handling
    numeric_columns = [ 'Age', 'Academic Pressure','Study Satisfaction','Work/Study Hours']
    print('\n📌Boxplt for Numerical columns\n\n')
    for col in numeric_columns :
        sns.boxplot(data = df , x = col) # boxplot figer
        plt.show()
    print()
    print()


    # check outlier percentage in num columns
    print("\n🔍 outlier values ratio per column:\n\n")
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25) 
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        print(f"Column: {col}")
        print(f"Number of outliers: {len(outliers)}")
        print(f'percentage of outlier = {(len(outliers)/len(df))*100}')
        print("-" * 20)
    print()
    print()

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]













def data_split(df):
    df = df.copy()
    
    df = df.drop("City",axis =1)
    
    X = df.drop('Depression', axis=1)
    Y = df['Depression']

    

    # Split to Train / Test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.30, random_state=42
    )



    return X_train, X_test, Y_train, Y_test



def preprosses():
    # Columns
    cat_col = ['Sleep Duration', 'Dietary Habits','Degree']
    yes_no_col = ['Family History of Mental Illness', "Have you ever had suicidal thoughts ?"]
    multi_cat_col = ["Gender"]
    num_col = ['Age','Academic Pressure', 'Study Satisfaction','Work/Study Hours',
               'Financial Stress','GPA']

    # ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[

        ('one_hot_main', OneHotEncoder(handle_unknown='ignore'), cat_col),
        ('one_hot_multi', OneHotEncoder(handle_unknown='ignore'), multi_cat_col),
        ('ordinal_yes_no', OrdinalEncoder(categories=[["No","Yes"]]*len(yes_no_col)), yes_no_col),
        ('num_scaler', RobustScaler(), num_col)

    ], remainder='drop')

    return preprocessor
    


def save_datafile(data, name):
    folder = r"data\preprocessed"
    os.makedirs(folder, exist_ok=True)

    parquet_file = fr"{folder}\{name}.parquet"

    # Convert DataFrame to Parquet using PyArrow
    table = pa.Table.from_pandas(data, preserve_index=False)

    # Save file
    pq.write_table(table, parquet_file, compression="snappy")

    print(f"✔ Saved {name}.parquet | Shape: {data.shape}")



def run_data_pipeline():
    read_data()
    dataframe = read_data()
    df = dataframe.copy()

    show_data(df)
    describe_df(df)
    missing_values(df)
    check_uniques(df)
    degree_mapping = {
        

        # Bachelor level
        "'Class 12'": "Bachelor",
        "BSc": "Bachelor",
        "BA": "Bachelor",
        "BCA": "Bachelor",
        "B.Ed": "Bachelor",
        "BHM": "Bachelor",
        "B.Pharm": "Bachelor",
        "BE": "Bachelor",
        "B.Com": "Bachelor",
        "B.Arch": "Bachelor",
        "B.Tech": "Bachelor",
        "BBA": "Business",

        # Master level
        "M.Tech": "Master",
        "MSc": "Master",
        "MA": "Master",
        "M.Ed": "Master",
        "MHM": "Master",
        "M.Pharm": "Master",
        "MCA": "Master",
        "MBA": "Master",
        "M.Com": "Master",
        "ME": "Master",
        "MD": "Master",

        # Others
        "MBBS": "Others",
        "LLB": "Others",
        "LLM": "Others",
        "PhD": "Others",
        "Others": "Others"
    }
    df['Degree'] = df['Degree'].map(degree_mapping)
    df.drop("id", axis = 1 , inplace = True)
    df = df[df['Sleep Duration'] != 'Others']
    df = df[df['Academic Pressure']!=0.0]
    df = df[df['Study Satisfaction']!=0.0]
    df = df[df["Dietary Habits"]!="Others"]
    print("Duplicates = ",df.duplicated().sum())

    print('Wrong values was changed✅')


    df["GPA"] = df["CGPA"] * 0.4
    df.drop(["CGPA","Job Satisfaction","Work Pressure","Profession"], axis=1, inplace=True)





    outliers_Checking(df)
    df = remove_outliers(df, "Age")

    df.reset_index()
    Data = df.copy()

    X_train, X_test, Y_train, Y_test = data_split(Data)

    datasets = {
    "data": Data,
    "X_train": X_train,
    "X_test": X_test,
    "Y_train": Y_train,
    "Y_test": Y_test
}

    for name, df in datasets.items():
        if isinstance(df, pd.Series):
            df = df.to_frame(name) 
        save_datafile(df, name)



if __name__ == "__main__":
    run_data_pipeline()