# First, you need to import the Pandas library
import pandas as pd

# Creating a DataFrame from a dictionary
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']}
df = pd.DataFrame(data)

# Displaying the DataFrame
print("DataFrame:")
print(df)

# Accessing columns
print("\nAccessing columns:")
print(df['Name'])  # Accessing a single column
print(df[['Name', 'Age']])  # Accessing multiple columns

# Accessing rows
print("\nAccessing rows:")
print(df.loc[0])  # Accessing a single row by label
print(df.iloc[1])  # Accessing a single row by integer index
print(df[1:3])  # Accessing multiple rows by slicing

# Adding a new column
df['Gender'] = ['Female', 'Male', 'Male', 'Male']
print("\nAfter adding a new column:")
print(df)

# Removing a column
df.drop(columns=['City'], inplace=True)
print("\nAfter removing a column:")
print(df)

# Filtering data
print("\nFiltering data:")
print(df[df['Age'] > 30])  # Filtering based on a condition

# Sorting data
print("\nSorting data:")
print(df.sort_values(by='Age', ascending=False))  # Sorting by Age in descending order

# Basic statistics
print("\nBasic statistics:")
print(df.describe())  # Summary statistics for numerical columns

# Reading from and writing to files
df.to_csv('data.csv', index=False)  # Writing to a CSV file
file_path = r"C:\Users\asus\Downloads\city_day.csv"
df_from_csv = pd.read_csv(file_path)  # Reading from a CSV file
print("\nDataFrame from CSV:")
print(df_from_csv)

