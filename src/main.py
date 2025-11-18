import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import plotly.express as px
from IPython.display import display, Latex, Markdown
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import csv
import warnings
warnings.filterwarnings('ignore')



# df.head()

def load_data():
    df = pd.read_csv("../data/Influenza__COVID-19__RSV__and_Other_Respiratory_Virus_Laboratory_Surveillance.csv", na_filter=False)
    vaccinations_by_ethinicity = pd.read_csv("../data/COVID-19_Vaccinations_by_Age_and_Race-Ethnicity_-_Historical-2.csv", na_filter=False)
    return df, vaccinations_by_ethinicity

def clean_data(df, vaccinations_by_ethinicity):
    df = df[df['season'] > '2018-2019']
    df_covid_era = df.copy()
    df_covid_era.reset_index(drop=True)
    df_covid_era.dropna(inplace=True)
    df_covid_era['week_start'] = pd.to_datetime(df_covid_era['week_start'])
    df_covid_era['week_end'] = pd.to_datetime(df_covid_era['week_end'])

    covid_data = df_covid_era[(df_covid_era['week_start'] >= '2020-01-01') & (df_covid_era['pathogen']  == 'SARS-CoV-2')]
    vaccinations_by_ethinicity = pd.read_csv("../data/COVID-19_Vaccinations_by_Age_and_Race-Ethnicity_-_Historical-2.csv", na_filter=False)
    vaccinations_by_ethinicity.rename(columns={'Week End': 'week_end'}, inplace=True)
    vaccinations_by_ethinicity['week_end'] = pd.to_datetime(vaccinations_by_ethinicity['week_end'], format='%m/%d/%y')
    df_final = pd.merge(covid_data, vaccinations_by_ethinicity, on='week_end', how='inner')
    df_final.dropna(inplace=True)
    df_final.replace({"" : np.nan, None : np.nan, "NA" : np.nan, "N/A" : np.nan}, inplace=True)
    df_final.to_csv('../data/final_df.csv', index=False)
    df_final_copy = df_final.copy()
    df_final_copy.replace({"" : np.nan, None : np.nan, "NA" : np.nan, "N/A" : np.nan}, inplace=True)
    return df_final_copy

# df1, df2 = load_data()
# df_final = clean_data(df1, df2)
# print(df_final.head())
# print(df_final.describe())

def exploratory_data_analysis_1(df_final_copy):
    plt.figure(figsize=(8, 4))
    ax = sns.lineplot(data=df_final_copy, x='week_start', y='Vaccine Series Completed Percent', hue='Race/Ethnicity')
    plt.title('Vaccination Series Completion Percent Over Time by Ethnicity')
    plt.xlabel('Week Start')
    plt.ylabel('Vaccine Series Completed Percent (%)')
    # Positioning the legend outside the plot
    plt.legend(title='Ethnicity', loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlap
    plt.show()


# exploratory_data_analysis_1(df_final)


def exploratory_data_analysis_2(df_final_copy):
    df_processed = pd.get_dummies(df_final_copy, columns=['Race/Ethnicity'], drop_first=True)
    df_processed.shape, df_processed.head()
    # Ensure 'Vaccine Series Completed Percent' is numeric
    df_processed['Vaccine Series Completed Percent'] = pd.to_numeric(df_processed['Vaccine Series Completed Percent'], errors='coerce')

    # Since we've coerced errors, there might be NaNs introduced if there were non-numeric values. Let's drop these.
    df_processed.dropna(subset=['Vaccine Series Completed Percent'], inplace=True)

    latest_week = df_processed['week_start'].max()
    demographic_columns = [col for col in df_processed.columns if col.startswith('Race/Ethnicity_')]

    # Retry summarizing the average vaccination completion percent across different demographics for the latest week
    avg_vaccination_by_demographic = df_processed[df_processed['week_start'] == latest_week][demographic_columns + ['Vaccine Series Completed Percent']].groupby(demographic_columns).mean().reset_index()

    # Melt the DataFrame to make it suitable for seaborn barplot again
    avg_vaccination_melted = avg_vaccination_by_demographic.melt(id_vars=['Vaccine Series Completed Percent'], var_name='Demographic Group', value_name='Presence')

    # Filter out rows where the demographic group is not present
    avg_vaccination_melted_filtered = avg_vaccination_melted[avg_vaccination_melted['Presence'] == 1]

    bar_colors = ['skyblue', 'orange', 'green', 'red', 'purple']


    # Visualize again
    plt.figure(figsize=(6, 4))
    sns.barplot(data=avg_vaccination_melted_filtered, x='Demographic Group', y='Vaccine Series Completed Percent', errorbar=None, palette=bar_colors, hue='Demographic Group', legend=False)
    plt.title('Vaccination Completion Percent by Demographic Group for the Latest Week')
    plt.xlabel('Demographic Group')
    plt.ylabel('Average Vaccination Completion Percent')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()

# exploratory_data_analysis_2(df_final)


def preprocessing(df_final_copy):

    df_processed = pd.get_dummies(df_final_copy, columns=['Race/Ethnicity'], drop_first=True)
    # df_processed.shape, df_processed.head()
    # Ensure 'Vaccine Series Completed Percent' is numeric
    df_processed['Vaccine Series Completed Percent'] = pd.to_numeric(df_processed['Vaccine Series Completed Percent'], errors='coerce')
    df_processed.dropna(subset=['Vaccine Series Completed Percent'], inplace=True)
    return df_processed

def ML_sets(df_processed):
    # Convert 'week_start' to datetime format and calculate the number of days since the first record
    df_processed['week_start'] = pd.to_datetime(df_processed['week_start'])
    df_processed['days_since_start'] = (df_processed['week_start'] - df_processed['week_start'].min()).dt.days

    # Feature selection
    demographic_columns = [col for col in df_processed.columns if col.startswith('Race/Ethnicity_')]
    X_features = ['days_since_start', 'lab_tot_positive'] + demographic_columns
    y_feature = 'Vaccine Series Completed Percent'

    df_cleaned = df_processed.dropna(subset=[y_feature])

    # Prepare the data again with the cleaned DataFrame
    X_cleaned = df_cleaned[X_features]
    y_cleaned = df_cleaned[y_feature]

    # Split the data into training and testing sets
    X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)
    return X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned, X_features

def base_line_MSE(df_final_copy):
    df_processed = preprocessing(df_final_copy)
    X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned, X_features = ML_sets(df_processed)
    # Calculate the mean of the target variable from the training data
    mean_vaccination_rate = y_train_cleaned.mean()

    # Create an array of the same mean value to serve as our baseline predictions
    baseline_predictions = np.full(shape=y_test_cleaned.shape, fill_value=mean_vaccination_rate)

    # Calculate the Mean Squared Error (MSE) for the baseline model
    baseline_mse = mean_squared_error(y_test_cleaned, baseline_predictions)

    # Return the baseline MSE
    return baseline_mse

# print(f"Base line MSE: {base_line_MSE(df_final)}")

def train_and_evaluate_model(df_final):
    df_processed = preprocessing(df_final)
    X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned, X_features = ML_sets(df_processed)


    # Re-initialize and re-train the Linear Regression model on the cleaned data
    model_cleaned = LinearRegression()
    model_cleaned.fit(X_train_cleaned, y_train_cleaned)

    # Predict on the testing set
    y_pred_cleaned = model_cleaned.predict(X_test_cleaned)

    # Calculate the Mean Squared Error (MSE) for the test set with cleaned data
    mse_cleaned = mean_squared_error(y_test_cleaned, y_pred_cleaned)

    # Retrieve the model coefficients
    coefficients = model_cleaned.coef_

    # Create a DataFrame to display feature names alongside their coefficients
    coefficients_df = pd.DataFrame(data={'Feature': X_features, 'Coefficient': coefficients})

    # Display the DataFrame sorted by the absolute value of coefficients for better interpretation
    coefficients_df = coefficients_df.sort_values(by='Coefficient', key=abs, ascending=False)

    return mse_cleaned, coefficients_df, y_pred_cleaned, y_test_cleaned

def create_model_performane_graph( y_test_cleaned, y_pred_cleaned):
    # Scatter plot of Actual vs. Predicted values
    plt.figure(figsize=(5, 3))
    plt.scatter(y_test_cleaned, y_pred_cleaned, alpha=0.5, color='blue')
    plt.title('Actual vs. Predicted Vaccination Completion Percent')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([y_test_cleaned.min(), y_test_cleaned.max()], [y_test_cleaned.min(), y_test_cleaned.max()], 'k--', lw=2)
    plt.tight_layout()

    # Show the plot
    plt.show()


# mse_cleaned, coefficients_df = train_and_evaluate_model(df_final)

# print(f"Model MSE: {mse_cleaned}")
# print(coefficients_df)

def exploratory_data_analysis_3():
    #OBESITY BLOCK

    # Load dataframe2 from the CSV file
    df2 = pd.read_csv("../data/Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System (1).csv")

    # Convert any columns containing year information to datetime type
    # For example, if there's a column named 'YearStart'
    df2['YearStart'] = pd.to_datetime(df2['YearStart'], format='%Y')

    # Filter dataframe2 to include only rows with data from 2018 and above
    df2_filtered = df2[df2['YearStart'].dt.year >= 2019]
    df2_filtered['YearStart'] = pd.to_datetime(df2_filtered['YearStart']).dt.year # Modify this line
    df2_fs = df2_filtered.sort_values(by='YearStart', ascending=True)
    #df2_filtered['YearStart'] = pd.to_datetime(df2_filtered['YearStart']).dt.year

    df2_fs.dropna(subset=['Race/Ethnicity'], inplace=True)
    df2_fs = df2_fs[['YearStart','YearEnd','Race/Ethnicity','Sample_Size']]


    df2_fs['Race/Ethnicity'] = df2_fs['Race/Ethnicity'].replace({
        'Non-Hispanic Black': '"Black, non-Latinx"',
        'Non-Hispanic White': '"White, non-Latinx"',
        'Hispanic': 'Latinx',
        'Asian': '"Asian, non-Latinx"',
        '2 or more races': '"Other, non-Latinx"',
        'Other': '"Other, non-Latinx"',
        'American Indian/Alaska Native': '"Other, non-Latinx"',
        'Hawaiian/Pacific Islander': '"Other, non-Latinx"'
    })



    #df2_fs.to_csv("obesity2.csv", index=False)
    df2_fs['Race/Ethnicity'] = df2_fs['Race/Ethnicity'].str.replace('"', '')
    df2_fs.to_csv('obesity2.csv', index=False, quoting=csv.QUOTE_ALL)

    # Extracting the year from the 'Week End' column
    vaccineO_df = pd.read_csv("../data/COVID-19_Vaccinations_by_Age_and_Race-Ethnicity_-_Historical-2.csv")

    vaccineO_df.replace('', pd.NA, inplace=True)
    vaccineO_df = vaccineO_df.dropna(subset=['Week End'])

    #NaN
    vaccineO_df.fillna(0, inplace=True)
    numeric_columns = ['Population Size', '1st Dose', 'Vaccine Series Completed', 'Boosted', 'Bivalent']
    vaccineO_df[numeric_columns] = vaccineO_df[numeric_columns].apply(pd.to_numeric)


    vaccineO_df['Week End'] = pd.to_datetime(vaccineO_df['Week End'])
    vaccineO_df['Year'] = vaccineO_df['Week End'].dt.year
    vaccineO_df = vaccineO_df[vaccineO_df['Year'] <= 2022]


    # Grouping by 'Year' and 'Race/Ethnicity', and aggregating the columns
    grouped_df = vaccineO_df.groupby(['Year', 'Race/Ethnicity']).agg({
        'Population Size': 'sum',
        '1st Dose': 'sum',
        '1st Dose Percent': 'mean',
        'Vaccine Series Completed': 'sum',
        'Vaccine Series Completed Percent': 'mean', 
        'Boosted': 'sum',
        'Boosted Percent': 'mean', 
        'Bivalent': 'sum',
        'Bivalent Percent': 'mean'  
    }).reset_index()

    # Optionally, you can rename the columns for clarity
    grouped_df.columns = ['Year', 'Race/Ethnicity', 'Total Population', 'Total 1st Dose', 'Average 1st Dose Percent',
                        'Total Vaccine Series Completed', 'Average Vaccine Series Completed Percent', 'Total Boosted',
                        'Average Boosted Percent', 'Total Bivalent', 'Average Bivalent Percent']

    grouped_df.to_csv('vaccine_Demographics.csv', index=False)
    vaccine_df = grouped_df
    obesity_df = df2_fs

    df_obesity = obesity_df
    df_vaccine = vaccine_df
    df_vaccine = df_vaccine[df_vaccine['Race/Ethnicity'] != 'All']
    # Filter relevant columns for vaccine data
    vaccine_cols = ['Year', 'Race/Ethnicity', 'Total Population', 'Total 1st Dose', 'Total Bivalent']
    df_vaccine = df_vaccine[vaccine_cols]

    # Filter relevant columns for obesity data
    obesity_cols = ['YearStart', 'Race/Ethnicity', 'Sample_Size']
    df_obesity = df_obesity[obesity_cols]

    # Rename columns in obesity dataframe to match vaccine dataframe
    df_obesity = df_obesity.rename(columns={'YearStart': 'Year'})

    # Group vaccine data by year and ethnicity and sum the values
    df_vaccine_grouped = df_vaccine.groupby(['Year', 'Race/Ethnicity']).sum().reset_index()

    # Group obesity data by year and ethnicity and sum the values
    df_obesity_grouped = df_obesity.groupby(['Year', 'Race/Ethnicity']).sum().reset_index()
    plt.figure(figsize=(7, 4))

    for ethnicity in df_vaccine_grouped['Race/Ethnicity'].unique():
        df_vaccine_ethnicity = df_vaccine_grouped[df_vaccine_grouped['Race/Ethnicity'] == ethnicity]
        plt.plot(df_vaccine_ethnicity['Year'], df_vaccine_ethnicity['Total 1st Dose'], label=f'{ethnicity} - Vaccine 1st Dose', linestyle='-')
        
        # Plotting bivalent if available
        if 'Total Bivalent' in df_vaccine_grouped.columns:
            plt.plot(df_vaccine_ethnicity['Year'], df_vaccine_ethnicity['Total Bivalent'], linestyle='-.', label=f'{ethnicity} - Vaccine Bivalent')

    for ethnicity in df_obesity_grouped['Race/Ethnicity'].unique():
        df_obesity_ethnicity = df_obesity_grouped[df_obesity_grouped['Race/Ethnicity'] == ethnicity]
        plt.plot(df_obesity_ethnicity['Year'], df_obesity_ethnicity['Sample_Size'], label=f'{ethnicity} - Obesity', linestyle=':')

    plt.title('Vaccine Metrics and Obesity by Race/Ethnicity Over Time')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.legend()
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.grid(True)
    plt.show()

# exploratory_data_analysis_3()