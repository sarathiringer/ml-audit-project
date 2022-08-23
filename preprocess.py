import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import valohai


def main():
    """
    Load, preprocess and save the data.
    """

    # Specify columns to keep
    columns = ['age', 'workclass', 'education-num', 'hours-per-week', 'salary']

    # Fetch data
    data = pd.read_csv('data/adult.csv')[columns]
    target = data.pop('salary')

    # Fetch the categories of the variable 'workclass'
    workclass_categories = [
        'Private',
        'Self-emp-not-inc',
        'Self-emp-inc',
        'Federal-gov',
        'Local-gov',
        'State-gov',
        'Without-pay',
        'Never-worked',
    ]

    # Combine into a dictionary mapping category to the number in the dataframe
    workclass_mapping = dict(zip(np.sort(data.workclass.unique()), workclass_categories))

    # Replace the numerical value with the category
    data['workclass'] = data['workclass'].map(workclass_mapping)

    # Replace the missing values with 'Missing' - maybe there's an interesting pattern there?
    data['workclass'] = data['workclass'].fillna('Missing')

    # Turn the categorical column (workclass) into dummy features, aka One-Hot-Encoding
    data = pd.get_dummies(data)

    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Put back to pandas dataframe
    df = pd.DataFrame(data_scaled, columns=data.columns)

    # Add protected attribute and label to the dataset
    df['salary'] = target

    # Save the preprocessed data
    path = valohai.outputs().path('adult_preprocessed.csv')
    df.to_csv(path)
    print('Data preprocessed and uploaded')


if __name__ == '__main__':
    main()
