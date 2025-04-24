import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def preprocess_data(data):
    """Handle missing values and feature engineering"""
    # Calculate medians/modes first
    age_median = data['Age'].median()
    fare_median = data['Fare'].median()
    embarked_mode = data['Embarked'].mode()[0]
    
    # Create copies to avoid chained assignment
    data = data.copy()
    
    # Fill missing values
    data['Age'] = data['Age'].fillna(age_median)
    data['Fare'] = data['Fare'].fillna(fare_median)
    data['Embarked'] = data['Embarked'].fillna(embarked_mode)
    
    # Convert categorical variables
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data = pd.get_dummies(data, columns=['Embarked'])
    
    # Feature engineering
    data['FamilySize'] = data['SibSp'] + data['Parch']
    data['IsAlone'] = (data['FamilySize'] == 0).astype(int)
    
    return data

def plot_visualizations(data, model=None, X_test=None, y_test=None):
    """Generate all visualizations"""
    plt.figure(figsize=(15, 10))
    
    # 1. Survival Rate by Gender (Pie Chart)
    plt.subplot(2, 2, 1)
    gender_survival = data.groupby('Sex')['Survived'].mean()
    labels = ['Male', 'Female']
    colors = ['#ff9999', '#66b3ff']
    plt.pie(gender_survival, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Survival Rate by Gender')
    
    # 2. Survival by Passenger Class (Bar Plot)
    plt.subplot(2, 2, 2)
    sns.barplot(x='Pclass', y='Survived', data=data, palette='viridis')
    plt.title('Survival Rate by Ticket Class')
    plt.xlabel('Class (1 = Highest)')
    plt.ylabel('Survival Rate')
    
    # 3. Age Distribution (Histogram + KDE)
    plt.subplot(2, 2, 3)
    sns.histplot(data=data, x='Age', hue='Survived', kde=True, bins=20, 
                 palette=['#ff6b6b', '#4ecdc4'], alpha=0.6)
    plt.title('Age Distribution: Survivors vs Non-Survivors')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend(['Died', 'Survived'])
    
    # 4. Correlation Heatmap
    plt.subplot(2, 2, 4)
    sns.heatmap(data[['Survived', 'Pclass', 'Age', 'Fare', 'FamilySize']].corr(), 
                annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Feature Importance (if model is provided)
    if model:
        plt.figure(figsize=(10, 5))
        importances = model.feature_importances_
        features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 
                   'Embarked_C', 'Embarked_Q', 'Embarked_S']
        sns.barplot(x=importances, y=features, palette='rocket')
        plt.title('Feature Importances')
        plt.xlabel('Importance Score')
        plt.show()
    
    # 6. Confusion Matrix (if test data is provided)
    if X_test is not None and y_test is not None:
        plt.figure(figsize=(6, 6))
        predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Died', 'Survived'], 
                    yticklabels=['Died', 'Survived'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

def main():
    # Load data
    try:
        data = pd.read_csv('Titanic-Dataset.csv')
    except FileNotFoundError:
        print("Error: Could not find 'Titanic-Dataset.csv'")
        print("Please ensure the file exists in the current directory.")
        return

    print("\nPreprocessing data...")
    data = preprocess_data(data)

    # Select features
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 
                'Embarked_C', 'Embarked_Q', 'Embarked_S']
    X = data[features]
    y = data['Survived']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel Accuracy: {accuracy:.2%}")

    # Feature importance
    importances = model.feature_importances_
    print("\nFeature Importances:")
    for feature, importance in zip(features, importances):
        print(f"{feature}: {importance:.2%}")

    # Generate all visualizations
    print("\nGenerating visualizations...")
    plot_visualizations(data, model, X_test, y_test)

if __name__ == "__main__":
    main()