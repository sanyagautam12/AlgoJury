import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def create_demo_dataset():
    """Create a synthetic hiring dataset with intentional bias patterns"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate base features
    data = {
        'age': np.random.normal(35, 10, n_samples).astype(int),
        'education_years': np.random.choice([12, 14, 16, 18, 20], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        'experience_years': np.random.exponential(5, n_samples).astype(int),
        'previous_salary': np.random.normal(60000, 20000, n_samples),
        'interview_score': np.random.normal(75, 15, n_samples),
        'technical_score': np.random.normal(80, 12, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
        'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], n_samples, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    }
    
    df = pd.DataFrame(data)
    
    # Ensure reasonable ranges
    df['age'] = np.clip(df['age'], 22, 65)
    df['experience_years'] = np.clip(df['experience_years'], 0, df['age'] - 18)
    df['previous_salary'] = np.clip(df['previous_salary'], 30000, 150000)
    df['interview_score'] = np.clip(df['interview_score'], 0, 100)
    df['technical_score'] = np.clip(df['technical_score'], 0, 100)
    
    # Create biased hiring decisions
    # Base probability from qualifications
    base_prob = (
        (df['education_years'] - 12) * 0.05 +
        (df['experience_years']) * 0.02 +
        (df['previous_salary'] - 30000) / 120000 * 0.3 +
        df['interview_score'] / 100 * 0.4 +
        df['technical_score'] / 100 * 0.3
    )
    
    # Add bias factors
    gender_bias = np.where(df['gender'] == 'Male', 0.15, -0.15)  # Male bias
    race_bias = np.where(df['race'] == 'White', 0.1, 
                np.where(df['race'] == 'Asian', 0.05, -0.1))  # White/Asian bias
    
    # Final hiring probability with bias
    hire_prob = np.clip(base_prob + gender_bias + race_bias, 0, 1)
    
    # Generate hiring decisions
    df['hired'] = np.random.binomial(1, hire_prob, n_samples)
    
    return df

def create_demo_model(df):
    """Train a model on the biased dataset"""
    # Prepare features for training
    df_model = df.copy()
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_race = LabelEncoder()
    
    df_model['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df_model['race_encoded'] = le_race.fit_transform(df['race'])
    
    # Define features for model training
    feature_columns = ['age', 'education_years', 'experience_years', 'previous_salary', 
                      'interview_score', 'technical_score', 'gender_encoded', 'race_encoded']
    
    X = df_model[feature_columns]
    y = df_model['hired']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Store encoders and feature info with the model
    model.label_encoders = {
        'gender': le_gender,
        'race': le_race
    }
    model.feature_columns = feature_columns
    
    return model

def main():
    """Create demo dataset and model"""
    print("Creating demo dataset...")
    df = create_demo_dataset()
    
    print(f"Dataset created with {len(df)} samples")
    print(f"Hiring rate: {df['hired'].mean():.2%}")
    print(f"Male hiring rate: {df[df['gender'] == 'Male']['hired'].mean():.2%}")
    print(f"Female hiring rate: {df[df['gender'] == 'Female']['hired'].mean():.2%}")
    
    print("\nTraining model...")
    model = create_demo_model(df)
    
    print("Saving files...")
    # Save dataset
    df.to_csv('demo_dataset.csv', index=False)
    print("Saved: demo_dataset.csv")
    
    # Save model
    joblib.dump(model, 'demo_model.pkl')
    print("Saved: demo_model.pkl")
    
    print("\nDemo files created successfully!")
    print("Dataset columns:", df.columns.tolist())
    print("Model features:", model.feature_columns)

if __name__ == '__main__':
    main()