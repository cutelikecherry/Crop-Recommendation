import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_and_export_model(data_path="Crop_recommendation.csv"):
    #Load Dataset
    df = pd.read_csv(data_path)
    
    #Features: N, P, K, temperature, humidity, ph, rainfall
    #Target: label
    X = df.drop(columns=['label'])
    y = df['label']
    
    #Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    #Model Initialization & Training
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    #Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy of MODEL: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))
    
    #Pre-calculate Class-wise Feature Medians for the Deficit Engine
    #Calculates ideal N, P, K, pH profiles per crop type
    
    crop_profiles = df.groupby('label')[['N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall']].median().to_dict(orient='index')
    
    #Save Artifacts
    artifacts = {
        "model": model,
        "feature_names": list(X.columns),
        "crop_list": sorted(list(y.unique())),
        "crop_profiles": crop_profiles
    }
    
    with open("model.pkl", "wb") as f:
        pickle.dump(artifacts, f)
        
    print("Model saved successfully to 'model.pkl'.")

if __name__ == "__main__":
    train_and_export_model()