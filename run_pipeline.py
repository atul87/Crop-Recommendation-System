"""
Complete ML Pipeline for Crop Recommendation System
Runs data cleaning, analysis, and model training
"""

import subprocess
import sys

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}\n")
    
    try:
        subprocess.run([sys.executable, script_name],
                       capture_output=False,
                       text=True,
                       check=True)
        print(f"\n[SUCCESS] {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Error in {description}")
        print(f"Error: {e}")
        return False

def main():
    """Run the complete ML pipeline"""
    print("\n" + "="*60)
    print("CROP RECOMMENDATION SYSTEM - ML PIPELINE")
    print("="*60)
    
    steps = [
        ("data_preprocessing.py", "Data Cleaning"),
        ("data_analysis.py", "Data Analysis & EDA"),
        ("train_models.py", "Model Training")
    ]
    
    for script, description in steps:
        success = run_script(script, description)
        if not success:
            print(f"\nPipeline stopped due to error in {description}")
            return
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated Files:")
    print("  - data/Crop_recommendations_cleaned.csv")
    print("  - data/data_core_cleaned.csv")
    print("  - data/*_correlation.png")
    print("  - data/*_distributions.png")
    print("  - models/crop_recommendation_model.pkl")
    print("  - models/fertilizer_recommendation_model.pkl")
    print("  - models/crop_type_prediction_model.pkl")
    print("  - models/*_confusion_matrix.png")
    print("\nYou can now use these models for predictions!")

if __name__ == "__main__":
    main()
