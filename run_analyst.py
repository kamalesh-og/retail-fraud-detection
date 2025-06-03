# run_analyst.py
# Simple script to run the fraud detection analyst

import sys
import os
sys.path.append('src/agents')

from data_analyst_agent import FraudDataAnalyst

def main():
    print("🤖 FRAUD DETECTION ANALYST")
    print("=" * 40)
    
    # Initialize the analyst
    analyst = FraudDataAnalyst()
    
    # Run analysis with your data files
    try:
        analyst.analyze_fraud_data(
            train_path='data/raw/train.csv',
            test_path='data/raw/test.csv'
        )
        
        print("\n🎉 Analysis Complete!")
        
    except FileNotFoundError:
        print("❌ CSV files not found!")
        print("Make sure you have:")
        print("  - data/raw/train.csv")
        print("  - data/raw/test.csv")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()