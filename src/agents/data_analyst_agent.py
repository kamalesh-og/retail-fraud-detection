import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class FraudDataAnalyst:
    """
    Intelligent Data Analyst Agent for Fraud Detection
    Performs automated EDA and generates actionable insights
    """
    
    def __init__(self):
        self.insights = []
        self.red_flags = []
        self.recommendations = []
        self.train_df = None
        self.test_df = None
        
    def analyze_fraud_data(self, train_path='data/train.csv', test_path='data/test.csv'):
        """
        Main analysis function - call this to run complete analysis
        """
        print("🤖 FRAUD DETECTION ANALYST AGENT ACTIVATED")
        print("="*50)
        
        try:
            # Load data
            self.load_data(train_path, test_path)
            
            # Run analysis pipeline
            self.dataset_overview()
            self.fraud_distribution_analysis()
            self.pricing_anomaly_detection()
            self.sales_person_behavior_analysis()
            self.product_analysis()
            self.automated_anomaly_detection()
            self.generate_fraud_insights()
            
            # Create summary report
            self.print_executive_summary()
            
        except Exception as e:
            print(f"❌ Error in analysis: {str(e)}")
            print("Make sure your CSV files are in the correct location!")
    
    def load_data(self, train_path, test_path):
        """Load and validate data"""
        print("📂 Loading datasets...")
        
        try:
            self.train_df = pd.read_csv(train_path)
            print(f"✅ Training data loaded: {self.train_df.shape}")
            
            if test_path:
                self.test_df = pd.read_csv(test_path)
                print(f"✅ Test data loaded: {self.test_df.shape}")
        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            print("Please make sure your CSV files are in the data/ folder")
            raise
    
    def dataset_overview(self):
        """Basic dataset overview"""
        print("\n📊 DATASET OVERVIEW")
        print("-" * 30)
        
        print(f"Training Records: {len(self.train_df):,}")
        print(f"Features: {len(self.train_df.columns)}")
        print(f"Columns: {list(self.train_df.columns)}")
        
        # Check for missing values
        missing = self.train_df.isnull().sum()
        if missing.sum() == 0:
            print("✅ No missing values found")
        else:
            print(f"⚠️ Missing values: {missing[missing > 0].to_dict()}")
        
        # Basic stats
        print("\nNumerical Features Summary:")
        print(self.train_df[['Quantity', 'TotalSalesValue']].describe())
    
    def fraud_distribution_analysis(self):
        """Analyze fraud label distribution"""
        print("\n🚨 FRAUD DISTRIBUTION ANALYSIS")
        print("-" * 35)
        
        if 'Suspicious' in self.train_df.columns:
            fraud_counts = self.train_df['Suspicious'].value_counts()
            fraud_pct = self.train_df['Suspicious'].value_counts(normalize=True) * 100
            
            print("Fraud Categories:")
            for category in fraud_counts.index:
                count = fraud_counts[category]
                pct = fraud_pct[category]
                print(f"  {category}: {count:,} ({pct:.1f}%)")
            
            # Flag imbalanced data
            if fraud_pct.min() < 10:
                self.red_flags.append("⚠️ Highly imbalanced dataset detected")
                self.recommendations.append("Consider using SMOTE or other balancing techniques")
        
        else:
            print("❌ No 'Suspicious' column found in training data")
    
    def pricing_anomaly_detection(self):
        """Detect pricing anomalies - key for fraud detection"""
        print("\n💰 PRICING ANOMALY ANALYSIS")
        print("-" * 32)
        
        # Calculate price per unit
        self.train_df['PricePerUnit'] = self.train_df['TotalSalesValue'] / self.train_df['Quantity']
        
        # Basic pricing stats
        price_stats = self.train_df['PricePerUnit'].describe()
        print("Price Per Unit Statistics:")
        print(f"  Mean: ${price_stats['mean']:.2f}")
        print(f"  Median: ${price_stats['50%']:.2f}")
        print(f"  Std Dev: ${price_stats['std']:.2f}")
        print(f"  Range: ${price_stats['min']:.2f} - ${price_stats['max']:.2f}")
        
        # Detect extreme pricing
        Q1 = self.train_df['PricePerUnit'].quantile(0.25)
        Q3 = self.train_df['PricePerUnit'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        price_outliers = self.train_df[
            (self.train_df['PricePerUnit'] < lower_bound) | 
            (self.train_df['PricePerUnit'] > upper_bound)
        ]
        
        outlier_pct = len(price_outliers) / len(self.train_df) * 100
        print(f"\n🔍 Price Outliers: {len(price_outliers)} ({outlier_pct:.1f}%)")
        
        if len(price_outliers) > 0:
            print("Extreme Pricing Examples:")
            extreme_examples = price_outliers.nlargest(3, 'PricePerUnit')[['ReportID', 'ProductID', 'PricePerUnit', 'Suspicious']]
            print(extreme_examples.to_string(index=False))
            
            self.insights.append(f"Found {len(price_outliers)} transactions with unusual pricing")
            
        # Product-wise pricing consistency
        product_price_std = self.train_df.groupby('ProductID')['PricePerUnit'].std().fillna(0)
        inconsistent_products = product_price_std[product_price_std > product_price_std.quantile(0.8)]
        
        if len(inconsistent_products) > 0:
            print(f"\n⚠️ Products with inconsistent pricing: {len(inconsistent_products)}")
            self.red_flags.append(f"{len(inconsistent_products)} products show high price variability")
    
    def sales_person_behavior_analysis(self):
        """Analyze sales person behavior patterns"""
        print("\n👤 SALES PERSON BEHAVIOR ANALYSIS")
        print("-" * 38)
        
        # Sales person performance metrics
        sp_metrics = self.train_df.groupby('SalesPersonID').agg({
            'TotalSalesValue': ['count', 'mean', 'sum', 'std'],
            'Quantity': ['mean', 'std'],
            'PricePerUnit': ['mean', 'std']
        }).round(2)
        
        # Flatten column names
        sp_metrics.columns = ['_'.join(col) for col in sp_metrics.columns]
        
        print(f"Total Sales Persons: {len(sp_metrics)}")
        print(f"Avg Transactions per Person: {sp_metrics['TotalSalesValue_count'].mean():.1f}")
        
        # Identify high-risk sales persons
        # High price variability
        high_price_var = sp_metrics[sp_metrics['PricePerUnit_std'] > sp_metrics['PricePerUnit_std'].quantile(0.8)]
        
        # Unusual transaction patterns
        unusual_patterns = sp_metrics[
            (sp_metrics['TotalSalesValue_std'] > sp_metrics['TotalSalesValue_std'].quantile(0.9)) |
            (sp_metrics['Quantity_std'] > sp_metrics['Quantity_std'].quantile(0.9))
        ]
        
        print(f"\n🚩 High-Risk Sales Persons:")
        print(f"  High Price Variability: {len(high_price_var)}")
        print(f"  Unusual Transaction Patterns: {len(unusual_patterns)}")
        
        if len(high_price_var) > 0:
            self.red_flags.append(f"{len(high_price_var)} sales persons show high pricing inconsistency")
            
        # Top performers vs risk indicators
        top_volume = sp_metrics.nlargest(3, 'TotalSalesValue_sum')
        print(f"\nTop 3 Sales Persons by Volume:")
        print(top_volume[['TotalSalesValue_sum', 'TotalSalesValue_count', 'PricePerUnit_std']].to_string())
    
    def product_analysis(self):
        """Analyze product-specific patterns"""
        print("\n📦 PRODUCT ANALYSIS")
        print("-" * 20)
        
        product_metrics = self.train_df.groupby('ProductID').agg({
            'TotalSalesValue': ['count', 'mean', 'sum'],
            'Quantity': 'mean',
            'PricePerUnit': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        product_metrics.columns = ['_'.join(col) for col in product_metrics.columns]
        
        print(f"Total Products: {len(product_metrics)}")
        
        # High-risk products (high price variability)
        risky_products = product_metrics[product_metrics['PricePerUnit_std'] > product_metrics['PricePerUnit_std'].quantile(0.8)]
        
        print(f"High Price Variability Products: {len(risky_products)}")
        
        if len(risky_products) > 0:
            print("\nTop Risky Products:")
            risk_examples = risky_products.nlargest(3, 'PricePerUnit_std')
            print(risk_examples[['TotalSalesValue_count', 'PricePerUnit_mean', 'PricePerUnit_std']].to_string())
            
            self.insights.append(f"{len(risky_products)} products show suspicious pricing patterns")
    
    def automated_anomaly_detection(self):
        """Use ML to detect anomalies automatically"""
        print("\n🤖 AUTOMATED ANOMALY DETECTION")
        print("-" * 35)
        
        # Prepare features for anomaly detection
        features = self.train_df[['Quantity', 'TotalSalesValue', 'PricePerUnit']].copy()
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        anomaly_labels = iso_forest.fit_predict(features)
        
        # Add anomaly flag
        self.train_df['ML_Anomaly'] = (anomaly_labels == -1)
        
        anomaly_count = sum(anomaly_labels == -1)
        anomaly_pct = anomaly_count / len(self.train_df) * 100
        
        print(f"🔍 ML Detected Anomalies: {anomaly_count} ({anomaly_pct:.1f}%)")
        
        if anomaly_count > 0:
            # Analyze ML-detected anomalies
            ml_anomalies = self.train_df[self.train_df['ML_Anomaly']]
            
            print("\nAnomaly Characteristics:")
            print(f"  Avg Quantity: {ml_anomalies['Quantity'].mean():.1f}")
            print(f"  Avg Sales Value: ${ml_anomalies['TotalSalesValue'].mean():.2f}")
            print(f"  Avg Price/Unit: ${ml_anomalies['PricePerUnit'].mean():.2f}")
            
            # Cross-reference with fraud labels
            if 'Suspicious' in self.train_df.columns:
                anomaly_fraud_overlap = ml_anomalies['Suspicious'].value_counts()
                print(f"\nAnomaly-Fraud Overlap:")
                for category, count in anomaly_fraud_overlap.items():
                    pct = count / len(ml_anomalies) * 100
                    print(f"  {category}: {count} ({pct:.1f}%)")
            
            self.insights.append(f"Machine Learning identified {anomaly_count} anomalous transactions")
    
    def generate_fraud_insights(self):
        """Generate specific fraud detection insights"""
        print("\n🧠 FRAUD PATTERN INSIGHTS")
        print("-" * 28)
        
        # Pattern 1: Price manipulation
        if 'PricePerUnit' in self.train_df.columns:
            # Find products sold at very different prices
            price_ranges = self.train_df.groupby('ProductID')['PricePerUnit'].agg(['min', 'max', 'std'])
            price_ranges['price_range'] = price_ranges['max'] - price_ranges['min']
            
            suspicious_price_ranges = price_ranges[price_ranges['price_range'] > price_ranges['price_range'].quantile(0.9)]
            
            if len(suspicious_price_ranges) > 0:
                self.insights.append(f"⚠️ {len(suspicious_price_ranges)} products sold at suspiciously different prices")
        
        # Pattern 2: Volume anomalies
        volume_outliers = self.train_df[self.train_df['Quantity'] > self.train_df['Quantity'].quantile(0.95)]
        if len(volume_outliers) > 0:
            self.insights.append(f"📊 {len(volume_outliers)} transactions with unusually high quantities")
        
        # Pattern 3: Sales person concentration
        sp_transaction_counts = self.train_df['SalesPersonID'].value_counts()
        high_volume_sp = sp_transaction_counts[sp_transaction_counts > sp_transaction_counts.quantile(0.8)]
        
        if len(high_volume_sp) > 0:
            self.recommendations.append(f"👤 Monitor {len(high_volume_sp)} high-volume sales persons closely")
    
    def print_executive_summary(self):
        """Print executive summary for management"""
        print("\n" + "="*60)
        print("📊 EXECUTIVE SUMMARY - FRAUD RISK ASSESSMENT")
        print("="*60)
        
        print(f"\n📈 DATASET OVERVIEW:")
        print(f"  • Total Transactions Analyzed: {len(self.train_df):,}")
        print(f"  • Sales Persons: {self.train_df['SalesPersonID'].nunique()}")
        print(f"  • Products: {self.train_df['ProductID'].nunique()}")
        print(f"  • Total Sales Value: ${self.train_df['TotalSalesValue'].sum():,.2f}")
        
        if 'Suspicious' in self.train_df.columns:
            fraud_dist = self.train_df['Suspicious'].value_counts(normalize=True) * 100
            print(f"\n🚨 FRAUD DISTRIBUTION:")
            for category, pct in fraud_dist.items():
                print(f"  • {category}: {pct:.1f}%")
        
        print(f"\n🔍 KEY INSIGHTS ({len(self.insights)} found):")
        for i, insight in enumerate(self.insights, 1):
            print(f"  {i}. {insight}")
        
        print(f"\n⚠️ RED FLAGS ({len(self.red_flags)} identified):")
        for i, flag in enumerate(self.red_flags, 1):
            print(f"  {i}. {flag}")
        
        print(f"\n💡 RECOMMENDATIONS ({len(self.recommendations)} suggested):")
        for i, rec in enumerate(self.recommendations, 1):
            print(f"  {i}. {rec}")
        
        print(f"\n🎯 FRAUD RISK SCORE: {self.calculate_risk_score()}/10")
        print("="*60)
    
    def calculate_risk_score(self):
        """Calculate overall fraud risk score"""
        score = 0
        
        # Base score from red flags
        score += min(len(self.red_flags) * 2, 6)
        
        # Add score from insights
        score += min(len(self.insights) * 0.5, 3)
        
        # Price variability factor
        if 'PricePerUnit' in self.train_df.columns:
            price_cv = self.train_df['PricePerUnit'].std() / self.train_df['PricePerUnit'].mean()
            if price_cv > 0.5:
                score += 1
        
        return min(int(score), 10)

# Usage example
if __name__ == "__main__":
    # Initialize the analyst agent
    analyst = FraudDataAnalyst()
    
    # Run the complete analysis
    # Make sure your CSV files are in the data folder!
    analyst.analyze_fraud_data('data/train.csv', 'data/test.csv')