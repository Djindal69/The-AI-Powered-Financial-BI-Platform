import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timezone

# Generate sample financial data
def generate_sample_data(n=1000):
    np.random.seed(42)
    
    data = []
    loan_purposes = ['personal', 'home_improvement', 'debt_consolidation', 'business', 'auto', 'education']
    home_ownership = ['rent', 'own', 'mortgage', 'other']
    loan_statuses = ['approved', 'denied', 'default']
    
    for i in range(n):
        age = np.random.randint(18, 80)
        income = np.random.lognormal(10.5, 0.7)  # Log-normal distribution for income
        credit_score = np.random.normal(650, 100)
        credit_score = max(300, min(850, credit_score))  # Clamp to valid range
        
        # Create some correlations
        debt_to_income = np.random.beta(2, 5) * 0.6  # Beta distribution for DTI
        employment_length = np.random.poisson(5)  # Poisson for employment length
        loan_amount = np.random.lognormal(9.5, 0.8)  # Log-normal for loan amount
        
        # Risk-based loan approval
        risk_score = (credit_score / 850 * 0.4 + 
                     (1 - debt_to_income) * 0.3 + 
                     min(employment_length / 10, 1) * 0.2 + 
                     min(income / 100000, 1) * 0.1)
        
        if risk_score > 0.6:
            status = np.random.choice(['approved', 'denied'], p=[0.85, 0.15])
        elif risk_score > 0.4:
            status = np.random.choice(['approved', 'denied'], p=[0.6, 0.4])
        else:
            status = np.random.choice(['approved', 'denied', 'default'], p=[0.3, 0.6, 0.1])
        
        data.append({
            'customer_id': str(uuid.uuid4()),
            'age': int(age),
            'income': round(income, 2),
            'credit_score': int(credit_score),
            'debt_to_income': round(debt_to_income, 3),
            'employment_length': int(employment_length),
            'loan_amount': round(loan_amount, 2),
            'loan_purpose': np.random.choice(loan_purposes),
            'home_ownership': np.random.choice(home_ownership),
            'annual_income': round(income, 2),
            'loan_status': status
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data(1000)
    
    # Save to CSV
    df.to_csv('/app/backend/sample_financial_data.csv', index=False)
    print(f"Generated {len(df)} sample records")
    print("\nData summary:")
    print(df.describe())
    print("\nLoan status distribution:")
    print(df['loan_status'].value_counts())