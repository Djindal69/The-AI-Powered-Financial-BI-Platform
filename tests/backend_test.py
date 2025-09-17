import requests
import sys
import json
import time
from datetime import datetime

class FinancialBITester:
    def __init__(self, base_url="https://aifusion-project.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.sample_data = None

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'} if not files else {}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files, timeout=60)
                else:
                    response = requests.post(url, json=data, headers=headers, timeout=60)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    if isinstance(response_data, dict) and len(str(response_data)) < 500:
                        print(f"   Response: {response_data}")
                    elif isinstance(response_data, list):
                        print(f"   Response: List with {len(response_data)} items")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"   Error: {error_detail}")
                except:
                    print(f"   Error: {response.text[:200]}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test root API endpoint"""
        return self.run_test("Root API", "GET", "", 200)

    def test_analytics_summary(self):
        """Test analytics summary endpoint"""
        success, data = self.run_test("Analytics Summary", "GET", "analytics/summary", 200)
        if success and data:
            print(f"   📊 Total Records: {data.get('summary', {}).get('total_records', 0)}")
            print(f"   📊 Total Models: {data.get('summary', {}).get('total_models', 0)}")
            print(f"   📊 Total Insights: {data.get('summary', {}).get('total_insights', 0)}")
        return success, data

    def test_get_financial_data(self):
        """Test getting financial data"""
        success, data = self.run_test("Get Financial Data", "GET", "financial-data", 200)
        if success and data:
            self.sample_data = data[:5] if len(data) > 5 else data
            print(f"   📈 Retrieved {len(data)} financial records")
        return success, data

    def test_create_financial_data(self):
        """Test creating financial data"""
        test_data = {
            "customer_id": f"test_customer_{int(time.time())}",
            "age": 35,
            "income": 75000.0,
            "credit_score": 720,
            "debt_to_income": 0.25,
            "employment_length": 8,
            "loan_amount": 25000.0,
            "loan_purpose": "home_improvement",
            "home_ownership": "own",
            "annual_income": 75000.0,
            "loan_status": "approved"
        }
        return self.run_test("Create Financial Data", "POST", "financial-data", 200, test_data)

    def test_train_model(self, model_type="random_forest"):
        """Test training ML model"""
        training_request = {
            "dataset_name": f"test_dataset_{model_type}",
            "model_type": model_type,
            "target_column": "loan_status",
            "feature_columns": ["age", "income", "credit_score", "debt_to_income", "employment_length", "loan_amount"],
            "test_size": 0.2
        }
        
        print(f"   🤖 Training {model_type} model...")
        success, data = self.run_test(f"Train {model_type.title()} Model", "POST", "train-model", 200, training_request)
        
        if success and data:
            print(f"   📊 Model Accuracy: {data.get('accuracy', 0) * 100:.1f}%")
            print(f"   📊 Model Precision: {data.get('precision', 0) * 100:.1f}%")
            print(f"   📊 Model Recall: {data.get('recall', 0) * 100:.1f}%")
            print(f"   📊 F1-Score: {data.get('f1_score', 0) * 100:.1f}%")
        
        return success, data

    def test_get_models(self):
        """Test getting trained models"""
        success, data = self.run_test("Get Trained Models", "GET", "models", 200)
        if success and data:
            print(f"   🤖 Found {len(data)} trained models")
            for model in data[:3]:  # Show first 3 models
                print(f"      - {model.get('name', 'Unknown')}: {model.get('accuracy', 0) * 100:.1f}% accuracy")
        return success, data

    def test_generate_insights(self, insight_type="risk_analysis"):
        """Test AI insights generation"""
        context_data = {
            "sample_data": self.sample_data[:3] if self.sample_data else [],
            "timestamp": datetime.now().isoformat(),
            "analysis_type": insight_type
        }
        
        insight_request = {
            "type": insight_type,
            "context": json.dumps(context_data)
        }
        
        print(f"   🧠 Generating {insight_type} insights using GPT and Claude...")
        success, data = self.run_test(f"Generate {insight_type.title()} Insights", "POST", "generate-insights", 200, insight_request)
        
        if success and data:
            insights = data.get('insights', [])
            print(f"   💡 Generated {len(insights)} insights")
            for insight in insights:
                model_name = insight.get('generated_by', 'Unknown').upper()
                content_preview = insight.get('content', '')[:100] + "..." if len(insight.get('content', '')) > 100 else insight.get('content', '')
                print(f"      - {model_name}: {content_preview}")
        
        return success, data

    def test_get_insights(self):
        """Test getting AI insights"""
        success, data = self.run_test("Get AI Insights", "GET", "insights", 200)
        if success and data:
            print(f"   💡 Found {len(data)} AI insights")
            for insight in data[:2]:  # Show first 2 insights
                model_name = insight.get('generated_by', 'Unknown').upper()
                insight_type = insight.get('insight_type', 'Unknown').replace('_', ' ').title()
                print(f"      - {model_name} {insight_type}")
        return success, data

    def test_csv_upload(self):
        """Test CSV upload functionality"""
        # Create a simple test CSV
        csv_content = """customer_id,age,income,credit_score,debt_to_income,employment_length,loan_amount,loan_purpose,home_ownership,annual_income,loan_status
test_001,28,45000,680,0.35,3,15000,personal,rent,45000,approved
test_002,35,65000,720,0.25,7,20000,home_improvement,own,65000,approved
test_003,42,55000,650,0.40,5,12000,debt_consolidation,mortgage,55000,denied"""
        
        files = {'file': ('test_data.csv', csv_content, 'text/csv')}
        return self.run_test("CSV Upload", "POST", "upload-csv", 200, files=files)

def main():
    print("🚀 Starting AI-Powered Financial BI Platform Backend Tests")
    print("=" * 60)
    
    tester = FinancialBITester()
    
    # Test basic endpoints
    print("\n📋 BASIC API TESTS")
    print("-" * 30)
    tester.test_root_endpoint()
    tester.test_analytics_summary()
    
    # Test data operations
    print("\n📊 DATA MANAGEMENT TESTS")
    print("-" * 30)
    tester.test_get_financial_data()
    tester.test_create_financial_data()
    tester.test_csv_upload()
    
    # Test ML model training
    print("\n🤖 MACHINE LEARNING TESTS")
    print("-" * 30)
    
    # Test different model types
    model_types = ["random_forest", "gradient_boosting", "xgboost"]
    for model_type in model_types:
        success, _ = tester.test_train_model(model_type)
        if not success:
            print(f"⚠️  {model_type} training failed, continuing with other tests...")
        time.sleep(2)  # Brief pause between model training
    
    # Neural network might take longer, test separately
    print("\n🧠 Testing Neural Network (may take longer)...")
    tester.test_train_model("neural_network")
    
    tester.test_get_models()
    
    # Test AI insights generation
    print("\n💡 AI INSIGHTS TESTS")
    print("-" * 30)
    
    insight_types = ["risk_analysis", "model_explanation", "data_insights"]
    for insight_type in insight_types:
        success, _ = tester.test_generate_insights(insight_type)
        if not success:
            print(f"⚠️  {insight_type} generation failed, continuing...")
        time.sleep(3)  # AI generation takes time
    
    tester.test_get_insights()
    
    # Final results
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed! Backend is working perfectly.")
        return 0
    elif tester.tests_passed / tester.tests_run >= 0.8:
        print("✅ Most tests passed! Backend is mostly functional.")
        return 0
    else:
        print("❌ Many tests failed! Backend needs attention.")
        return 1

if __name__ == "__main__":
    sys.exit(main())