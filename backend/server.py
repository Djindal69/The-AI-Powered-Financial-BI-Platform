from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb
import joblib
import json
import io
from emergentintegrations.llm.chat import LlmChat, UserMessage

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LLM Integration Setup
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')

# Models
class FinancialData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str
    age: int
    income: float
    credit_score: int
    debt_to_income: float
    employment_length: int
    loan_amount: float
    loan_purpose: str
    home_ownership: str
    annual_income: float
    loan_status: str  # 'approved', 'denied', 'default'
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ModelTrainingRequest(BaseModel):
    dataset_name: str
    model_type: str  # 'random_forest', 'gradient_boosting', 'neural_network', 'xgboost'
    target_column: str
    feature_columns: List[str]
    test_size: float = 0.2

class ModelPrediction(BaseModel):
    model_id: str
    predictions: List[Dict[str, Any]]
    confidence_scores: List[float]
    risk_assessment: str

class TrainedModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    feature_importance: Dict[str, float]
    training_data_size: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AIInsight(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    insight_type: str  # 'risk_analysis', 'model_explanation', 'data_insights'
    content: str
    confidence: float
    generated_by: str  # 'gpt', 'claude'
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Utility functions
def prepare_data_for_ml(df: pd.DataFrame, target_column: str, feature_columns: List[str]):
    """Prepare data for machine learning"""
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    # Encode categorical variables
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object' and col in feature_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Encode target column if it's categorical
    target_encoder = None
    if df[target_column].dtype == 'object':
        target_encoder = LabelEncoder()
        df[target_column] = target_encoder.fit_transform(df[target_column].astype(str))
        label_encoders[target_column] = target_encoder
    
    X = df[feature_columns]
    y = df[target_column]
    
    return X, y, label_encoders

def calculate_feature_importance(model, feature_names):
    """Calculate feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return {}
    
    return dict(zip(feature_names, importances.tolist()))

async def generate_ai_insight(prompt: str, model_type: str = 'gpt') -> str:
    """Generate AI insights using GPT or Claude"""
    try:
        if model_type == 'gpt':
            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=f"financial_insights_{uuid.uuid4()}",
                system_message="You are a financial risk analysis expert specializing in machine learning model interpretation and business intelligence insights."
            ).with_model("openai", "gpt-4o")
        else:  # claude
            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=f"financial_insights_{uuid.uuid4()}",
                system_message="You are a financial risk analysis expert specializing in advanced statistical modeling and business intelligence."
            ).with_model("anthropic", "claude-3-7-sonnet-20250219")
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        return response
    except Exception as e:
        logger.error(f"Error generating AI insight: {e}")
        return f"Unable to generate insight: {str(e)}"

# API Endpoints
@api_router.get("/")
async def root():
    return {"message": "AI-Powered Financial BI Platform API", "version": "1.0", "status": "active"}

@api_router.post("/financial-data", response_model=FinancialData)
async def create_financial_data(data: FinancialData):
    """Create new financial data entry"""
    try:
        data_dict = data.dict()
        data_dict['created_at'] = data_dict['created_at'].isoformat()
        await db.financial_data.insert_one(data_dict)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/financial-data", response_model=List[FinancialData])
async def get_financial_data():
    """Get all financial data"""
    try:
        data_list = await db.financial_data.find().to_list(1000)
        for item in data_list:
            if isinstance(item.get('created_at'), str):
                item['created_at'] = datetime.fromisoformat(item['created_at'])
        return [FinancialData(**item) for item in data_list]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV data for analysis"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Convert DataFrame to financial data records
        records = []
        for _, row in df.iterrows():
            financial_data = {
                'id': str(uuid.uuid4()),
                'customer_id': str(row.get('customer_id', uuid.uuid4())),
                'age': int(row.get('age', 35)),
                'income': float(row.get('income', 50000)),
                'credit_score': int(row.get('credit_score', 650)),
                'debt_to_income': float(row.get('debt_to_income', 0.3)),
                'employment_length': int(row.get('employment_length', 5)),
                'loan_amount': float(row.get('loan_amount', 10000)),
                'loan_purpose': str(row.get('loan_purpose', 'personal')),
                'home_ownership': str(row.get('home_ownership', 'rent')),
                'annual_income': float(row.get('annual_income', row.get('income', 50000))),
                'loan_status': str(row.get('loan_status', 'approved')),
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            records.append(financial_data)
        
        # Insert into database
        if records:
            await db.financial_data.insert_many(records)
        
        return {"message": f"Successfully uploaded {len(records)} records", "records_count": len(records)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading CSV: {str(e)}")

@api_router.post("/train-model", response_model=TrainedModel)
async def train_model(request: ModelTrainingRequest):
    """Train a machine learning model"""
    try:
        # Get data from database
        data_list = await db.financial_data.find().to_list(10000)
        if not data_list:
            raise HTTPException(status_code=400, detail="No data available for training")
        
        df = pd.DataFrame(data_list)
        
        # Prepare data
        X, y, label_encoders = prepare_data_for_ml(df, request.target_column, request.feature_columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=request.test_size, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model based on type
        if request.model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None
            
        elif request.model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None
            
        elif request.model_type == 'xgboost':
            model = xgb.XGBClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None
            
        elif request.model_type == 'neural_network':
            # Prepare neural network
            input_dim = X_train_scaled.shape[1]
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
            
            y_pred_proba = model.predict(X_test_scaled).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        roc_auc = 0.0
        if y_pred_proba is not None and len(np.unique(y)) == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importance
        feature_importance = calculate_feature_importance(model, request.feature_columns)
        
        # Save model
        model_data = TrainedModel(
            name=f"{request.model_type}_{request.dataset_name}",
            model_type=request.model_type,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            feature_importance=feature_importance,
            training_data_size=len(X_train)
        )
        
        # Save to database
        model_dict = model_data.dict()
        model_dict['created_at'] = model_dict['created_at'].isoformat()
        await db.trained_models.insert_one(model_dict)
        
        # Save model file
        model_path = f"/tmp/model_{model_data.id}.joblib"
        joblib.dump({'model': model, 'scaler': scaler, 'label_encoders': label_encoders}, model_path)
        
        return model_data
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@api_router.get("/models", response_model=List[TrainedModel])
async def get_trained_models():
    """Get all trained models"""
    try:
        models = await db.trained_models.find().to_list(100)
        for model in models:
            if isinstance(model.get('created_at'), str):
                model['created_at'] = datetime.fromisoformat(model['created_at'])
        return [TrainedModel(**model) for model in models]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/generate-insights")
async def generate_insights(data: dict):
    """Generate AI-powered insights"""
    try:
        insight_type = data.get('type', 'risk_analysis')
        context = data.get('context', '')
        
        # Create prompts based on insight type
        if insight_type == 'risk_analysis':
            prompt = f"""
            Analyze the following financial risk data and provide actionable insights:
            {context}
            
            Please provide:
            1. Key risk factors identified
            2. Risk assessment summary
            3. Recommendations for risk mitigation
            4. Market trends that might affect these risks
            """
        elif insight_type == 'model_explanation':
            prompt = f"""
            Explain the machine learning model performance and results:
            {context}
            
            Please provide:
            1. Model performance interpretation
            2. Feature importance explanation
            3. Business implications
            4. Recommendations for model improvement
            """
        else:
            prompt = f"""
            Analyze the financial data patterns and provide business insights:
            {context}
            
            Please provide:
            1. Data patterns identified
            2. Business opportunities
            3. Potential concerns
            4. Strategic recommendations
            """
        
        # Generate insights from both GPT and Claude
        gpt_insight = await generate_ai_insight(prompt, 'gpt')
        claude_insight = await generate_ai_insight(prompt, 'claude')
        
        # Save insights
        insights = []
        for content, model_name in [(gpt_insight, 'gpt'), (claude_insight, 'claude')]:
            insight = AIInsight(
                insight_type=insight_type,
                content=content,
                confidence=0.85,  # Mock confidence score
                generated_by=model_name
            )
            
            insight_dict = insight.dict()
            insight_dict['created_at'] = insight_dict['created_at'].isoformat()
            await db.ai_insights.insert_one(insight_dict)
            insights.append(insight)
        
        return {"insights": insights, "total_generated": len(insights)}
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")

@api_router.get("/insights", response_model=List[AIInsight])
async def get_insights():
    """Get all AI insights"""
    try:
        insights = await db.ai_insights.find().sort("created_at", -1).to_list(50)
        for insight in insights:
            if isinstance(insight.get('created_at'), str):
                insight['created_at'] = datetime.fromisoformat(insight['created_at'])
        return [AIInsight(**insight) for insight in insights]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/analytics/summary")
async def get_analytics_summary():
    """Get comprehensive analytics summary"""
    try:
        # Get data counts
        total_records = await db.financial_data.count_documents({})
        total_models = await db.trained_models.count_documents({})
        total_insights = await db.ai_insights.count_documents({})
        
        # Get sample data for analysis
        sample_data = await db.financial_data.find().limit(1000).to_list(1000)
        
        if sample_data:
            df = pd.DataFrame(sample_data)
            
            # Calculate basic statistics
            avg_income = df['income'].mean() if 'income' in df.columns else 0
            avg_credit_score = df['credit_score'].mean() if 'credit_score' in df.columns else 0
            approval_rate = len(df[df['loan_status'] == 'approved']) / len(df) * 100 if 'loan_status' in df.columns else 0
            
            # Risk distribution
            risk_distribution = {}
            if 'loan_status' in df.columns:
                risk_distribution = df['loan_status'].value_counts().to_dict()
        else:
            avg_income = avg_credit_score = approval_rate = 0
            risk_distribution = {}
        
        return {
            "summary": {
                "total_records": total_records,
                "total_models": total_models,
                "total_insights": total_insights,
                "average_income": round(avg_income, 2),
                "average_credit_score": round(avg_credit_score, 2),
                "approval_rate": round(approval_rate, 2)
            },
            "risk_distribution": risk_distribution
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include router and shutdown handler
app.include_router(api_router)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()