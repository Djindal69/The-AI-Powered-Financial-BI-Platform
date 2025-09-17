import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Textarea } from './components/ui/textarea';
import { Badge } from './components/ui/badge';
import { Progress } from './components/ui/progress';
import { toast } from 'sonner';
import { motion } from 'framer-motion';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, Brain, Database, BarChart3, Upload, Zap, Target, Shield, Activity, Users, DollarSign, AlertTriangle } from 'lucide-react';
import { useDropzone } from 'react-dropzone';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Dashboard = () => {
  const [analytics, setAnalytics] = useState(null);
  const [models, setModels] = useState([]);
  const [insights, setInsights] = useState([]);
  const [financialData, setFinancialData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [trainingModel, setTrainingModel] = useState(false);

  // Model training form state
  const [modelForm, setModelForm] = useState({
    dataset_name: 'financial_risk',
    model_type: 'random_forest',
    target_column: 'loan_status',
    feature_columns: ['age', 'income', 'credit_score', 'debt_to_income', 'employment_length', 'loan_amount'],
    test_size: 0.2
  });

  const [insightForm, setInsightForm] = useState({
    type: 'risk_analysis',
    context: ''
  });

  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    try {
      setLoading(true);
      await Promise.all([
        fetchAnalytics(),
        fetchModels(),
        fetchInsights(),
        fetchFinancialData()
      ]);
    } catch (error) {
      console.error('Error fetching initial data:', error);
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const fetchAnalytics = async () => {
    try {
      const response = await axios.get(`${API}/analytics/summary`);
      setAnalytics(response.data);
    } catch (error) {
      console.error('Error fetching analytics:', error);
    }
  };

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API}/models`);
      setModels(response.data);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const fetchInsights = async () => {
    try {
      const response = await axios.get(`${API}/insights`);
      setInsights(response.data);
    } catch (error) {
      console.error('Error fetching insights:', error);
    }
  };

  const fetchFinancialData = async () => {
    try {
      const response = await axios.get(`${API}/financial-data`);
      setFinancialData(response.data.slice(0, 100)); // Limit for performance
    } catch (error) {
      console.error('Error fetching financial data:', error);
    }
  };

  const handleTrainModel = async () => {
    try {
      setTrainingModel(true);
      const response = await axios.post(`${API}/train-model`, modelForm);
      toast.success('Model trained successfully!');
      await fetchModels();
    } catch (error) {
      console.error('Error training model:', error);
      toast.error('Failed to train model');
    } finally {
      setTrainingModel(false);
    }
  };

  const handleGenerateInsights = async () => {
    try {
      setLoading(true);
      const contextData = {
        analytics_summary: analytics?.summary,
        recent_models: models.slice(0, 3),
        data_sample: financialData.slice(0, 10)
      };

      const response = await axios.post(`${API}/generate-insights`, {
        type: insightForm.type,
        context: JSON.stringify(contextData)
      });
      
      toast.success(`Generated ${response.data.total_generated} new insights!`);
      await fetchInsights();
    } catch (error) {
      console.error('Error generating insights:', error);
      toast.error('Failed to generate insights');
    } finally {
      setLoading(false);
    }
  };

  // CSV Upload
  const onDrop = async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const formData = new FormData();
      formData.append('file', file);
      
      try {
        setLoading(true);
        const response = await axios.post(`${API}/upload-csv`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        toast.success(response.data.message);
        await fetchAnalytics();
        await fetchFinancialData();
      } catch (error) {
        console.error('Error uploading file:', error);
        toast.error('Failed to upload file');
      } finally {
        setLoading(false);
      }
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false
  });

  // Chart data preparation
  const performanceData = models.map(model => ({
    name: model.name,
    accuracy: model.accuracy * 100,
    precision: model.precision * 100,
    recall: model.recall * 100,
    f1_score: model.f1_score * 100
  }));

  const riskDistributionData = analytics?.risk_distribution ? 
    Object.entries(analytics.risk_distribution).map(([key, value]) => ({
      name: key,
      value: value,
      color: key === 'approved' ? '#10B981' : key === 'denied' ? '#EF4444' : '#F59E0B'
    })) : [];

  const incomeDistribution = financialData.reduce((acc, item) => {
    const range = item.income < 30000 ? '< 30K' : 
                  item.income < 50000 ? '30K - 50K' :
                  item.income < 80000 ? '50K - 80K' :
                  item.income < 120000 ? '80K - 120K' : '> 120K';
    acc[range] = (acc[range] || 0) + 1;
    return acc;
  }, {});

  const incomeData = Object.entries(incomeDistribution).map(([range, count]) => ({
    range,
    count
  }));

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              AI-Powered Financial BI Platform
            </h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Advanced machine learning models, automated insights, and comprehensive risk analysis for modern financial intelligence
          </p>
        </motion.div>

        {/* Key Metrics */}
        {analytics && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
          >
            <Card className="border-0 shadow-lg bg-gradient-to-br from-blue-500 to-blue-600 text-white">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-blue-100 text-sm font-medium">Total Records</p>
                    <p className="text-3xl font-bold">{analytics.summary.total_records.toLocaleString()}</p>
                  </div>
                  <Database className="w-8 h-8 text-blue-200" />
                </div>
              </CardContent>
            </Card>

            <Card className="border-0 shadow-lg bg-gradient-to-br from-purple-500 to-purple-600 text-white">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-purple-100 text-sm font-medium">AI Models</p>
                    <p className="text-3xl font-bold">{analytics.summary.total_models}</p>
                  </div>
                  <Target className="w-8 h-8 text-purple-200" />
                </div>
              </CardContent>
            </Card>

            <Card className="border-0 shadow-lg bg-gradient-to-br from-green-500 to-green-600 text-white">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-green-100 text-sm font-medium">Approval Rate</p>
                    <p className="text-3xl font-bold">{analytics.summary.approval_rate}%</p>
                  </div>
                  <TrendingUp className="w-8 h-8 text-green-200" />
                </div>
              </CardContent>
            </Card>

            <Card className="border-0 shadow-lg bg-gradient-to-br from-orange-500 to-orange-600 text-white">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-orange-100 text-sm font-medium">AI Insights</p>
                    <p className="text-3xl font-bold">{analytics.summary.total_insights}</p>
                  </div>
                  <Zap className="w-8 h-8 text-orange-200" />
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Main Content Tabs */}
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5 bg-white/50 backdrop-blur-sm border-0 shadow-lg">
            <TabsTrigger value="overview" className="data-[state=active]:bg-white data-[state=active]:shadow-md">
              <BarChart3 className="w-4 h-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="data" className="data-[state=active]:bg-white data-[state=active]:shadow-md">
              <Database className="w-4 h-4 mr-2" />
              Data Management
            </TabsTrigger>
            <TabsTrigger value="models" className="data-[state=active]:bg-white data-[state=active]:shadow-md">
              <Target className="w-4 h-4 mr-2" />
              ML Models
            </TabsTrigger>
            <TabsTrigger value="insights" className="data-[state=active]:bg-white data-[state=active]:shadow-md">
              <Brain className="w-4 h-4 mr-2" />
              AI Insights
            </TabsTrigger>
            <TabsTrigger value="analytics" className="data-[state=active]:bg-white data-[state=active]:shadow-md">
              <Activity className="w-4 h-4 mr-2" />
              Analytics
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Risk Distribution */}
              <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Shield className="w-5 h-5 text-blue-600" />
                    Risk Distribution
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={riskDistributionData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        dataKey="value"
                      >
                        {riskDistributionData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Income Distribution */}
              <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <DollarSign className="w-5 h-5 text-green-600" />
                    Income Distribution
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={incomeData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="range" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count" fill="#3B82F6" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>

            {/* Model Performance */}
            {performanceData.length > 0 && (
              <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-purple-600" />
                    Model Performance Comparison
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={performanceData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="accuracy" stroke="#3B82F6" strokeWidth={2} />
                      <Line type="monotone" dataKey="precision" stroke="#10B981" strokeWidth={2} />
                      <Line type="monotone" dataKey="recall" stroke="#F59E0B" strokeWidth={2} />
                      <Line type="monotone" dataKey="f1_score" stroke="#8B5CF6" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Data Management Tab */}
          <TabsContent value="data" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Upload Data */}
              <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Upload className="w-5 h-5 text-blue-600" />
                    Upload Financial Data
                  </CardTitle>
                  <CardDescription>
                    Upload CSV files with financial data for analysis
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div
                    {...getRootProps()}
                    className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                      isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'
                    }`}
                  >
                    <input {...getInputProps()} />
                    <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    {isDragActive ? (
                      <p className="text-blue-600">Drop the CSV file here...</p>
                    ) : (
                      <div>
                        <p className="text-gray-600 mb-2">Drag & drop a CSV file here, or click to select</p>
                        <p className="text-sm text-gray-400">Supports CSV files up to 10MB</p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Data Summary */}
              <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Database className="w-5 h-5 text-green-600" />
                    Data Summary
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {analytics && (
                    <>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-600">Total Records:</span>
                        <Badge variant="outline">{analytics.summary.total_records.toLocaleString()}</Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-600">Average Income:</span>
                        <Badge variant="outline">${analytics.summary.average_income.toLocaleString()}</Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-600">Average Credit Score:</span>
                        <Badge variant="outline">{analytics.summary.average_credit_score}</Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-600">Approval Rate:</span>
                        <Badge variant="outline">{analytics.summary.approval_rate}%</Badge>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* ML Models Tab */}
          <TabsContent value="models" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Train New Model */}
              <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="w-5 h-5 text-purple-600" />
                    Train New Model
                  </CardTitle>
                  <CardDescription>
                    Create advanced ML models for financial risk analysis
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="model-type">Model Type</Label>
                    <Select value={modelForm.model_type} onValueChange={(value) => setModelForm({...modelForm, model_type: value})}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="random_forest">Random Forest</SelectItem>
                        <SelectItem value="gradient_boosting">Gradient Boosting</SelectItem>
                        <SelectItem value="xgboost">XGBoost</SelectItem>
                        <SelectItem value="neural_network">Neural Network</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div>
                    <Label htmlFor="dataset-name">Dataset Name</Label>
                    <Input 
                      value={modelForm.dataset_name}
                      onChange={(e) => setModelForm({...modelForm, dataset_name: e.target.value})}
                      placeholder="Enter dataset name"
                    />
                  </div>

                  <Button 
                    onClick={handleTrainModel} 
                    disabled={trainingModel}
                    className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
                  >
                    {trainingModel ? 'Training...' : 'Train Model'}
                  </Button>
                </CardContent>
              </Card>

              {/* Trained Models */}
              <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="w-5 h-5 text-green-600" />
                    Trained Models
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4 max-h-96 overflow-y-auto">
                    {models.length === 0 ? (
                      <p className="text-gray-500 text-center py-8">No models trained yet</p>
                    ) : (
                      models.map((model) => (
                        <div key={model.id} className="border rounded-lg p-4 bg-gray-50">
                          <div className="flex justify-between items-start mb-3">
                            <h4 className="font-semibold text-gray-800">{model.name}</h4>
                            <Badge variant="secondary">{model.model_type}</Badge>
                          </div>
                          <div className="grid grid-cols-2 gap-2 text-sm">
                            <div>Accuracy: <span className="font-medium">{(model.accuracy * 100).toFixed(1)}%</span></div>
                            <div>Precision: <span className="font-medium">{(model.precision * 100).toFixed(1)}%</span></div>
                            <div>Recall: <span className="font-medium">{(model.recall * 100).toFixed(1)}%</span></div>
                            <div>F1-Score: <span className="font-medium">{(model.f1_score * 100).toFixed(1)}%</span></div>
                          </div>
                          <Progress value={model.accuracy * 100} className="mt-3" />
                        </div>
                      ))
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* AI Insights Tab */}
          <TabsContent value="insights" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Generate Insights */}
              <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="w-5 h-5 text-indigo-600" />
                    Generate AI Insights
                  </CardTitle>
                  <CardDescription>
                    Leverage GPT and Claude for automated financial analysis
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="insight-type">Insight Type</Label>
                    <Select value={insightForm.type} onValueChange={(value) => setInsightForm({...insightForm, type: value})}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="risk_analysis">Risk Analysis</SelectItem>
                        <SelectItem value="model_explanation">Model Explanation</SelectItem>
                        <SelectItem value="data_insights">Data Insights</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <Button 
                    onClick={handleGenerateInsights} 
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
                  >
                    {loading ? 'Generating...' : 'Generate Insights'}
                  </Button>
                </CardContent>
              </Card>

              {/* Recent Insights */}
              <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="w-5 h-5 text-yellow-600" />
                    Recent Insights
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4 max-h-96 overflow-y-auto">
                    {insights.length === 0 ? (
                      <p className="text-gray-500 text-center py-8">No insights generated yet</p>
                    ) : (
                      insights.slice(0, 5).map((insight) => (
                        <div key={insight.id} className="border rounded-lg p-4 bg-gray-50">
                          <div className="flex justify-between items-start mb-2">
                            <Badge variant={insight.generated_by === 'gpt' ? 'default' : 'secondary'}>
                              {insight.generated_by === 'gpt' ? 'GPT' : 'Claude'}
                            </Badge>
                            <span className="text-xs text-gray-500">
                              {new Date(insight.created_at).toLocaleDateString()}
                            </span>
                          </div>
                          <h4 className="font-semibold text-gray-800 mb-2 capitalize">
                            {insight.insight_type.replace('_', ' ')}
                          </h4>
                          <p className="text-sm text-gray-600 line-clamp-3">
                            {insight.content.substring(0, 200)}...
                          </p>
                        </div>
                      ))
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Full Insights List */}
            <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <CardTitle>All AI Insights</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {insights.map((insight) => (
                    <div key={insight.id} className="border-l-4 border-blue-500 pl-4 py-2">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant={insight.generated_by === 'gpt' ? 'default' : 'secondary'}>
                          {insight.generated_by.toUpperCase()}
                        </Badge>
                        <Badge variant="outline" className="capitalize">
                          {insight.insight_type.replace('_', ' ')}
                        </Badge>
                        <span className="text-sm text-gray-500 ml-auto">
                          {new Date(insight.created_at).toLocaleString()}
                        </span>
                      </div>
                      <div className="prose prose-sm max-w-none">
                        <p className="text-gray-700 whitespace-pre-line">{insight.content}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Key Performance Indicators */}
              <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-green-600" />
                    Key Performance Indicators
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {analytics && (
                    <>
                      <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
                        <span className="font-medium text-blue-800">Data Quality Score</span>
                        <div className="text-right">
                          <div className="text-2xl font-bold text-blue-600">87%</div>
                          <div className="text-xs text-blue-500">Excellent</div>
                        </div>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
                        <span className="font-medium text-green-800">Model Accuracy Avg</span>
                        <div className="text-right">
                          <div className="text-2xl font-bold text-green-600">
                            {models.length > 0 ? Math.round(models.reduce((acc, m) => acc + m.accuracy, 0) / models.length * 100) : 0}%
                          </div>
                          <div className="text-xs text-green-500">High Performance</div>
                        </div>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-orange-50 rounded-lg">
                        <span className="font-medium text-orange-800">Risk Alert Level</span>
                        <div className="text-right">
                          <div className="text-2xl font-bold text-orange-600">Medium</div>
                          <div className="text-xs text-orange-500">Monitor closely</div>
                        </div>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>

              {/* System Health */}
              <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="w-5 h-5 text-purple-600" />
                    System Health
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Data Processing</span>
                        <span>95%</span>
                      </div>
                      <Progress value={95} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Model Training Pipeline</span>
                        <span>89%</span>
                      </div>
                      <Progress value={89} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>AI Insights Generation</span>
                        <span>92%</span>
                      </div>
                      <Progress value={92} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Database Performance</span>
                        <span>98%</span>
                      </div>
                      <Progress value={98} className="h-2" />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Dashboard />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;