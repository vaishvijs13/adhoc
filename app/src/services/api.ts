import axios from 'axios';
import {
  AnalyzeRequest,
  AnalyzeResponse,
  CompareRequest,
  CompareResponse,
  VisualizationRequest,
  VisualizationResponse,
} from '../types';

const API_BASE_URL = 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} from ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const apiService = {
  // Health check
  async healthCheck() {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  // Analyze text
  async analyzeText(payload: AnalyzeRequest): Promise<AnalyzeResponse> {
    try {
      const response = await api.post('/analyze/', payload);
      return response.data;
    } catch (error) {
      console.error('Text analysis failed:', error);
      throw error;
    }
  },

  // Compare texts
  async compareTexts(payload: CompareRequest): Promise<CompareResponse> {
    try {
      const response = await api.post('/compare/', payload);
      return response.data;
    } catch (error) {
      console.error('Text comparison failed:', error);
      throw error;
    }
  },

  // Visualizations
  async createIdeologyVisualization(payload: VisualizationRequest): Promise<VisualizationResponse> {
    try {
      const response = await api.post('/visualize/ideology-space', payload);
      return response.data;
    } catch (error) {
      console.error('Ideology visualization failed:', error);
      throw error;
    }
  },

  async createClusterAnalysis(payload: { texts: string[]; n_clusters?: number; method?: string }): Promise<VisualizationResponse> {
    try {
      const response = await api.post('/visualize/cluster-analysis', payload);
      return response.data;
    } catch (error) {
      console.error('Cluster analysis failed:', error);
      throw error;
    }
  },

  async createDimensionalAnalysis(payload: VisualizationRequest): Promise<VisualizationResponse> {
    try {
      const response = await api.post('/visualize/dimensional-analysis', payload);
      return response.data;
    } catch (error) {
      console.error('Dimensional analysis failed:', error);
      throw error;
    }
  },

  // Get reference data
  async getReferenceData() {
    try {
      const response = await api.get('/visualize/reference-data');
      return response.data;
    } catch (error) {
      console.error('Failed to get reference data:', error);
      throw error;
    }
  },

  // Database stats
  async getDatabaseStats() {
    try {
      const response = await api.get('/update/stats');
      return response.data;
    } catch (error) {
      console.error('Failed to get database stats:', error);
      throw error;
    }
  },
};

export default apiService; 