import React, { useState, useEffect } from 'react';
import { apiService } from '../services/api';

const About: React.FC = () => {
  const [healthData, setHealthData] = useState<any>(null);
  const [dbStats, setDbStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        const [health, stats] = await Promise.all([
          apiService.healthCheck(),
          apiService.getDatabaseStats(),
        ]);
        setHealthData(health);
        setDbStats(stats);
      } catch (error) {
        console.error('Failed to fetch system info:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchSystemInfo();
  }, []);

  const features = [
    {
      icon: '🧠',
      title: 'Advanced ML Analysis',
      description: 'Uses state-of-the-art transformer models (BERT-based) for political text classification with multi-dimensional ideology scoring.',
    },
    {
      icon: '⚖️',
      title: 'Text Comparison',
      description: 'Compare two political texts using cosine similarity and euclidean distance metrics with detailed ideological breakdown.',
    },
    {
      icon: '📊',
      title: 'Interactive Visualizations',
      description: 'Visualize political texts in ideological space using UMAP, t-SNE, or PCA dimensionality reduction techniques.',
    },
    {
      icon: '👥',
      title: 'Politician Matching',
      description: 'Find politicians with similar ideological statements using FAISS-powered vector similarity search.',
    },
    {
      icon: '🎯',
      title: 'Multi-Label Classification',
      description: 'Analyze texts across multiple dimensions: social issues, economic policy, and foreign policy.',
    },
    {
      icon: '🔍',
      title: 'Token Importance',
      description: 'Identify the most influential words and phrases that drive the political classification.',
    },
  ];

  const techStack = [
    { category: 'Backend', technologies: ['FastAPI', 'PyTorch', 'Transformers', 'FAISS', 'UMAP', 'Scikit-learn'] },
    { category: 'Frontend', technologies: ['React', 'TypeScript', 'Axios', 'CSS3'] },
    { category: 'ML Models', technologies: ['BERT', 'Political BERT', 'Custom Classifiers'] },
    { category: 'Data', technologies: ['Vector Embeddings', 'CSV Storage', 'NLTK'] },
  ];

  return (
    <div className="about">
      <div className="card">
        <div className="card-header">
          <h2>ℹ️ About Adhoc Political Analysis</h2>
          <p>Advanced machine learning platform for political text analysis and ideology classification</p>
        </div>
        <div className="card-body">
          <div className="result-item">
            <h3>🚀 What is Adhoc?</h3>
            <p style={{ lineHeight: 1.6, color: '#5a6c7d' }}>
              Adhoc is a sophisticated political analysis platform that uses cutting-edge machine learning 
              techniques to analyze political text and classify ideological positions. Built with FastAPI 
              and PyTorch, it provides real-time analysis of political statements, speeches, and documents 
              across multiple ideological dimensions.
            </p>
          </div>

          <div className="result-item">
            <h3>✨ Key Features</h3>
            <div className="ideology-grid">
              {features.map((feature, index) => (
                <div key={index} className="ideology-item" style={{ textAlign: 'left' }}>
                  <h4>
                    <span style={{ fontSize: '1.5rem', marginRight: '0.5rem' }}>{feature.icon}</span>
                    {feature.title}
                  </h4>
                  <p style={{ color: '#7f8c8d', fontSize: '0.9rem', lineHeight: 1.4 }}>
                    {feature.description}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div className="result-item">
            <h3>🛠️ Technology Stack</h3>
            <div className="ideology-grid">
              {techStack.map((stack, index) => (
                <div key={index} className="ideology-item" style={{ textAlign: 'left' }}>
                  <h4>{stack.category}</h4>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem' }}>
                    {stack.technologies.map((tech, techIndex) => (
                      <span
                        key={techIndex}
                        style={{
                          background: '#3498db',
                          color: 'white',
                          padding: '0.2rem 0.5rem',
                          borderRadius: '10px',
                          fontSize: '0.8rem',
                          fontWeight: 600,
                        }}
                      >
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="result-item">
            <h3>📈 System Status</h3>
            {loading ? (
              <div className="loading">
                <div className="spinner"></div>
                <p>Loading system information...</p>
              </div>
            ) : (
              <div className="ideology-grid">
                <div className="ideology-item">
                  <h4>🏥 Health Status</h4>
                  <div style={{ textAlign: 'left' }}>
                    <p>
                      <strong>Status:</strong>{' '}
                      <span
                        style={{
                          background: healthData?.status === 'healthy' ? '#27ae60' : '#e74c3c',
                          color: 'white',
                          padding: '0.2rem 0.5rem',
                          borderRadius: '10px',
                          fontSize: '0.8rem',
                          fontWeight: 600,
                        }}
                      >
                        {healthData?.status || 'Unknown'}
                      </span>
                    </p>
                    <p>
                      <strong>Version:</strong> {healthData?.version || 'N/A'}
                    </p>
                    <p>
                      <strong>Models Loaded:</strong>{' '}
                      <span style={{ color: healthData?.models_loaded ? '#27ae60' : '#e74c3c' }}>
                        {healthData?.models_loaded ? '✓ Yes' : '✗ No'}
                      </span>
                    </p>
                  </div>
                </div>

                <div className="ideology-item">
                  <h4>💾 Database Stats</h4>
                  <div style={{ textAlign: 'left' }}>
                    <p><strong>Total Embeddings:</strong> {dbStats?.total_embeddings || 0}</p>
                    <p><strong>Unique Politicians:</strong> {dbStats?.unique_politicians || 0}</p>
                    <p><strong>Database Size:</strong> {dbStats?.database_size || 'N/A'}</p>
                    <p><strong>Last Updated:</strong> {dbStats?.last_updated || 'N/A'}</p>
                  </div>
                </div>

                <div className="ideology-item">
                  <h4>🔧 API Endpoints</h4>
                  <div style={{ textAlign: 'left' }}>
                    <p><strong>Base URL:</strong> http://localhost:8001</p>
                    <p><strong>Documentation:</strong> <a href="http://localhost:8001/docs" target="_blank" rel="noopener noreferrer">/docs</a></p>
                    <p><strong>Health Check:</strong> <a href="http://localhost:8001/health" target="_blank" rel="noopener noreferrer">/health</a></p>
                    <p><strong>Analysis:</strong> /analyze/</p>
                  </div>
                </div>

                <div className="ideology-item">
                  <h4>🎯 Model Info</h4>
                  <div style={{ textAlign: 'left' }}>
                    <p><strong>Primary Model:</strong> Political BERT</p>
                    <p><strong>Embedding Dim:</strong> 256D</p>
                    <p><strong>Vector Search:</strong> FAISS</p>
                    <p><strong>Visualizations:</strong> UMAP, t-SNE, PCA</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="result-item">
            <h3>🔬 How It Works</h3>
            <div style={{ textAlign: 'left', lineHeight: 1.6, color: '#5a6c7d' }}>
              <ol style={{ paddingLeft: '1.5rem' }}>
                <li style={{ marginBottom: '0.5rem' }}>
                  <strong>Text Preprocessing:</strong> Input text is cleaned, tokenized, and prepared for analysis using NLTK.
                </li>
                <li style={{ marginBottom: '0.5rem' }}>
                  <strong>Embedding Generation:</strong> The text is processed through a fine-tuned BERT model to create 256-dimensional embeddings.
                </li>
                <li style={{ marginBottom: '0.5rem' }}>
                  <strong>Classification:</strong> Multiple classifiers analyze the embeddings to determine ideology across social, economic, and foreign policy dimensions.
                </li>
                <li style={{ marginBottom: '0.5rem' }}>
                  <strong>Similarity Search:</strong> FAISS performs efficient vector similarity search against a database of politician statements.
                </li>
                <li style={{ marginBottom: '0.5rem' }}>
                  <strong>Visualization:</strong> Dimensionality reduction techniques project high-dimensional embeddings into 2D space for visualization.
                </li>
              </ol>
            </div>
          </div>

          <div className="result-item">
            <h3>📚 Use Cases</h3>
            <div className="ideology-grid">
              <div className="ideology-item" style={{ textAlign: 'left' }}>
                <h4>📰 Media Analysis</h4>
                <p style={{ color: '#7f8c8d', fontSize: '0.9rem' }}>
                  Analyze news articles, editorials, and opinion pieces for political bias and ideological positioning.
                </p>
              </div>
              <div className="ideology-item" style={{ textAlign: 'left' }}>
                <h4>🗳️ Campaign Research</h4>
                <p style={{ color: '#7f8c8d', fontSize: '0.9rem' }}>
                  Compare candidate positions, track ideological shifts, and analyze campaign messaging.
                </p>
              </div>
              <div className="ideology-item" style={{ textAlign: 'left' }}>
                <h4>🎓 Academic Research</h4>
                <p style={{ color: '#7f8c8d', fontSize: '0.9rem' }}>
                  Study political discourse, ideology evolution, and comparative politics across different contexts.
                </p>
              </div>
              <div className="ideology-item" style={{ textAlign: 'left' }}>
                <h4>📱 Social Media</h4>
                <p style={{ color: '#7f8c8d', fontSize: '0.9rem' }}>
                  Analyze political posts, comments, and discussions on social media platforms.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About; 