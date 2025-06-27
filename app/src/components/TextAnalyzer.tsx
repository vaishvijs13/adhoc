import React, { useState } from 'react';
import { apiService } from '../services/api';
import { AnalyzeResponse, AnalyzeRequest } from '../types';

const TextAnalyzer: React.FC = () => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [includeNeighbors, setIncludeNeighbors] = useState(true);
  const [topK, setTopK] = useState(5);

  const exampleTexts = [
    "Healthcare is a fundamental right and we need universal coverage for all Americans",
    "We must secure our borders and enforce immigration laws to protect American workers",
    "Climate change is an existential threat requiring immediate government action",
    "Lower taxes and deregulation will stimulate economic growth and job creation",
    "We need criminal justice reform and police accountability to ensure equality"
  ];

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const request: AnalyzeRequest = {
        text: text.trim(),
        include_neighbors: includeNeighbors,
        top_k: topK,
        return_attention: true,
      };

      const response = await apiService.analyzeText(request);
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (exampleText: string) => {
    setText(exampleText);
  };

  const getIdeologyColor = (ideology: string) => {
    switch (ideology.toLowerCase()) {
      case 'left':
      case 'liberal':
        return '#3498db';
      case 'center':
      case 'moderate':
      case 'neutral':
        return '#95a5a6';
      case 'right':
      case 'conservative':
        return '#e74c3c';
      case 'isolationist':
        return '#f39c12';
      case 'interventionist':
        return '#9b59b6';
      default:
        return '#95a5a6';
    }
  };

  return (
    <div className="text-analyzer">
      <div className="card">
        <div className="card-header">
          <h2>🧠 Political Text Analysis</h2>
          <p>Analyze political text for ideology classification and insights</p>
        </div>
        <div className="card-body">
          <div className="form-group">
            <label htmlFor="text-input">Enter Political Text</label>
            <textarea
              id="text-input"
              className="form-input form-textarea"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter political text to analyze (e.g., policy statements, speeches, social media posts)..."
              maxLength={10000}
            />
            <small style={{ color: '#7f8c8d' }}>
              {text.length}/10,000 characters
            </small>
          </div>

          <div className="example-texts">
            <p style={{ marginBottom: '0.5rem', fontWeight: 600, color: '#2c3e50' }}>
              📝 Try these examples:
            </p>
            <div className="example-buttons" style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1rem' }}>
              {exampleTexts.map((example, index) => (
                <button
                  key={index}
                  className="btn btn-secondary"
                  style={{ fontSize: '0.8rem', padding: '0.5rem 1rem' }}
                  onClick={() => handleExampleClick(example)}
                >
                  Example {index + 1}
                </button>
              ))}
            </div>
          </div>

          <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <input
                type="checkbox"
                checked={includeNeighbors}
                onChange={(e) => setIncludeNeighbors(e.target.checked)}
              />
              Find similar politicians
            </label>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <label htmlFor="top-k">Similar results:</label>
              <select
                id="top-k"
                className="form-input"
                style={{ width: 'auto', padding: '0.5rem' }}
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
              >
                <option value={3}>3</option>
                <option value={5}>5</option>
                <option value={10}>10</option>
              </select>
            </div>
          </div>

          <button
            className="btn"
            onClick={handleAnalyze}
            disabled={loading || !text.trim()}
          >
            {loading ? 'Analyzing...' : 'Analyze Text'}
          </button>

          {error && (
            <div className="error">
              <strong>Error:</strong> {error}
            </div>
          )}

          {loading && (
            <div className="loading">
              <div className="spinner"></div>
              <p>Analyzing political ideology...</p>
            </div>
          )}

          {result && (
            <div className="results">
              <div className="result-item">
                <h3>🎯 Primary Classification</h3>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
                  <span
                    className={`ideology-label ${result.predicted_label}`}
                    style={{ backgroundColor: getIdeologyColor(result.predicted_label) }}
                  >
                    {result.predicted_label.toUpperCase()}
                  </span>
                  <div style={{ flex: 1 }}>
                    <div className="confidence-bar">
                      <div
                        className="confidence-fill"
                        style={{ width: `${result.label_confidence * 100}%` }}
                      ></div>
                    </div>
                    <small style={{ color: '#7f8c8d' }}>
                      {(result.label_confidence * 100).toFixed(1)}% confidence
                    </small>
                  </div>
                </div>
              </div>

              <div className="result-item">
                <h3>📊 Multi-Dimensional Analysis</h3>
                <div className="ideology-grid">
                  <div className="ideology-item">
                    <h4>Social Issues</h4>
                    <span
                      className={`ideology-label ${result.multi_label_ideology.social}`}
                      style={{ backgroundColor: getIdeologyColor(result.multi_label_ideology.social) }}
                    >
                      {result.multi_label_ideology.social}
                    </span>
                    <div className="confidence-bar">
                      <div
                        className="confidence-fill"
                        style={{ width: `${result.multi_label_ideology.social_confidence * 100}%` }}
                      ></div>
                    </div>
                    <small>{(result.multi_label_ideology.social_confidence * 100).toFixed(1)}%</small>
                  </div>

                  <div className="ideology-item">
                    <h4>Economic Policy</h4>
                    <span
                      className={`ideology-label ${result.multi_label_ideology.economic}`}
                      style={{ backgroundColor: getIdeologyColor(result.multi_label_ideology.economic) }}
                    >
                      {result.multi_label_ideology.economic}
                    </span>
                    <div className="confidence-bar">
                      <div
                        className="confidence-fill"
                        style={{ width: `${result.multi_label_ideology.economic_confidence * 100}%` }}
                      ></div>
                    </div>
                    <small>{(result.multi_label_ideology.economic_confidence * 100).toFixed(1)}%</small>
                  </div>

                  <div className="ideology-item">
                    <h4>Foreign Policy</h4>
                    <span
                      className={`ideology-label ${result.multi_label_ideology.foreign}`}
                      style={{ backgroundColor: getIdeologyColor(result.multi_label_ideology.foreign) }}
                    >
                      {result.multi_label_ideology.foreign}
                    </span>
                    <div className="confidence-bar">
                      <div
                        className="confidence-fill"
                        style={{ width: `${result.multi_label_ideology.foreign_confidence * 100}%` }}
                      ></div>
                    </div>
                    <small>{(result.multi_label_ideology.foreign_confidence * 100).toFixed(1)}%</small>
                  </div>
                </div>
              </div>

              {result.important_tokens && result.important_tokens.length > 0 && (
                <div className="result-item">
                  <h3>🔍 Key Terms</h3>
                  <p style={{ marginBottom: '1rem', color: '#7f8c8d' }}>
                    Most influential words for this classification:
                  </p>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {result.important_tokens.map((token, index) => (
                      <span
                        key={index}
                        style={{
                          background: `rgba(52, 152, 219, ${0.2 + token.importance_score * 0.6})`,
                          color: '#2c3e50',
                          padding: '0.25rem 0.75rem',
                          borderRadius: '15px',
                          fontSize: '0.9rem',
                          fontWeight: 600,
                        }}
                      >
                        {token.token}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {result.neighbors && result.neighbors.length > 0 && (
                <div className="result-item">
                  <h3>👥 Similar Politicians</h3>
                  <p style={{ marginBottom: '1rem', color: '#7f8c8d' }}>
                    Politicians with similar ideological statements:
                  </p>
                  {result.neighbors.map((neighbor, index) => (
                    <div
                      key={index}
                      style={{
                        background: 'white',
                        border: '1px solid #e0e0e0',
                        borderRadius: '8px',
                        padding: '1rem',
                        marginBottom: '0.5rem',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                        <strong style={{ color: '#2c3e50' }}>{neighbor.politician}</strong>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          {neighbor.political_lean && (
                            <span
                              className={`ideology-label ${neighbor.political_lean}`}
                              style={{
                                backgroundColor: getIdeologyColor(neighbor.political_lean),
                                fontSize: '0.8rem',
                                padding: '0.2rem 0.5rem',
                              }}
                            >
                              {neighbor.political_lean}
                            </span>
                          )}
                          <span
                            style={{
                              background: '#27ae60',
                              color: 'white',
                              padding: '0.2rem 0.5rem',
                              borderRadius: '10px',
                              fontSize: '0.8rem',
                              fontWeight: 600,
                            }}
                          >
                            {(neighbor.similarity_score * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                      <p style={{ color: '#7f8c8d', fontSize: '0.9rem', fontStyle: 'italic' }}>
                        "{neighbor.sample_text}"
                      </p>
                      {neighbor.date && (
                        <small style={{ color: '#95a5a6' }}>Date: {neighbor.date}</small>
                      )}
                    </div>
                  ))}
                </div>
              )}

              <div className="result-item">
                <h3>📈 Processing Info</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                  <div>
                    <strong>Original length:</strong> {result.processing_info.original_length} chars
                  </div>
                  <div>
                    <strong>Processed length:</strong> {result.processing_info.cleaned_length} chars
                  </div>
                  <div>
                    <strong>Model:</strong> {result.processing_info.model_used}
                  </div>
                  <div>
                    <strong>Embedding dimensions:</strong> {result.processing_info.embedding_dim}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TextAnalyzer; 