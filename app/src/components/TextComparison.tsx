import React, { useState } from 'react';
import { apiService } from '../services/api';
import { CompareResponse, CompareRequest } from '../types';

const TextComparison: React.FC = () => {
  const [textA, setTextA] = useState('');
  const [textB, setTextB] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const examplePairs = [
    {
      textA: "Healthcare is a fundamental right and we need universal coverage for all Americans",
      textB: "We should let the free market determine healthcare prices and reduce government involvement"
    },
    {
      textA: "We must secure our borders and enforce immigration laws",
      textB: "We need comprehensive immigration reform with a path to citizenship"
    },
    {
      textA: "Climate change requires immediate government action and regulation",
      textB: "Market-based solutions are better for addressing environmental concerns"
    }
  ];

  const handleCompare = async () => {
    if (!textA.trim() || !textB.trim()) {
      setError('Please enter text in both fields');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const request: CompareRequest = {
        text_a: textA.trim(),
        text_b: textB.trim(),
      };

      const response = await apiService.compareTexts(request);
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Comparison failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (example: { textA: string; textB: string }) => {
    setTextA(example.textA);
    setTextB(example.textB);
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

  const getSimilarityColor = (similarity: number) => {
    if (similarity > 0.8) return '#27ae60';
    if (similarity > 0.6) return '#f39c12';
    if (similarity > 0.4) return '#e67e22';
    return '#e74c3c';
  };

  return (
    <div className="text-comparison">
      <div className="card">
        <div className="card-header">
          <h2>Text Comparison</h2>
          <p>Compare two political texts for ideological similarity and differences</p>
        </div>
        <div className="card-body">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
            <div className="form-group">
              <label htmlFor="text-a">Text A</label>
              <textarea
                id="text-a"
                className="form-input form-textarea"
                value={textA}
                onChange={(e) => setTextA(e.target.value)}
                placeholder="Enter first political text..."
                maxLength={5000}
              />
              <small style={{ color: '#7f8c8d' }}>
                {textA.length}/5,000 characters
              </small>
            </div>

            <div className="form-group">
              <label htmlFor="text-b">Text B</label>
              <textarea
                id="text-b"
                className="form-input form-textarea"
                value={textB}
                onChange={(e) => setTextB(e.target.value)}
                placeholder="Enter second political text..."
                maxLength={5000}
              />
              <small style={{ color: '#7f8c8d' }}>
                {textB.length}/5,000 characters
              </small>
            </div>
          </div>

          <div className="example-texts">
            <p style={{ marginBottom: '0.5rem', fontWeight: 600, color: '#2c3e50' }}>
              Try these comparison examples:
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '1rem' }}>
              {examplePairs.map((example, index) => (
                <button
                  key={index}
                  className="btn btn-secondary"
                  style={{ fontSize: '0.8rem', padding: '0.5rem 1rem', textAlign: 'left' }}
                  onClick={() => handleExampleClick(example)}
                >
                  <strong>Example {index + 1}:</strong> {example.textA.substring(0, 50)}... vs {example.textB.substring(0, 50)}...
                </button>
              ))}
            </div>
          </div>

          <button
            className="btn"
            onClick={handleCompare}
            disabled={loading || !textA.trim() || !textB.trim()}
          >
            {loading ? 'Comparing...' : 'Compare Texts'}
          </button>

          {error && (
            <div className="error">
              <strong>Error:</strong> {error}
            </div>
          )}

          {loading && (
            <div className="loading">
              <div className="spinner"></div>
              <p>Comparing political texts...</p>
            </div>
          )}

          {result && (
            <div className="results">
              <div className="result-item">
                <h3>📊 Similarity Metrics</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
                  <div>
                    <h4>Cosine Similarity</h4>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                      <span
                        style={{
                          background: getSimilarityColor(result.cosine_similarity),
                          color: 'white',
                          padding: '0.5rem 1rem',
                          borderRadius: '8px',
                          fontWeight: 'bold',
                          fontSize: '1.1rem',
                        }}
                      >
                        {(result.cosine_similarity * 100).toFixed(1)}%
                      </span>
                      <div style={{ flex: 1 }}>
                        <div className="confidence-bar">
                          <div
                            className="confidence-fill"
                            style={{
                              width: `${result.cosine_similarity * 100}%`,
                              background: getSimilarityColor(result.cosine_similarity),
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4>Euclidean Distance</h4>
                    <span
                      style={{
                        background: '#95a5a6',
                        color: 'white',
                        padding: '0.5rem 1rem',
                        borderRadius: '8px',
                        fontWeight: 'bold',
                        fontSize: '1.1rem',
                      }}
                    >
                      {result.euclidean_distance.toFixed(3)}
                    </span>
                    <p style={{ fontSize: '0.9rem', color: '#7f8c8d', marginTop: '0.5rem' }}>
                      Lower values indicate higher similarity
                    </p>
                  </div>
                </div>

                <div style={{ marginTop: '1rem', textAlign: 'center' }}>
                  <h4>Interpretation:</h4>
                  <span
                    style={{
                      background: getSimilarityColor(result.cosine_similarity),
                      color: 'white',
                      padding: '0.5rem 1.5rem',
                      borderRadius: '20px',
                      fontWeight: 'bold',
                      fontSize: '1.1rem',
                    }}
                  >
                    {result.similarity_interpretation}
                  </span>
                </div>
              </div>

              <div className="result-item">
                <h3>🏛️ Ideological Analysis</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                  <div>
                    <h4 style={{ textAlign: 'center', marginBottom: '1rem', color: '#3498db' }}>Text A Ideology</h4>
                    <div className="ideology-grid">
                      <div className="ideology-item">
                        <h5>Political</h5>
                        <span
                          className={`ideology-label ${result.ideology_analysis.text_a_ideology.political}`}
                          style={{ backgroundColor: getIdeologyColor(result.ideology_analysis.text_a_ideology.political) }}
                        >
                          {result.ideology_analysis.text_a_ideology.political}
                        </span>
                      </div>
                      <div className="ideology-item">
                        <h5>Social</h5>
                        <span
                          className={`ideology-label ${result.ideology_analysis.text_a_ideology.social}`}
                          style={{ backgroundColor: getIdeologyColor(result.ideology_analysis.text_a_ideology.social) }}
                        >
                          {result.ideology_analysis.text_a_ideology.social}
                        </span>
                      </div>
                      <div className="ideology-item">
                        <h5>Economic</h5>
                        <span
                          className={`ideology-label ${result.ideology_analysis.text_a_ideology.economic}`}
                          style={{ backgroundColor: getIdeologyColor(result.ideology_analysis.text_a_ideology.economic) }}
                        >
                          {result.ideology_analysis.text_a_ideology.economic}
                        </span>
                      </div>
                      <div className="ideology-item">
                        <h5>Foreign</h5>
                        <span
                          className={`ideology-label ${result.ideology_analysis.text_a_ideology.foreign}`}
                          style={{ backgroundColor: getIdeologyColor(result.ideology_analysis.text_a_ideology.foreign) }}
                        >
                          {result.ideology_analysis.text_a_ideology.foreign}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 style={{ textAlign: 'center', marginBottom: '1rem', color: '#e74c3c' }}>Text B Ideology</h4>
                    <div className="ideology-grid">
                      <div className="ideology-item">
                        <h5>Political</h5>
                        <span
                          className={`ideology-label ${result.ideology_analysis.text_b_ideology.political}`}
                          style={{ backgroundColor: getIdeologyColor(result.ideology_analysis.text_b_ideology.political) }}
                        >
                          {result.ideology_analysis.text_b_ideology.political}
                        </span>
                      </div>
                      <div className="ideology-item">
                        <h5>Social</h5>
                        <span
                          className={`ideology-label ${result.ideology_analysis.text_b_ideology.social}`}
                          style={{ backgroundColor: getIdeologyColor(result.ideology_analysis.text_b_ideology.social) }}
                        >
                          {result.ideology_analysis.text_b_ideology.social}
                        </span>
                      </div>
                      <div className="ideology-item">
                        <h5>Economic</h5>
                        <span
                          className={`ideology-label ${result.ideology_analysis.text_b_ideology.economic}`}
                          style={{ backgroundColor: getIdeologyColor(result.ideology_analysis.text_b_ideology.economic) }}
                        >
                          {result.ideology_analysis.text_b_ideology.economic}
                        </span>
                      </div>
                      <div className="ideology-item">
                        <h5>Foreign</h5>
                        <span
                          className={`ideology-label ${result.ideology_analysis.text_b_ideology.foreign}`}
                          style={{ backgroundColor: getIdeologyColor(result.ideology_analysis.text_b_ideology.foreign) }}
                        >
                          {result.ideology_analysis.text_b_ideology.foreign}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="result-item">
                <h3>🤝 Ideological Agreement</h3>
                <div className="ideology-grid">
                  {Object.entries(result.ideology_analysis.ideology_agreement).map(([dimension, isAgreed]) => (
                    <div key={dimension} className="ideology-item">
                      <h4 style={{ textTransform: 'capitalize' }}>{dimension}</h4>
                      <span
                        style={{
                          background: isAgreed ? '#27ae60' : '#e74c3c',
                          color: 'white',
                          padding: '0.5rem 1rem',
                          borderRadius: '20px',
                          fontWeight: 'bold',
                        }}
                      >
                        {isAgreed ? '✓ Agree' : '✗ Disagree'}
                      </span>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: '1rem', textAlign: 'center' }}>
                  <p style={{ color: '#7f8c8d' }}>
                    Agreement score: {Object.values(result.ideology_analysis.ideology_agreement).filter(Boolean).length}/4 dimensions
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TextComparison; 