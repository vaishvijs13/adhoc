import React, { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import { VisualizationResponse } from '../types';

const Visualization: React.FC = () => {
  const [texts, setTexts] = useState<string[]>(['']);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<VisualizationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [method, setMethod] = useState('umap');
  const [colorBy, setColorBy] = useState('ideology');
  const [includeReferences, setIncludeReferences] = useState(true);
  const [analysisType, setAnalysisType] = useState<'ideology' | 'cluster' | 'dimensional'>('ideology');

  const exampleTexts = [
    "Healthcare is a fundamental right requiring universal coverage",
    "Free market solutions work best for healthcare reform",
    "We need stronger border security and immigration enforcement",
    "Comprehensive immigration reform with pathway to citizenship is needed",
    "Climate change requires immediate government intervention",
    "Market-based environmental solutions are more effective"
  ];

  const addTextInput = () => {
    setTexts([...texts, '']);
  };

  const removeTextInput = (index: number) => {
    if (texts.length > 1) {
      setTexts(texts.filter((_, i) => i !== index));
    }
  };

  const updateText = (index: number, value: string) => {
    const newTexts = [...texts];
    newTexts[index] = value;
    setTexts(newTexts);
  };

  const loadExamples = () => {
    setTexts(exampleTexts);
  };

  const handleVisualize = async () => {
    const validTexts = texts.filter(text => text.trim().length > 0);
    
    if (validTexts.length < 2) {
      setError('Please enter at least 2 texts to visualize');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let response: VisualizationResponse;
      
      switch (analysisType) {
        case 'ideology':
          response = await apiService.createIdeologyVisualization({
            texts: validTexts,
            method,
            color_by: colorBy,
            include_references: includeReferences,
          });
          break;
        case 'cluster':
          response = await apiService.createClusterAnalysis({
            texts: validTexts,
            n_clusters: Math.min(4, Math.max(2, Math.ceil(validTexts.length / 2))),
            method,
          });
          break;
        case 'dimensional':
          response = await apiService.createDimensionalAnalysis({
            texts: validTexts,
            method,
            color_by: colorBy,
            include_references: includeReferences,
          });
          break;
        default:
          throw new Error('Invalid analysis type');
      }
      
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Visualization failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="visualization">
      <div className="card">
        <div className="card-header">
          <h2>📊 Political Text Visualization</h2>
          <p>Visualize political texts in ideological space with advanced analytics</p>
        </div>
        <div className="card-body">
          <div className="form-group">
            <label>Analysis Type</label>
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <input
                  type="radio"
                  name="analysisType"
                  value="ideology"
                  checked={analysisType === 'ideology'}
                  onChange={(e) => setAnalysisType(e.target.value as any)}
                />
                Ideology Space
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <input
                  type="radio"
                  name="analysisType"
                  value="cluster"
                  checked={analysisType === 'cluster'}
                  onChange={(e) => setAnalysisType(e.target.value as any)}
                />
                Cluster Analysis
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <input
                  type="radio"
                  name="analysisType"
                  value="dimensional"
                  checked={analysisType === 'dimensional'}
                  onChange={(e) => setAnalysisType(e.target.value as any)}
                />
                Dimensional Analysis
              </label>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1rem' }}>
            <div className="form-group">
              <label htmlFor="method">Reduction Method</label>
              <select
                id="method"
                className="form-input"
                value={method}
                onChange={(e) => setMethod(e.target.value)}
              >
                <option value="umap">UMAP</option>
                <option value="tsne">t-SNE</option>
                <option value="pca">PCA</option>
              </select>
            </div>

            {analysisType !== 'cluster' && (
              <div className="form-group">
                <label htmlFor="colorBy">Color By</label>
                <select
                  id="colorBy"
                  className="form-input"
                  value={colorBy}
                  onChange={(e) => setColorBy(e.target.value)}
                >
                  <option value="ideology">Ideology</option>
                  <option value="similarity">Similarity</option>
                  <option value="confidence">Confidence</option>
                </select>
              </div>
            )}

            <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <input
                type="checkbox"
                id="includeReferences"
                checked={includeReferences}
                onChange={(e) => setIncludeReferences(e.target.checked)}
              />
              <label htmlFor="includeReferences">Include Reference Politicians</label>
            </div>
          </div>

          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <label>Political Texts to Analyze</label>
              <div>
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ marginRight: '0.5rem', fontSize: '0.8rem', padding: '0.5rem 1rem' }}
                  onClick={loadExamples}
                >
                  Load Examples
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ fontSize: '0.8rem', padding: '0.5rem 1rem' }}
                  onClick={addTextInput}
                >
                  + Add Text
                </button>
              </div>
            </div>
            
            {texts.map((text, index) => (
              <div key={index} style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <textarea
                  className="form-input form-textarea"
                  style={{ minHeight: '80px' }}
                  value={text}
                  onChange={(e) => updateText(index, e.target.value)}
                  placeholder={`Enter political text ${index + 1}...`}
                  maxLength={2000}
                />
                {texts.length > 1 && (
                  <button
                    type="button"
                    className="btn btn-secondary"
                    style={{ 
                      padding: '0.5rem',
                      minWidth: '40px',
                      height: '40px',
                      alignSelf: 'flex-start',
                      marginTop: '0.5rem'
                    }}
                    onClick={() => removeTextInput(index)}
                  >
                    ✕
                  </button>
                )}
              </div>
            ))}
            <small style={{ color: '#7f8c8d' }}>
              {texts.filter(t => t.trim()).length} texts entered
            </small>
          </div>

          <button
            className="btn"
            onClick={handleVisualize}
            disabled={loading || texts.filter(t => t.trim()).length < 2}
          >
            {loading ? 'Generating Visualization...' : 'Create Visualization'}
          </button>

          {error && (
            <div className="error">
              <strong>Error:</strong> {error}
            </div>
          )}

          {loading && (
            <div className="loading">
              <div className="spinner"></div>
              <p>Creating visualization...</p>
            </div>
          )}

          {result && (
            <div className="results">
              <div className="result-item">
                <h3>📈 Visualization Results</h3>
                
                {result.plot_html && (
                  <div style={{ marginBottom: '2rem' }}>
                    <h4>Interactive Plot</h4>
                    <div 
                      style={{ 
                        border: '1px solid #e0e0e0',
                        borderRadius: '8px',
                        overflow: 'hidden',
                        height: '500px'
                      }}
                      dangerouslySetInnerHTML={{ __html: result.plot_html }}
                    />
                  </div>
                )}

                <div className="ideology-grid">
                  <div className="ideology-item">
                    <h4>Analysis Summary</h4>
                    <div style={{ textAlign: 'left' }}>
                      {result.analysis.total_points && (
                        <p><strong>Total Points:</strong> {result.analysis.total_points}</p>
                      )}
                      {result.analysis.input_points && (
                        <p><strong>Input Texts:</strong> {result.analysis.input_points}</p>
                      )}
                      {result.analysis.reference_points && (
                        <p><strong>Reference Politicians:</strong> {result.analysis.reference_points}</p>
                      )}
                    </div>
                  </div>

                  {result.analysis.ideology_distribution && (
                    <div className="ideology-item">
                      <h4>Ideology Distribution</h4>
                      <div style={{ textAlign: 'left' }}>
                        <p><strong>Left:</strong> {result.analysis.ideology_distribution.left}%</p>
                        <p><strong>Center:</strong> {result.analysis.ideology_distribution.center}%</p>
                        <p><strong>Right:</strong> {result.analysis.ideology_distribution.right}%</p>
                      </div>
                    </div>
                  )}

                  {result.clusters && (
                    <div className="ideology-item">
                      <h4>Cluster Information</h4>
                      <p><strong>Number of Clusters:</strong> {Math.max(...result.clusters) + 1}</p>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem', marginTop: '0.5rem' }}>
                        {result.clusters.map((cluster, index) => (
                          <span
                            key={index}
                            style={{
                              background: `hsl(${cluster * 60}, 70%, 50%)`,
                              color: 'white',
                              padding: '0.2rem 0.5rem',
                              borderRadius: '10px',
                              fontSize: '0.8rem',
                            }}
                          >
                            T{index + 1}: C{cluster}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {result.plot_data && (
                  <div style={{ marginTop: '2rem' }}>
                    <h4>Data Points</h4>
                    <div style={{ 
                      maxHeight: '300px', 
                      overflowY: 'auto',
                      border: '1px solid #e0e0e0',
                      borderRadius: '8px',
                      padding: '1rem'
                    }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                          <tr style={{ borderBottom: '2px solid #e0e0e0' }}>
                            <th style={{ padding: '0.5rem', textAlign: 'left' }}>Text</th>
                            <th style={{ padding: '0.5rem', textAlign: 'center' }}>X</th>
                            <th style={{ padding: '0.5rem', textAlign: 'center' }}>Y</th>
                            {result.plot_data.politicians && <th style={{ padding: '0.5rem', textAlign: 'left' }}>Politician</th>}
                            {result.plot_data.types && <th style={{ padding: '0.5rem', textAlign: 'left' }}>Type</th>}
                          </tr>
                        </thead>
                        <tbody>
                          {result.plot_data.texts.map((text, index) => (
                            <tr key={index} style={{ borderBottom: '1px solid #f0f0f0' }}>
                              <td style={{ padding: '0.5rem', maxWidth: '300px' }}>
                                {text.length > 50 ? `${text.substring(0, 50)}...` : text}
                              </td>
                              <td style={{ padding: '0.5rem', textAlign: 'center', fontFamily: 'monospace' }}>
                                {result.plot_data.x[index]?.toFixed(2)}
                              </td>
                              <td style={{ padding: '0.5rem', textAlign: 'center', fontFamily: 'monospace' }}>
                                {result.plot_data.y[index]?.toFixed(2)}
                              </td>
                              {result.plot_data.politicians && (
                                <td style={{ padding: '0.5rem' }}>
                                  {result.plot_data.politicians[index] || 'User Input'}
                                </td>
                              )}
                              {result.plot_data.types && (
                                <td style={{ padding: '0.5rem' }}>
                                  <span
                                    style={{
                                      background: result.plot_data.types[index] === 'input' ? '#3498db' : '#95a5a6',
                                      color: 'white',
                                      padding: '0.2rem 0.5rem',
                                      borderRadius: '10px',
                                      fontSize: '0.8rem',
                                    }}
                                  >
                                    {result.plot_data.types[index]}
                                  </span>
                                </td>
                              )}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Visualization; 