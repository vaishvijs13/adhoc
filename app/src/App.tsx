import React, { useState } from 'react';
import './App.css';
import TextAnalyzer from './components/TextAnalyzer';
import TextComparison from './components/TextComparison';
import Visualization from './components/Visualization';
import About from './components/About';

function App() {
  const [activeTab, setActiveTab] = useState('analyze');

  const tabs = [
    { id: 'analyze', label: 'Analyze Text', component: TextAnalyzer },
    { id: 'compare', label: 'Compare Texts', component: TextComparison },
    { id: 'visualize', label: 'Visualize', component: Visualization },
    { id: 'about', label: 'About', component: About },
  ];

  const ActiveComponent = tabs.find(tab => tab.id === activeTab)?.component || TextAnalyzer;

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <h1>adhoc</h1>
          <p>Advanced ML-powered political text analysis and ideology classification</p>
        </div>
      </header>

      <nav className="App-nav">
        <div className="nav-tabs">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </nav>

      <main className="App-main">
        <div className="main-content">
          <ActiveComponent />
        </div>
      </main>

    </div>
  );
}

export default App;
