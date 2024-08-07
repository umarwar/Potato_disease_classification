import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      setError(null);
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPrediction(response.data);
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="App">
      <h1>Potato Disease Classification</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit">Predict</button>
        <button type="button" onClick={handleClear}>Clear</button>
      </form>
      {loading && <p>Loading...</p>}
      {error && <p>{error}</p>}
      {prediction && (
        <div className="prediction">
          <h2>Prediction</h2>
          <p>Class: {prediction.class}</p>
          <p>Confidence: {prediction.confidence.toFixed(2) * 100}%</p>
        </div>
      )}
    </div>
  );
}

export default App;
