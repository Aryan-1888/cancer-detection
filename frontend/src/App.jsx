import { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
  };

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);

    try {
      const res = await axios.post(
        "https://cancer-detection-4-g11i.onrender.com/predict",
        formData
      );
      setResult(res.data);
    } catch (err) {
      alert("Backend not running!");
    }

    setLoading(false);
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>🩺 Skin Cancer Detection</h1>

      <input type="file" onChange={handleChange} />

      {preview && (
        <img src={preview} alt="preview" width="300" />
      )}

      <br /><br />

      <button onClick={handleUpload}>Predict</button>

      {loading && <p>Analyzing...</p>}

      {result && (
        <div>
          <h2>{result.result}</h2>
          <p>
            Confidence: {(result.confidence * 100).toFixed(2)}%
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
