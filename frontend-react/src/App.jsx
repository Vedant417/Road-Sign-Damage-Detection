import React, { useState, useRef, useEffect } from 'react';
import {
  Video, Map, Download, Activity, AlertTriangle, ShieldCheck, Clock, Disc, Target, Info, Crosshair
} from 'lucide-react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const API_URL = 'http://localhost:8000/api/detect';

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (selectedFile) => {
    if (!selectedFile.type.startsWith('image/')) {
      alert('Please upload a valid image file.');
      return;
    }
    setFile(selectedFile);
    setError('');

    const reader = new FileReader();
    reader.onload = (e) => {
      setImageSrc(e.target.result);
      uploadAndDetect(selectedFile);
    };
    reader.readAsDataURL(selectedFile);
  };

  const uploadAndDetect = async (fileToUpload) => {
    setLoading(true);
    setResults(null);
    setError('');

    const formData = new FormData();
    formData.append('file', fileToUpload);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error(`HTTP Error ${response.status}`);

      const data = await response.json();

      // Multi-stage filtering complete
      if (data.status === 'Success') {
        setResults(data);
      } else if (data.status === 'NO_SIGN_DETECTED') {
        // High-confidence rejection triggered
        setResults({ detections: [], global_severity_score: 0 });
        setError('');
      } else {
        setError(data.error || 'Server processing failed.');
      }
    } catch (err) {
      console.error(err);
      setError('Connection refused. Ensure Node.js API is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (results && imageSrc && canvasRef.current) {
      const image = new Image();
      image.onload = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        canvas.width = image.width;
        canvas.height = image.height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0);

        if (results.detections && results.detections.length > 0) {
          results.detections.forEach((det) => {
            const [x, y, w, h] = det.bounding_box;

            let color = '#00ff9d'; // LOW
            if (det.severity_level === 'MEDIUM') color = '#ffcc00';
            else if (det.severity_level === 'HIGH') color = '#ff3a3a';

            if (det.detection_flag !== 'NORMAL') color = '#00ccff';

            ctx.lineWidth = Math.max(3, canvas.width / 400);
            ctx.strokeStyle = color;
            ctx.strokeRect(x, y, w, h);

            ctx.fillStyle = color;
            const fontSize = Math.max(16, canvas.width / 60);
            ctx.font = `600 ${fontSize}px "JetBrains Mono", monospace`;

            const label = det.sign_type.replace(/_/g, ' ');
            const textWidth = ctx.measureText(label).width;

            ctx.fillRect(x, y - fontSize - 8, textWidth + 12, fontSize + 8);
            ctx.fillStyle = '#000000';
            ctx.fillText(label, x + 6, y - 6);
          });
        }
      };
      image.src = imageSrc;
    }
  }, [results, imageSrc]);

  let totalDetections = 0;
  let highestSeverityLevel = 'LOW';
  let severityScore = 0;

  if (results) {
    totalDetections = results.detections?.length || 0;
    severityScore = results.global_severity_score;
    if (severityScore > 70) highestSeverityLevel = 'HIGH';
    else if (severityScore > 30) highestSeverityLevel = 'MEDIUM';
    else highestSeverityLevel = 'LOW';
  }

  return (
    <div className="dashboard-root">

      <nav className="top-nav">
        <div className="brand">
          <div className="brand-icon">
            <Map size={24} />
          </div>
          <div className="brand-text">
            <h1>DeepSign <span>Vision</span></h1>
            <p>AI-Powered Infrastructure Monitoring</p>
          </div>
        </div>

        <div className="system-status">
          <div className="status-dot"></div>
          SYSTEM ONLINE
        </div>
      </nav>

      <main className="dashboard-main">

        <div className="col-left">

          <div className="camera-view">
            <div className="camera-header">
              <div className="live-badge">
                <div className="dot"></div> LIVE
              </div>
              <div className="cam-meta mono">
                4K | 60 FPS
              </div>
            </div>

            {loading ? (
              <div className="awaiting-feed">
                <div className="camera-icon-ring fast">
                  <Activity size={48} strokeWidth={2} />
                </div>
                <p className="green">PROCESSING DEEP TELEMETRY...</p>
              </div>
            ) : !imageSrc ? (
              <div
                className="awaiting-feed"
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="camera-icon-ring">
                  <Video size={48} strokeWidth={1} />
                </div>
                <p>AWAITING FEED SOURCE</p>
                <button className="upload-btn">
                  <Download size={14} strokeWidth={2.5} /> UPLOAD SENSOR DATA
                </button>
                <input type="file" ref={fileInputRef} hidden onChange={handleFileChange} />
              </div>
            ) : (
              <div className="feed-content">
                <input type="file" ref={fileInputRef} hidden onChange={handleFileChange} />
                <canvas ref={canvasRef}></canvas>
                <button className="upload-btn overlay-btn" onClick={() => fileInputRef.current?.click()}>
                  <Download size={14} strokeWidth={2.5} /> UPLOAD NEXT IMAGE
                </button>
              </div>
            )}
          </div>

          <div className="metrics-row">
            <div className="metric-box">
              <h4>Total Targets</h4>
              <div className="metric-value">
                <div className="num mono">{totalDetections}</div>
              </div>
              <Activity className="metric-icon" size={120} />
            </div>

            <div className="metric-box">
              <h4>Max Degradation</h4>
              <div className="metric-value">
                <div className={`num mono ${highestSeverityLevel === 'HIGH' ? 'red' : highestSeverityLevel === 'MEDIUM' ? 'yellow' : 'green'}`}>{severityScore}</div>
                {results && totalDetections > 0 && <div className={`sub-badge ${highestSeverityLevel === 'HIGH' ? 'red' : highestSeverityLevel === 'MEDIUM' ? 'yellow' : 'green'}`}>{highestSeverityLevel}</div>}
              </div>
              <AlertTriangle className="metric-icon" size={120} />
            </div>

            <div className="metric-box">
              <h4>Network Scan</h4>
              <div className="metric-value">
                <div className="num mono">42.8</div>
                <div className="sub-label">Miles<br />Scanned</div>
              </div>
              <Disc className="metric-icon" size={120} />
            </div>
          </div>

        </div>

        <div className="col-right">
          <div className="stream-header">
            <div className="stream-title">
              <Activity size={18} className="green" /> Inference Stream
            </div>
            <div className="stream-live">
              <div className="dot"></div> DETERMINISTIC
            </div>
          </div>

          <div className="stream-list">
            {results && results.detections && results.detections.length > 0 ? (
              results.detections.map((det, i) => {

                const lvl = det.severity_level.toLowerCase();
                let Icon = ShieldCheck;
                if (lvl === 'high') Icon = AlertTriangle;
                if (lvl === 'medium') Icon = Info;

                const isAmbiguous = det.detection_flag === 'AMBIGUOUS';
                const isUncertain = det.detection_flag === 'UNCERTAIN';
                if (isAmbiguous || isUncertain) Icon = Crosshair;

                return (
                  <div key={i} className={`event-card ${lvl}`} style={{ borderColor: (isAmbiguous || isUncertain) ? 'var(--neon-blue)' : '' }}>
                    <div className={`event-icon ${lvl}`} style={{ backgroundColor: (isAmbiguous || isUncertain) ? 'rgba(0, 204, 255, 0.1)' : '', color: (isAmbiguous || isUncertain) ? 'var(--neon-blue)' : '' }}>
                      <Icon size={20} strokeWidth={2.5} />
                    </div>

                    <div className="event-content">
                      <div className="event-top">
                        <span className="event-name" style={{ color: (isAmbiguous || isUncertain) ? 'var(--neon-blue)' : '#fff' }}>
                          {det.sign_type.replace(/_/g, ' ')}
                        </span>

                        <div style={{ display: 'flex', gap: '0.4rem' }}>
                          {(isAmbiguous || isUncertain) && (
                            <span className="event-severity" style={{ background: 'var(--neon-blue)', color: '#000' }}>
                              {det.detection_flag}
                            </span>
                          )}
                          <span className={`event-severity ${lvl}`}>
                            {det.severity_level} ({det.severity_score})
                          </span>
                        </div>
                      </div>

                      <div className="analysis-grid">
                        <div className="analysis-item">
                          <span>Blur Variance</span>
                          <span className="val mono">{det.analysis.blur_score}</span>
                        </div>
                        <div className="analysis-item">
                          <span>Color Qual</span>
                          <span className="val mono">{det.analysis.color_score}</span>
                        </div>
                        <div className="analysis-item">
                          <span>Struct Int.</span>
                          <span className="val mono">{det.analysis.edge_integrity}</span>
                        </div>
                        <div className="analysis-item">
                          <span>Occlusion</span>
                          <span className="val mono">{det.analysis.occlusion_ratio}</span>
                        </div>
                      </div>

                      <div className="event-description">
                        <span className="damage-badge">{det.damage_type}</span>
                        <p>{det.explanation}</p>
                      </div>

                      <div className="event-bottom">
                        <div className="event-coords">
                          <Target size={12} />
                          [ x: {det.bounding_box[0]}, y: {det.bounding_box[1]} ]
                        </div>
                        <div className="event-conf mono" style={{ color: (isAmbiguous || isUncertain) ? 'var(--neon-blue)' : '' }}>
                          {(det.confidence * 100).toFixed(1)}% CONF
                        </div>
                      </div>
                    </div>
                  </div>
                )
              })
            ) : results && results.detections && results.detections.length === 0 ? (
              <div style={{ color: 'var(--text-dim)', textAlign: 'center', marginTop: '3rem', fontSize: '0.85rem' }}>
                <ShieldCheck size={48} style={{ opacity: 0.2, margin: '0 auto 1rem', display: 'block' }} />
                <p>NO VALID TRAFFIC SIGNS IDENTIFIED</p>
                <p style={{ opacity: 0.5, marginTop: '0.5rem', fontSize: '0.75rem' }}>Background noise dynamically suppressed</p>
              </div>
            ) : error ? (
              <p style={{ color: 'var(--neon-red)', textAlign: 'center', marginTop: '3rem', fontSize: '0.85rem' }}>
                {error}
              </p>
            ) : (
              <p style={{ color: 'var(--text-dim)', textAlign: 'center', marginTop: '3rem', fontSize: '0.85rem' }}>
                System standby. Waiting for telemetry data...
              </p>
            )}
          </div>
        </div>
      </main>

    </div>
  );
}

export default App;
