import { useState } from 'react';
import { uploadImages } from '../services/api';
import './ImageUpload.css';

/**
 * ImageUpload — file-selector + upload UI for reference and target images.
 *
 * States: idle → uploading → success | error
 * On success the returned metadata for both images is displayed.
 */
export default function ImageUpload() {
  const [referenceFile, setReferenceFile] = useState(null);
  const [targetFile, setTargetFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const canUpload = referenceFile && targetFile && !uploading;

  const handleUpload = async () => {
    setUploading(true);
    setError(null);
    setResult(null);
    try {
      const data = await uploadImages(referenceFile, targetFile);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleReset = () => {
    setReferenceFile(null);
    setTargetFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <section className="upload-card">
      <h2>Image Upload</h2>
      <p className="upload-hint">
        Select a reference and target lunar image to begin.
      </p>

      {/* File selectors */}
      <div className="file-inputs">
        <label className="file-label" htmlFor="ref-input">
          <span className="file-label-title">Reference Image</span>
          <span className="file-label-name">
            {referenceFile ? referenceFile.name : 'No file selected'}
          </span>
          <input
            id="ref-input"
            type="file"
            accept=".png,.jpg,.jpeg,.tif,.tiff"
            onChange={(e) => setReferenceFile(e.target.files[0] || null)}
          />
        </label>

        <label className="file-label" htmlFor="tgt-input">
          <span className="file-label-title">Target Image</span>
          <span className="file-label-name">
            {targetFile ? targetFile.name : 'No file selected'}
          </span>
          <input
            id="tgt-input"
            type="file"
            accept=".png,.jpg,.jpeg,.tif,.tiff"
            onChange={(e) => setTargetFile(e.target.files[0] || null)}
          />
        </label>
      </div>

      {/* Actions */}
      <div className="upload-actions">
        <button
          className="btn btn-primary"
          onClick={handleUpload}
          disabled={!canUpload}
        >
          {uploading ? 'Uploading…' : 'Upload'}
        </button>

        {(result || error) && (
          <button className="btn btn-secondary" onClick={handleReset}>
            Reset
          </button>
        )}
      </div>

      {/* Error state */}
      {error && (
        <div className="upload-error">
          <strong>Upload failed:</strong> {error}
        </div>
      )}

      {/* Success state — metadata display */}
      {result && (
        <div className="upload-result">
          <h3>Upload Successful</h3>
          <div className="meta-grid">
            <MetaCard title="Reference" meta={result.reference} />
            <MetaCard title="Target" meta={result.target} />
          </div>
        </div>
      )}
    </section>
  );
}

/** Renders metadata for a single image. */
function MetaCard({ title, meta }) {
  return (
    <div className="meta-card">
      <h4>{title}</h4>
      <table>
        <tbody>
          <tr><td>Original file</td><td>{meta.original_filename}</td></tr>
          <tr><td>Stored as</td><td className="mono">{meta.stored_filename}</td></tr>
          <tr><td>Size</td><td>{formatBytes(meta.size_bytes)}</td></tr>
          <tr><td>Type</td><td>{meta.content_type}</td></tr>
          <tr><td>Extension</td><td>{meta.extension}</td></tr>
          <tr>
            <td>Dimensions</td>
            <td>
              {meta.width && meta.height
                ? `${meta.width} × ${meta.height}`
                : <span className="dim-pending">pending preprocessing</span>}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}
