import { useState, useRef, useEffect } from 'react';
import './ImageUploader.css';

/**
 * Reusable ImageUploader component for reference and target images.
 * Includes Stage B image preview so operators can visually confirm uploads.
 *
 * @param {object} props
 * @param {string} props.id - Unique DOM element identifier.
 * @param {string} props.label - Display label (e.g. "Reference Image").
 * @param {string} [props.subtitle] - Helper text (e.g. "Base lunar mosaic").
 * @param {File | null} props.selectedFile - Current file instance.
 * @param {(file: File | null) => void} props.onFileSelect - Callback when file changes.
 * @param {boolean} [props.disabled=false] - Whether inputs are disabled.
 */
export default function ImageUploader({
  id = 'image-uploader',
  label = 'Select Image',
  subtitle = 'Drag & drop or click to browse',
  selectedFile = null,
  onFileSelect,
  disabled = false,
}) {
  const [dragOver, setDragOver] = useState(false);
  const [validationError, setValidationError] = useState('');
  const [prevFile, setPrevFile] = useState(selectedFile);
  const [previewUrl, setPreviewUrl] = useState(() => (selectedFile ? URL.createObjectURL(selectedFile) : null));
  const [imageDims, setImageDims] = useState(null);
  const fileInputRef = useRef(null);

  // Synchronize preview URL when selectedFile prop changes
  if (prevFile !== selectedFile) {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPrevFile(selectedFile);
    setPreviewUrl(selectedFile ? URL.createObjectURL(selectedFile) : null);
    setImageDims(null);
  }

  // Cleanup object URL on unmount
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const formatFileSize = (bytes) => {
    if (!bytes || bytes <= 0) return '0 B';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const validateAndSetFile = (file) => {
    setValidationError('');
    if (!file) {
      onFileSelect(null);
      return;
    }

    // MIME and extension validation
    const validMimes = ['image/png', 'image/jpeg', 'image/tiff', 'image/bmp', 'image/webp'];
    const isImageMime = file.type && (file.type.startsWith('image/') || validMimes.includes(file.type));
    const hasImageExtension = /\.(png|jpe?g|tif|tiff|bmp|webp)$/i.test(file.name);

    if (!isImageMime && !hasImageExtension) {
      setValidationError('Invalid file type. Please select an image file (PNG, JPG, TIFF).');
      return;
    }

    onFileSelect(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setDragOver(true);
    }
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    if (disabled) return;

    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleRemove = (e) => {
    e.stopPropagation();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    setValidationError('');
    setImageDims(null);
    onFileSelect(null);
  };

  const triggerBrowse = () => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handlePreviewLoad = (e) => {
    if (e.target.naturalWidth && e.target.naturalHeight) {
      setImageDims({
        width: e.target.naturalWidth,
        height: e.target.naturalHeight,
      });
    }
  };

  return (
    <div
      className={`uploader-card ${dragOver ? 'drag-over' : ''} ${
        selectedFile ? 'has-file' : ''
      } ${disabled ? 'disabled' : ''}`}
    >
      <div className="uploader-header">
        <span className="uploader-label">{label}</span>
        {selectedFile && (
          <span className="uploader-status-badge">Loaded & Verified</span>
        )}
      </div>

      <div
        className="uploader-dropzone"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={triggerBrowse}
        role="button"
        tabIndex={disabled ? -1 : 0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            triggerBrowse();
          }
        }}
        aria-label={`${label} upload area`}
      >
        <input
          ref={fileInputRef}
          id={id}
          type="file"
          accept="image/*,.tif,.tiff"
          onChange={handleInputChange}
          disabled={disabled}
          className="uploader-native-input"
        />

        {selectedFile ? (
          <div className="uploader-preview-container">
            {/* Thumbnail Preview Area */}
            <div className="uploader-thumbnail-wrapper">
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt={`${label} preview thumbnail`}
                  className="uploader-thumbnail-img"
                  onLoad={handlePreviewLoad}
                />
              ) : (
                <div className="uploader-thumbnail-placeholder">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
                    <circle cx="9" cy="9" r="2" />
                    <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
                  </svg>
                </div>
              )}
              <div className="uploader-thumbnail-hover-hint">
                <span>Click to change</span>
              </div>
            </div>

            {/* File Details */}
            <div className="uploader-file-details">
              <p className="uploader-filename" title={selectedFile.name}>
                {selectedFile.name}
              </p>
              <div className="uploader-meta-tags">
                <span className="uploader-tag">{formatFileSize(selectedFile.size)}</span>
                {imageDims && (
                  <span className="uploader-tag dims">
                    {imageDims.width} × {imageDims.height} px
                  </span>
                )}
                <span className="uploader-tag">{selectedFile.type || 'image/raw'}</span>
              </div>
            </div>

            {/* Remove Button */}
            <button
              type="button"
              className="uploader-remove-btn"
              onClick={handleRemove}
              disabled={disabled}
              title="Remove file"
              aria-label="Remove selected file"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        ) : (
          <div className="uploader-placeholder">
            <div className="uploader-cloud-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <p className="uploader-hint-title">{subtitle}</p>
            <p className="uploader-hint-sub">Supports PNG, JPG, TIFF · Max 50 MB</p>
          </div>
        )}
      </div>

      {validationError && (
        <div className="uploader-validation-error" role="alert">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span>{validationError}</span>
        </div>
      )}
    </div>
  );
}
