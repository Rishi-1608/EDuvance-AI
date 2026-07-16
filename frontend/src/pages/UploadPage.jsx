import { useState, useRef } from 'react';
import {
  Upload, Film, Image as ImageIcon, Mic, FileText, X,
  CheckCircle, AlertCircle, Zap, Plus, Play, Layers
} from 'lucide-react';
import { uploadVideo, uploadImage, uploadAudio, uploadDocument } from '@/lib/api';
import './UploadPage.css';

const ACCEPT = {
  video: '.mp4,.avi,.mov,.mkv,.wmv,.flv,.webm,.m4v',
  image: '.jpg,.jpeg,.png,.bmp,.tiff,.tif,.webp',
  audio: '.wav,.mp3,.m4a,.flac,.ogg,.aac,.wma',
  document: '.pdf,.doc,.docx,.ppt,.pptx,.txt',
};

const ALL_EXTENSIONS = Object.values(ACCEPT).join(',');

function getFileType(filename) {
  const ext = '.' + filename.split('.').pop().toLowerCase();
  for (const [type, exts] of Object.entries(ACCEPT)) {
    if (exts.includes(ext)) return type;
  }
  return null;
}

const TYPE_ICON = {
  video: Film,
  image: ImageIcon,
  audio: Mic,
  document: FileText,
};

function ModalFileRow({ file, onRemove }) {
  const type = getFileType(file.name);
  const size = (file.size / (1024 * 1024)).toFixed(1);
  const Icon = TYPE_ICON[type] || FileText;

  return (
    <div className="file-row animate-in">
      <div className="file-row-icon">
        <Icon size={16} />
      </div>
      <div className="file-info">
        <span className="file-name">{file.name}</span>
        <span className="file-meta">{size} MB &middot; {type ? type.toUpperCase() : 'FILE'}</span>
      </div>
      <button className="file-remove-btn" onClick={onRemove} title="Remove file">
        <X size={14} />
      </button>
    </div>
  );
}

export function UploadPage({ onSuccess, onStart, onUploadComplete }) {
  const [files, setFiles] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const inputRef = useRef();

  const handleFileChange = (e) => {
    const incoming = Array.from(e.target.files).filter(f => getFileType(f.name) !== null);
    if (incoming.length > 0) {
      setFiles(prev => [...prev, ...incoming].slice(0, 10));
      setIsModalOpen(true);
      setError(null);
      setSuccess(null);
    }
    // Reset input so re-selecting the same file works
    e.target.value = '';
  };

  const removeFile = (idx) => {
    setFiles(prev => {
      const next = prev.filter((_, i) => i !== idx);
      if (next.length === 0) setIsModalOpen(false);
      return next;
    });
  };

  const closeModal = () => {
    if (loading) return;
    setIsModalOpen(false);
    setFiles([]);
  };

  const handleStartProcessing = async () => {
    if (!files.length) return;
    setLoading(true);
    setError(null);

    try {
      const groups = { video: [], image: [], audio: [], document: [] };
      const presentTypes = [];
      files.forEach(f => {
        const type = getFileType(f.name);
        groups[type].push(f);
        if (!presentTypes.includes(type)) presentTypes.push(type);
      });

      // Immediate redirect trigger
      onStart?.(presentTypes);

      const fns = { video: uploadVideo, image: uploadImage, audio: uploadAudio, document: uploadDocument };

      const successMessages = [];
      for (const type of ['video', 'image', 'audio', 'document']) {
        if (groups[type].length > 0) {
          const res = await fns[type](groups[type]);
          successMessages.push(`${groups[type].length} ${type}(s)`);
          onSuccess?.(type, res);
        }
      }

      onUploadComplete?.();

      setSuccess(`Pipeline started for ${successMessages.join(', ')}`);
      setTimeout(() => {
        setIsModalOpen(false);
        setFiles([]);
        setLoading(false);
        setSuccess(null);
      }, 1800);
    } catch (e) {
      setError(e.message);
      setLoading(false);
      onUploadComplete?.();
    }
  };

  return (
    <div className="upload-page container">
      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ALL_EXTENSIONS}
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      {/* ═══ Hero Section ═══ */}
      <section className="home-hero">
        <div className="hero-eyebrow badge badge-accent">
          <Zap size={11} /> AI-POWERED ACADEMIC ENGINE
        </div>

        <h1 className="home-title">
          Your Lectures,{' '}
          <span className="gradient-text">Reimagined</span>
        </h1>

        <p className="home-sub">
          Transform videos, slides, PDFs, and audio into intelligent study notes,
          flashcards, and quizzes — powered by multi-modal AI.
        </p>

        <div className="home-cta-container">
          <button
            className="home-primary-btn"
            onClick={() => inputRef.current?.click()}
          >
            <div className="home-btn-icon">
              <Plus size={20} />
            </div>
            Add New Content
          </button>

          <span className="home-formats mono">
            VIDEO &bull; PDF &bull; IMAGES &bull; AUDIO
          </span>
        </div>
      </section>

      {/* ═══ Feature Pills ═══ */}
      <div className="home-features animate-up stagger-1">
        <div className="feature-pill">
          <div className="feature-icon"><Film size={20} /></div>
          <div className="feature-label">Video Analysis</div>
          <div className="feature-text">
            Frame-by-frame OCR &amp; Whisper transcription for deep lecture understanding.
          </div>
        </div>
        <div className="feature-pill">
          <div className="feature-icon"><ImageIcon size={20} /></div>
          <div className="feature-label">Slide OCR</div>
          <div className="feature-text">
            Extract text and concepts from presentation slides with precision.
          </div>
        </div>
        <div className="feature-pill">
          <div className="feature-icon"><Layers size={20} /></div>
          <div className="feature-label">Study Assets</div>
          <div className="feature-text">
            Auto-generate flashcards, MCQ quizzes, and formatted PDF notes.
          </div>
        </div>
      </div>

      {/* ═══ Upload Modal ═══ */}
      {isModalOpen && (
        <div
          className="modal-overlay"
          onClick={(e) => e.target === e.currentTarget && closeModal()}
        >
          <div className="upload-modal">
            {/* Header */}
            <div className="modal-header">
              <div className="modal-title">Prepare Content</div>
              <button
                className="modal-close"
                onClick={closeModal}
                disabled={loading}
              >
                <X size={18} />
              </button>
            </div>

            <p className="modal-desc">
              Review the files below, then hit <strong>Start</strong> to launch the AI pipeline.
            </p>

            {/* File List */}
            <div className="modal-file-list">
              {files.map((f, i) => (
                <ModalFileRow key={i} file={f} onRemove={() => removeFile(i)} />
              ))}
            </div>

            {/* Messages */}
            {error && (
              <div className="modal-msg modal-msg-error">
                <AlertCircle size={14} /> {error}
              </div>
            )}
            {success && (
              <div className="modal-msg modal-msg-success">
                <CheckCircle size={14} /> {success}
              </div>
            )}

            {/* Actions */}
            <div className="modal-footer">
              <button
                className="btn btn-primary modal-start-btn"
                onClick={handleStartProcessing}
                disabled={loading || files.length === 0}
              >
                {loading ? (
                  <><div className="spinner" style={{ width: 16, height: 16 }} /> Processing&hellip;</>
                ) : (
                  <><Play size={16} /> Start AI Analysis</>
                )}
              </button>

              {!loading && (
                <button
                  className="btn btn-ghost modal-add-btn"
                  onClick={() => inputRef.current?.click()}
                >
                  <Plus size={14} /> Add More Files
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
