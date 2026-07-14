import { useState } from 'react';
import { Book, Play, Clock, ChevronRight, FileText, Layers, HelpCircle, Search, Filter, Trash2, Image as ImageIcon, Mic } from 'lucide-react';
import { useStatus } from '@/hooks/useStatus';
import { deleteLecture } from '@/lib/api';
import { ConfirmModal } from '@/components/ConfirmModal';
import { shortStem } from '@/lib/utils';
import './LibraryPage.css';

export function LibraryPage({ onSelectVideo }) {
  const getMediaConfig = (type) => {
    switch (type) {
      case 'video':
        return { icon: <Play size={20} />, btnIcon: <Play size={10} />, label: 'Video', btnText: 'Play Video', color: 'video' };
      case 'images':
        return { icon: <ImageIcon size={20} />, btnIcon: <ImageIcon size={10} />, label: 'Images', btnText: 'View Photos', color: 'images' };
      case 'audio':
        return { icon: <Mic size={20} />, btnIcon: <Mic size={10} />, label: 'Audio', btnText: 'Listen Audio', color: 'audio' };
      case 'document':
      case 'pdf':
        return { icon: <FileText size={20} />, btnIcon: <FileText size={10} />, label: 'Document', btnText: 'Read Doc', color: 'document' };
      default:
        return { icon: <Play size={20} />, btnIcon: <Play size={10} />, label: 'Media', btnText: 'View', color: 'video' };
    }
  };

  const { status, loading, error, refetch } = useStatus(5000);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all'); // all, processed, pending

  const [deleteModal, setDeleteModal] = useState({ isOpen: false, stem: null });

  const confirmDelete = async () => {
    if (!deleteModal.stem) return;
    try {
      await deleteLecture(deleteModal.stem);
      setDeleteModal({ isOpen: false, stem: null });
      refetch();
    } catch (err) {
      alert("Failed to delete: " + err.message);
    }
  };

  const handleDeleteClick = (e, stem) => {
    e.stopPropagation();
    setDeleteModal({ isOpen: true, stem });
  };

  const videos = status?.videos || {};
  const videoEntries = Object.entries(videos)
    .map(([key, data]) => ({
      key,
      stem: data.video_stem || key.replace(/\.[^/.]+$/, "") || key,
      ...data
    }))
    .filter(v => {
      const matchesSearch = v.lecture_title?.toLowerCase().includes(search.toLowerCase()) ||
        v.stem?.toLowerCase().includes(search.toLowerCase());
      if (!matchesSearch) return false;

      if (filter === 'processed') return v.study_notes_ready;
      if (filter === 'pending') return !v.study_notes_ready;
      return true;
    })
    .sort((a, b) => (b.created_at || 0) - (a.created_at || 0));

  if (loading && !status) {
    return (
      <div className="library-page">
        <div className="loading-state">
          <div className="spinner" />
          <p>Loading your library...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="library-page">
      <div className="library-header animate-up">
        <div />

        <div className="library-controls">
          <div className="search-box glass">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search lectures..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>

          <div className="filter-tabs glass">
            <button
              className={`filter-tab ${filter === 'all' ? 'active' : ''}`}
              onClick={() => setFilter('all')}
            >
              All
            </button>
            <button
              className={`filter-tab ${filter === 'processed' ? 'active' : ''}`}
              onClick={() => setFilter('processed')}
            >
              Completed
            </button>
            <button
              className={`filter-tab ${filter === 'pending' ? 'active' : ''}`}
              onClick={() => setFilter('pending')}
            >
              Processing
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="status-msg error animate-up">
          Error loading library: {error}
        </div>
      )}

      <div className="library-grid-container">
        <div className="library-grid">
          {videoEntries.length > 0 ? (
            videoEntries.map((video, idx) => (
              <div
                key={video.key}
                className="library-card glass animate-up"
                style={{ '--delay': `${idx * 0.05}s` }}
                onClick={() => onSelectVideo(video.stem)}
              >
                <div className="library-card-header">
                  <div className={`library-card-icon type-${getMediaConfig(video.input_type).color}`}>
                    {getMediaConfig(video.input_type).icon}
                  </div>
                  <div className="library-card-badges">
                    <span className={`media-type-badge ${getMediaConfig(video.input_type).color}`}>
                      {getMediaConfig(video.input_type).label}
                    </span>
                    {video.study_notes_ready ? (
                      <span className="status-tag success">Ready</span>
                    ) : (
                      <span className="status-tag warning">Processing</span>
                    )}
                  </div>
                </div>

                <div className="library-card-content">
                  <h3 className="library-card-title">
                    {video.lecture_title || video.display_name || video.stem}
                  </h3>
                  <p className="library-card-meta">
                    {video.subject_area || 'General'} • {video.difficulty || 'Medium'}
                  </p>
                </div>

                <div className="library-card-footer">
                  <div className="library-card-actions">
                    <button className="btn btn-primary btn-xs">
                      {getMediaConfig(video.input_type).btnIcon}
                      <span style={{ marginLeft: '4px' }}>{getMediaConfig(video.input_type).btnText}</span>
                    </button>
                    <button
                      className="btn btn-ghost btn-xs delete-btn"
                      onClick={(e) => handleDeleteClick(e, video.stem)}
                      title="Delete permanently"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                  <ChevronRight size={18} className="arrow-icon" />
                </div>
              </div>
            ))
          ) : (
            <div className="empty-state glass animate-up">
              <Book size={48} />
              <h3>No Lectures Found</h3>
              <p>You haven't uploaded any videos yet, or none match your search.</p>
            </div>
          )}
        </div>
      </div>

      <ConfirmModal
        isOpen={deleteModal.isOpen}
        onClose={() => setDeleteModal({ isOpen: false, stem: null })}
        onConfirm={confirmDelete}
        title="Delete Resource?"
        message={`This will permanently remove "${deleteModal.stem || 'this lecture'}" and all its generated notes, flashcards, and quizzes from local storage. Do you really want to delete?`}
      />
    </div>
  );
}
