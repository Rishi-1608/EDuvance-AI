import React from 'react';
import { AlertTriangle, X } from 'lucide-react';
import './ConfirmModal.css';

export function ConfirmModal({ isOpen, onClose, onConfirm, title, message, confirmText = "Delete Permanently", onProcess, processText = "Process" }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="confirm-modal glass animate-scale-in" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div className="warning-icon-box">
            <AlertTriangle size={24} />
          </div>
          <button className="close-btn" onClick={onClose}>
            <X size={20} />
          </button>
        </div>
         
        <div className="modal-content">
          <h3 className="modal-title">{title || 'Confirm Action'}</h3>
          <p className="modal-message">{message || 'Are you sure you want to proceed?'}</p>
        </div>
        
        <div className="modal-actions">
          {onProcess ? (
             <>
               <button className="btn btn-danger" onClick={onConfirm}>{confirmText}</button>
               <button className="btn btn-primary" onClick={onProcess}>{processText}</button>
             </>
          ) : (
             <>
               <button className="btn btn-ghost" onClick={onClose}>Cancel</button>
               <button className="btn btn-danger" onClick={onConfirm}>{confirmText}</button>
             </>
          )}
        </div>
      </div>
    </div>
  );
}
