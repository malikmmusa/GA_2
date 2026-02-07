/**
 * ImageUpload Component
 * Handles file upload, date selection, and eye side selector
 */
import React, { useRef, useState } from 'react';

interface ImageUploadProps {
  title: string;
  onImageUpload: (file: File, date: string) => void;
  onDateChange?: (date: string) => void;
  currentDate?: string;
  eyeSide?: 'OD' | 'OS';
  onEyeSideChange?: (side: 'OD' | 'OS') => void;
  isProcessing?: boolean;
}

export const ImageUpload: React.FC<ImageUploadProps> = ({
  title,
  onImageUpload,
  onDateChange,
  currentDate,
  eyeSide,
  onEyeSideChange,
  isProcessing = false,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedDate, setSelectedDate] = useState<string>(
    currentDate || new Date().toISOString().split('T')[0]
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelection(files[0]);
    }
  };

  const handleFileSelection = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file');
      return;
    }

    setSelectedFile(file);
    onImageUpload(file, selectedDate);
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelection(files[0]);
    }
  };

  const handleDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newDate = e.target.value;
    setSelectedDate(newDate);
    // Propagate date change to parent to update ImageAnalysis state
    if (onDateChange) {
      onDateChange(newDate);
    }
  };

  return (
    <div className="card">
      <h2 className="text-xl font-bold mb-4">{title}</h2>

      {/* Upload Area */}
      <div
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          transition-colors
          ${isDragging ? 'border-primary bg-blue-50' : 'border-gray-300 hover:border-primary'}
          ${isProcessing ? 'opacity-50 pointer-events-none' : ''}
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileInputChange}
          disabled={isProcessing}
        />

        {selectedFile ? (
          <div>
            <p className="text-green-600 font-semibold">✓ {selectedFile.name}</p>
            <p className="text-sm text-gray-500 mt-2">Click or drag to replace</p>
          </div>
        ) : (
          <div>
            <p className="text-gray-600 font-semibold">Drop image here or click to upload</p>
            <p className="text-sm text-gray-500 mt-2">PNG, JPG up to 10MB</p>
          </div>
        )}
      </div>

      {/* Date Picker */}
      <div className="mt-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Date
        </label>
        <input
          type="date"
          value={selectedDate}
          onChange={handleDateChange}
          className="input-field"
          disabled={isProcessing}
        />
      </div>

      {/* Eye Side Selector */}
      {eyeSide && onEyeSideChange && (
        <div className="mt-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Eye Side {eyeSide && <span className="text-blue-600">(Auto: {eyeSide})</span>}
          </label>
          <select
            value={eyeSide}
            onChange={(e) => onEyeSideChange(e.target.value as 'OD' | 'OS')}
            className="input-field"
            disabled={isProcessing}
          >
            <option value="OD">OD (Right Eye)</option>
            <option value="OS">OS (Left Eye)</option>
          </select>
        </div>
      )}

      {isProcessing && (
        <div className="mt-4 text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          <p className="text-sm text-gray-600 mt-2">Processing...</p>
        </div>
      )}
    </div>
  );
};
