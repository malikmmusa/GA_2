/**
 * ResultsPanel Component
 * Displays progression analysis results and PDF download button
 */
import React from 'react';
import type { ImageAnalysis, ProgressionCalculationResponse } from '../types/api';

interface ResultsPanelProps {
  imageBefore: ImageAnalysis | null;
  imageAfter: ImageAnalysis | null;
  progression: ProgressionCalculationResponse | null;
  onDownloadPDF?: () => void;
}

export const ResultsPanel: React.FC<ResultsPanelProps> = ({
  imageBefore,
  imageAfter,
  progression,
  onDownloadPDF,
}) => {
  // Don't show results panel until both images are analyzed
  if (!imageBefore?.distance || !imageAfter?.distance || !progression) {
    return null;
  }

  const formatDate = (dateStr: string) => {
    const parsed = new Date(dateStr);
    if (Number.isNaN(parsed.getTime())) {
      return dateStr;
    }

    return parsed.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  // Format "YYYY-MM" predicted date as "Month YYYY"
  const formatPredictedMonth = (yearMonth: string): string => {
    const parsed = new Date(`${yearMonth}-01`);
    if (Number.isNaN(parsed.getTime())) return yearMonth;
    return parsed.toLocaleDateString('en-US', { year: 'numeric', month: 'long' });
  };

  return (
    <div className="card bg-blue-50 border-2 border-primary mt-8">
      <h2 className="text-2xl font-bold text-primary mb-6 text-center">
        RESULTS
      </h2>

      {/* Summary Grid */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* Before Image Data */}
        <div>
          <h3 className="font-semibold text-lg mb-2">Before Image</h3>
          <div className="space-y-1 text-sm">
            <p><span className="font-medium">Date:</span> {formatDate(imageBefore.date)}</p>
            <p><span className="font-medium">Eye:</span> {imageBefore.fovea?.eye_side}</p>
            <p><span className="font-medium">Distance:</span> {imageBefore.distance.distance_microns.toFixed(1)} µm</p>
          </div>
        </div>

        {/* After Image Data */}
        <div>
          <h3 className="font-semibold text-lg mb-2">After Image</h3>
          <div className="space-y-1 text-sm">
            <p><span className="font-medium">Date:</span> {formatDate(imageAfter.date)}</p>
            <p><span className="font-medium">Eye:</span> {imageAfter.fovea?.eye_side}</p>
            <p><span className="font-medium">Distance:</span> {imageAfter.distance.distance_microns.toFixed(1)} µm</p>
          </div>
        </div>
      </div>

      {/* Progression Analysis */}
      <div className="bg-white rounded-lg p-6 shadow-md">
        <h3 className="font-semibold text-lg mb-4">Progression Analysis</h3>

        {progression.status === 'error' ? (
          <div className="bg-red-50 border border-red-200 rounded p-4">
            <p className="text-red-800 font-semibold">⚠️ Error</p>
            <p className="text-red-700 text-sm mt-2">{progression.error_message}</p>
          </div>
        ) : progression.status === 'no_progression' ? (
          <div className="bg-yellow-50 border border-yellow-200 rounded p-4">
            <p className="text-yellow-800 font-semibold">No Progression Detected</p>
            <p className="text-yellow-700 text-sm mt-2">
              The distance to fovea has not changed between the two images.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Time Elapsed */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-gray-600">Time Elapsed</p>
                <p className="text-xl font-bold">{progression.days_elapsed} days</p>
              </div>
              <div>
                <p className="text-gray-600">Distance Change</p>
                <p className="text-xl font-bold text-orange-600">
                  {progression.distance_change_microns.toFixed(1)} µm
                </p>
              </div>
            </div>

            {/* Rate of Progression */}
            <div className="bg-orange-50 border border-orange-200 rounded p-4">
              <p className="text-gray-700 font-medium mb-2">Rate of Progression</p>
              <div className="flex items-baseline gap-4 flex-wrap">
                <p className="text-3xl font-bold text-orange-600">
                  {progression.rate_microns_per_year?.toFixed(1)} µm/year
                </p>
                <p className="text-base text-orange-400">
                  ({progression.rate_microns_per_month?.toFixed(1)} µm/month)
                </p>
              </div>
            </div>

            {/* Prediction */}
            {(progression.years_until_involvement != null || progression.predicted_foveal_involvement_date) && (
              <div className="bg-red-50 border-2 border-red-500 rounded-lg p-6 text-center">
                <p className="text-red-800 font-semibold text-lg mb-2">
                  ⚠️ PREDICTED FOVEAL INVOLVEMENT
                </p>
                {progression.years_until_involvement != null && (
                  <p className="text-3xl font-bold text-red-600">
                    ~{progression.years_until_involvement} years
                  </p>
                )}
                {progression.predicted_foveal_involvement_date && (
                  <p className="text-lg text-red-500 mt-1">
                    approx. {formatPredictedMonth(progression.predicted_foveal_involvement_date)}
                  </p>
                )}
                <p className="text-sm text-gray-600 mt-2">
                  Based on current rate of progression
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* PDF Download Button */}
      {onDownloadPDF && progression.status === 'progression' && (
        <div className="mt-6 text-center">
          <button
            onClick={onDownloadPDF}
            className="btn-primary px-8 py-3 text-lg"
          >
            📄 Download PDF Report
          </button>
        </div>
      )}
    </div>
  );
};
