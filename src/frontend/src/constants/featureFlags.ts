/**
 * Flags for optional parts of the measurement workflow.
 */

/**
 * Automatic GA detection: the full-image clustering pass (`/segment-ga`) and the
 * click-to-segment-locally fallback (`/segment-ga-local`).
 *
 * Disabled, so the GA edge is always placed by hand and the distance is measured
 * from that point. The backend endpoints and `ga_segmenter` are untouched and
 * still used by the validation scripts — set this back to `true` to restore the
 * automatic flow in the UI.
 */
export const AUTO_GA_DETECTION_ENABLED = false;
