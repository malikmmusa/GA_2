/**
 * Error Handling Utilities
 * Safely extracts error messages from API responses
 */

/**
 * Pydantic validation error structure (FastAPI 422 responses)
 */
interface ValidationError {
  type: string;
  loc: Array<string | number>;
  msg: string;
  input?: any;
}

/**
 * Safely extract a human-readable error message from any error object
 * 
 * Handles:
 * - Axios errors with response data
 * - Pydantic validation errors (422)
 * - Standard Error objects
 * - Raw objects
 * - Null/undefined
 * 
 * @param error - The error object to extract message from
 * @param fallbackMessage - Default message if extraction fails
 * @returns A safe string error message that can be rendered in React
 */
export function extractErrorMessage(
  error: any,
  fallbackMessage: string = 'An unexpected error occurred'
): string {
  // Handle null/undefined
  if (!error) {
    return fallbackMessage;
  }

  // Try axios error response structure first
  if (error.response?.data) {
    const data = error.response.data;

    // Check if it's a FastAPI validation error (422)
    if (Array.isArray(data.detail)) {
      // Pydantic validation errors - format them nicely
      const validationErrors = data.detail as ValidationError[];
      const errorMessages = validationErrors.map((err) => {
        const field = err.loc.slice(1).join('.') || 'field'; // Skip first element (usually "body")
        return `${field}: ${err.msg}`;
      });

      return errorMessages.length > 0
        ? `Validation error: ${errorMessages.join('; ')}`
        : 'Invalid request data';
    }

    // Check if detail is a string
    if (typeof data.detail === 'string') {
      return data.detail;
    }

    // Check if detail is an object with msg field
    if (data.detail && typeof data.detail === 'object' && 'msg' in data.detail) {
      return String(data.detail.msg);
    }

    // Fallback: stringify the detail object
    if (data.detail && typeof data.detail === 'object') {
      try {
        return `Error: ${JSON.stringify(data.detail)}`;
      } catch {
        return 'Error processing response';
      }
    }

    // Check for message field
    if (typeof data.message === 'string') {
      return data.message;
    }
  }

  // Try standard Error object
  if (error instanceof Error && error.message) {
    return error.message;
  }

  // Try error.message property
  if (error.message && typeof error.message === 'string') {
    return error.message;
  }

  // Try error.msg property (some APIs use this)
  if (error.msg && typeof error.msg === 'string') {
    return error.msg;
  }

  // If error is a string itself
  if (typeof error === 'string') {
    return error;
  }

  // Last resort: try to stringify
  if (typeof error === 'object') {
    try {
      return `Error: ${JSON.stringify(error)}`;
    } catch {
      return fallbackMessage;
    }
  }

  // Absolute fallback
  return fallbackMessage;
}

/**
 * Type guard to ensure error is a string
 * Use this in components to satisfy TypeScript
 */
export function ensureStringError(error: any): string {
  return extractErrorMessage(error);
}
