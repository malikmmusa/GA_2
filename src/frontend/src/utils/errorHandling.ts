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
  input?: unknown;
}

interface FastAPIErrorPayload {
  detail?: unknown;
  message?: unknown;
}

interface ErrorWithResponse {
  response?: {
    data?: FastAPIErrorPayload;
  };
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
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
  error: unknown,
  fallbackMessage: string = 'An unexpected error occurred'
): string {
  // Handle null/undefined
  if (!error) {
    return fallbackMessage;
  }

  // Try axios error response structure first
  const responseData = (error as ErrorWithResponse)?.response?.data;
  if (responseData && isObject(responseData)) {
    const data = responseData;

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
    if (isObject(data.detail) && 'msg' in data.detail) {
      return String(data.detail.msg);
    }

    // Fallback: stringify the detail object
    if (isObject(data.detail)) {
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

  if (isObject(error)) {
    if (typeof error.message === 'string') {
      return error.message;
    }
    if (typeof error.msg === 'string') {
      return error.msg;
    }
  }

  // If error is a string itself
  if (typeof error === 'string') {
    return error;
  }

  // Last resort: try to stringify
  if (isObject(error)) {
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
export function ensureStringError(error: unknown): string {
  return extractErrorMessage(error);
}
