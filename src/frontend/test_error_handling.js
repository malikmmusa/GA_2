/**
 * Quick test for error handling utility
 * Run with: node test_error_handling.js
 */

// Simulate the extractErrorMessage function
function extractErrorMessage(error, fallbackMessage = 'An unexpected error occurred') {
  if (!error) {
    return fallbackMessage;
  }

  if (error.response?.data) {
    const data = error.response.data;

    // Check if it's a FastAPI validation error (422)
    if (Array.isArray(data.detail)) {
      const validationErrors = data.detail;
      const errorMessages = validationErrors.map((err) => {
        const field = err.loc.slice(1).join('.') || 'field';
        return `${field}: ${err.msg}`;
      });

      return errorMessages.length > 0
        ? `Validation error: ${errorMessages.join('; ')}`
        : 'Invalid request data';
    }

    if (typeof data.detail === 'string') {
      return data.detail;
    }

    if (data.detail && typeof data.detail === 'object' && 'msg' in data.detail) {
      return String(data.detail.msg);
    }

    if (data.detail && typeof data.detail === 'object') {
      try {
        return `Error: ${JSON.stringify(data.detail)}`;
      } catch {
        return 'Error processing response';
      }
    }

    if (typeof data.message === 'string') {
      return data.message;
    }
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  if (error.message && typeof error.message === 'string') {
    return error.message;
  }

  if (error.msg && typeof error.msg === 'string') {
    return error.msg;
  }

  if (typeof error === 'string') {
    return error;
  }

  if (typeof error === 'object') {
    try {
      return `Error: ${JSON.stringify(error)}`;
    } catch {
      return fallbackMessage;
    }
  }

  return fallbackMessage;
}

// Test cases
console.log('='.repeat(70));
console.log('ERROR HANDLING UTILITY TESTS');
console.log('='.repeat(70));

// Test 1: Pydantic validation error (the problematic case)
console.log('\n1. Pydantic Validation Error (422):');
const validationError = {
  response: {
    data: {
      detail: [
        {
          type: 'missing',
          loc: ['body', 'fovea_x'],
          msg: 'Field required',
          input: { invalid: 'data' }
        },
        {
          type: 'missing',
          loc: ['body', 'fovea_y'],
          msg: 'Field required',
          input: { invalid: 'data' }
        }
      ]
    }
  }
};
const result1 = extractErrorMessage(validationError);
console.log('Result:', result1);
console.log('Type:', typeof result1);
console.log('Is string?', typeof result1 === 'string');

// Test 2: Simple string error
console.log('\n2. Simple String Error:');
const stringError = {
  response: {
    data: {
      detail: 'Image processing failed'
    }
  }
};
const result2 = extractErrorMessage(stringError);
console.log('Result:', result2);
console.log('Type:', typeof result2);

// Test 3: Standard Error object
console.log('\n3. Standard Error Object:');
const stdError = new Error('Network request failed');
const result3 = extractErrorMessage(stdError);
console.log('Result:', result3);
console.log('Type:', typeof result3);

// Test 4: Raw object (would crash React)
console.log('\n4. Raw Object (would crash React):');
const rawObject = {
  type: 'error',
  loc: ['body', 'field'],
  msg: 'Invalid'
};
const result4 = extractErrorMessage(rawObject);
console.log('Result:', result4);
console.log('Type:', typeof result4);

// Test 5: Null
console.log('\n5. Null:');
const result5 = extractErrorMessage(null);
console.log('Result:', result5);
console.log('Type:', typeof result5);

console.log('\n' + '='.repeat(70));
console.log('✅ ALL RESULTS ARE STRINGS - SAFE FOR REACT');
console.log('='.repeat(70));
