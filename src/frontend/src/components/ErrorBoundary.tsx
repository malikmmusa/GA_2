/**
 * ErrorBoundary Component
 * Catches React errors and displays them instead of white screen
 */
import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({
      error,
      errorInfo,
    });
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
          <div className="bg-white border-2 border-red-500 rounded-lg shadow-xl p-8 max-w-3xl w-full">
            <div className="flex items-start mb-4">
              <div className="flex-shrink-0">
                <svg
                  className="h-12 w-12 text-red-500"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
              </div>
              <div className="ml-4 flex-1">
                <h1 className="text-2xl font-bold text-red-600 mb-2">
                  Application Error
                </h1>
                <p className="text-gray-700 mb-4">
                  The application encountered an unexpected error. This has been logged for debugging.
                </p>

                {/* Error Details */}
                <div className="bg-red-50 border border-red-200 rounded p-4 mb-4">
                  <p className="font-mono text-sm text-red-800 font-semibold mb-2">
                    {this.state.error?.name}: {this.state.error?.message}
                  </p>
                  {this.state.error?.stack && (
                    <details className="mt-2">
                      <summary className="cursor-pointer text-sm text-red-700 hover:text-red-900">
                        Show stack trace
                      </summary>
                      <pre className="mt-2 text-xs overflow-x-auto bg-white p-2 rounded border border-red-300">
                        {this.state.error.stack}
                      </pre>
                    </details>
                  )}
                  {this.state.errorInfo?.componentStack && (
                    <details className="mt-2">
                      <summary className="cursor-pointer text-sm text-red-700 hover:text-red-900">
                        Show component stack
                      </summary>
                      <pre className="mt-2 text-xs overflow-x-auto bg-white p-2 rounded border border-red-300">
                        {this.state.errorInfo.componentStack}
                      </pre>
                    </details>
                  )}
                </div>

                {/* Action Buttons */}
                <div className="flex gap-4">
                  <button
                    onClick={this.handleReset}
                    className="bg-red-600 text-white px-6 py-2 rounded hover:bg-red-700 transition"
                  >
                    Reload Application
                  </button>
                  <button
                    onClick={() => {
                      const errorText = `Error: ${this.state.error?.message}\n\nStack:\n${this.state.error?.stack}`;
                      navigator.clipboard.writeText(errorText);
                      alert('Error details copied to clipboard');
                    }}
                    className="bg-gray-200 text-gray-800 px-6 py-2 rounded hover:bg-gray-300 transition"
                  >
                    Copy Error Details
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
