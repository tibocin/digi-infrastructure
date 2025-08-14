/**
 * @fileoverview Error types and interfaces for PCS TypeScript SDK
 */

/**
 * Error severity levels
 */
export type PCSErrorType = 
  | 'authentication'
  | 'authorization'
  | 'validation'
  | 'rate_limit'
  | 'connection'
  | 'timeout'
  | 'server'
  | 'not_found'
  | 'conflict'
  | 'unknown'

/**
 * Detailed error information
 */
export interface PCSErrorDetails {
  /** Error type classification */
  type: PCSErrorType
  /** HTTP status code if applicable */
  status_code?: number
  /** Specific error code from API */
  error_code?: string
  /** Additional error details */
  details?: Record<string, any>
  /** Request ID for tracking */
  request_id?: string
  /** Whether this error should trigger a retry */
  is_retryable: boolean
  /** Validation errors for form submissions */
  validation_errors?: Array<{
    field: string
    message: string
    code?: string
  }>
  /** Retry-after time in seconds for rate limit errors */
  retry_after?: number
  /** Timeout duration for timeout errors */
  timeout_seconds?: number
  /** Resource information for not found errors */
  resource_type?: string
  resource_id?: string
}

/**
 * Base PCS error interface
 */
export interface PCSError {
  /** Error message */
  message: string
  /** Error details */
  details: PCSErrorDetails
  /** Original error if this is a wrapped error */
  cause?: Error
  /** Timestamp when error occurred */
  timestamp: string
}

/**
 * HTTP response error data
 */
export interface ErrorResponseData {
  /** Error message from server */
  message?: string
  /** Specific error code */
  error_code?: string
  /** Additional error details */
  details?: Record<string, any>
  /** Validation errors array */
  validation_errors?: Array<{
    field: string
    message: string
    code?: string
  }>
  /** Retry-after header value */
  retry_after?: number
}

/**
 * Type guard to check if an error is a PCS error
 */
export function isPCSError(error: any): error is PCSError {
  return (
    error && 
    typeof error === 'object' &&
    typeof error.message === 'string' &&
    error.details &&
    typeof error.details.type === 'string' &&
    typeof error.details.is_retryable === 'boolean'
  )
}

/**
 * Type guard to check if an error is retryable
 */
export function isRetryableError(error: any): boolean {
  if (isPCSError(error)) {
    return error.details.is_retryable
  }
  return false
}
