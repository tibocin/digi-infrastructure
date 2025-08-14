/**
 * @fileoverview Error factory for creating PCS errors from HTTP responses
 */

import type { ErrorResponseData } from '../types/errors'
import {
  PCSSDKError,
  PCSAuthenticationError,
  PCSAuthorizationError,
  PCSValidationError,
  PCSRateLimitError,
  PCSConnectionError,
  PCSTimeoutError,
  PCSServerError,
  PCSNotFoundError,
  PCSConflictError,
} from './PCSErrors'

/**
 * Create appropriate PCS error from HTTP response
 */
export function createPCSErrorFromResponse(
  statusCode: number,
  responseData: ErrorResponseData,
  requestId?: string
): PCSSDKError {
  const message = responseData.message || 'API request failed'
  const errorCode = responseData.error_code
  const details = responseData.details || {}
  
  const commonDetails = {
    status_code: statusCode,
    error_code: errorCode,
    details,
    request_id: requestId,
  }

  // Authentication/Authorization errors
  if (statusCode === 401) {
    return new PCSAuthenticationError(message, commonDetails)
  }
  
  if (statusCode === 403) {
    return new PCSAuthorizationError(message, commonDetails)
  }

  // Client errors
  if (statusCode === 404) {
    return new PCSNotFoundError(message, commonDetails)
  }
  
  if (statusCode === 409) {
    return new PCSConflictError(message, commonDetails)
  }
  
  if (statusCode === 422) {
    return new PCSValidationError(message, {
      ...commonDetails,
      validation_errors: responseData.validation_errors,
    })
  }
  
  if (statusCode === 429) {
    return new PCSRateLimitError(message, {
      ...commonDetails,
      retry_after: responseData.retry_after,
    })
  }

  // Server errors
  if (statusCode >= 500 && statusCode < 600) {
    return new PCSServerError(message, commonDetails)
  }

  // Timeout-like errors
  if (statusCode === 408) {
    return new PCSTimeoutError(message, commonDetails)
  }

  // Generic client errors
  if (statusCode >= 400 && statusCode < 500) {
    return new PCSValidationError(message, commonDetails)
  }

  // Fallback to generic error
  return new PCSSDKError(message, commonDetails)
}

/**
 * Create connection error from network failure
 */
export function createConnectionError(
  message: string,
  cause?: Error
): PCSConnectionError {
  return new PCSConnectionError(
    `Connection failed: ${message}`,
    {},
    cause
  )
}

/**
 * Create timeout error
 */
export function createTimeoutError(
  timeoutMs: number,
  cause?: Error
): PCSTimeoutError {
  return new PCSTimeoutError(
    `Request timeout after ${timeoutMs}ms`,
    { timeout_seconds: timeoutMs / 1000 },
    cause
  )
}
