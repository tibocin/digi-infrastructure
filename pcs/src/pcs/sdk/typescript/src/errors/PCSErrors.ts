/**
 * @fileoverview Error classes for PCS TypeScript SDK
 */

import type { PCSError, PCSErrorDetails, PCSErrorType } from '../types/errors'

/**
 * Base PCS SDK error class
 */
export class PCSSDKError extends Error implements PCSError {
  public readonly details: PCSErrorDetails
  public readonly timestamp: string
  public readonly cause?: Error

  constructor(
    message: string,
    details: Partial<PCSErrorDetails> = {},
    cause?: Error
  ) {
    super(message)
    this.name = this.constructor.name
    this.cause = cause
    this.timestamp = new Date().toISOString()
    
    this.details = {
      type: 'unknown' as PCSErrorType,
      is_retryable: false,
      ...details,
    }

    // Maintain proper stack trace (V8 only)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor)
    }
  }

  /**
   * Convert error to JSON representation
   */
  toJSON(): Record<string, any> {
    return {
      name: this.name,
      message: this.message,
      timestamp: this.timestamp,
      details: this.details,
      stack: this.stack,
    }
  }

  /**
   * String representation with context
   */
  toString(): string {
    const parts = [this.message]
    
    if (this.details.status_code) {
      parts.push(`Status: ${this.details.status_code}`)
    }
    
    if (this.details.error_code) {
      parts.push(`Code: ${this.details.error_code}`)
    }
    
    if (this.details.request_id) {
      parts.push(`Request ID: ${this.details.request_id}`)
    }
    
    return parts.join(' | ')
  }
}

/**
 * Authentication failed error
 */
export class PCSAuthenticationError extends PCSSDKError {
  constructor(
    message: string = 'Authentication failed',
    details: Partial<PCSErrorDetails> = {},
    cause?: Error
  ) {
    super(message, {
      type: 'authentication',
      status_code: 401,
      is_retryable: false,
      ...details,
    }, cause)
  }
}

/**
 * Authorization failed error
 */
export class PCSAuthorizationError extends PCSSDKError {
  constructor(
    message: string = 'Authorization failed - insufficient permissions',
    details: Partial<PCSErrorDetails> = {},
    cause?: Error
  ) {
    super(message, {
      type: 'authorization',
      status_code: 403,
      is_retryable: false,
      ...details,
    }, cause)
  }
}

/**
 * Request validation failed error
 */
export class PCSValidationError extends PCSSDKError {
  constructor(
    message: string = 'Request validation failed',
    details: Partial<PCSErrorDetails> = {},
    cause?: Error
  ) {
    super(message, {
      type: 'validation',
      status_code: 422,
      is_retryable: false,
      ...details,
    }, cause)
  }

  /**
   * Get validation errors
   */
  get validationErrors() {
    return this.details.validation_errors || []
  }
}

/**
 * Rate limit exceeded error
 */
export class PCSRateLimitError extends PCSSDKError {
  constructor(
    message: string = 'Rate limit exceeded',
    details: Partial<PCSErrorDetails> = {},
    cause?: Error
  ) {
    super(message, {
      type: 'rate_limit',
      status_code: 429,
      is_retryable: true,
      ...details,
    }, cause)
  }

  /**
   * Get retry-after time in seconds
   */
  get retryAfter(): number | undefined {
    return this.details.retry_after
  }
}

/**
 * Network connection error
 */
export class PCSConnectionError extends PCSSDKError {
  constructor(
    message: string = 'Connection error',
    details: Partial<PCSErrorDetails> = {},
    cause?: Error
  ) {
    super(message, {
      type: 'connection',
      is_retryable: true,
      ...details,
    }, cause)
  }
}

/**
 * Request timeout error
 */
export class PCSTimeoutError extends PCSSDKError {
  constructor(
    message: string = 'Request timeout',
    details: Partial<PCSErrorDetails> = {},
    cause?: Error
  ) {
    super(message, {
      type: 'timeout',
      status_code: 408,
      is_retryable: true,
      ...details,
    }, cause)
  }
}

/**
 * Server error (5xx status codes)
 */
export class PCSServerError extends PCSSDKError {
  constructor(
    message: string = 'Server error',
    details: Partial<PCSErrorDetails> = {},
    cause?: Error
  ) {
    super(message, {
      type: 'server',
      status_code: 500,
      is_retryable: true,
      ...details,
    }, cause)
  }
}

/**
 * Resource not found error
 */
export class PCSNotFoundError extends PCSSDKError {
  constructor(
    message: string = 'Resource not found',
    details: Partial<PCSErrorDetails> = {},
    cause?: Error
  ) {
    super(message, {
      type: 'not_found',
      status_code: 404,
      is_retryable: false,
      ...details,
    }, cause)
  }
}

/**
 * Resource conflict error
 */
export class PCSConflictError extends PCSSDKError {
  constructor(
    message: string = 'Resource conflict',
    details: Partial<PCSErrorDetails> = {},
    cause?: Error
  ) {
    super(message, {
      type: 'conflict',
      status_code: 409,
      is_retryable: false,
      ...details,
    }, cause)
  }
}
