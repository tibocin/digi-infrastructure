/**
 * @fileoverview Base types and interfaces for PCS TypeScript SDK
 */

/**
 * Base response interface with common fields
 */
export interface PCSResponse {
  /** Unique identifier */
  id: string
  /** Creation timestamp (ISO 8601) */
  created_at: string
  /** Last update timestamp (ISO 8601) */
  updated_at: string
}

/**
 * Paginated response wrapper
 */
export interface PaginatedResponse<T = any> {
  /** List of items */
  items: T[]
  /** Total number of items */
  total: number
  /** Current page number (1-based) */
  page: number
  /** Page size */
  size: number
  /** Total number of pages */
  pages: number
  /** Whether there are more pages */
  has_next: boolean
  /** Whether there are previous pages */
  has_prev: boolean
}

/**
 * Health check response
 */
export interface HealthResponse {
  /** Health status */
  status: string
  /** Check timestamp (ISO 8601) */
  timestamp: string
  /** Service version */
  version: string
  /** Environment name */
  environment: string
  /** Service uptime in seconds */
  uptime_seconds?: number
}

/**
 * System metrics response
 */
export interface MetricsResponse {
  /** Total requests processed */
  requests_total: number
  /** Current requests per second */
  requests_per_second: number
  /** Average response time in milliseconds */
  avg_response_time_ms: number
  /** Error rate percentage */
  error_rate: number
  /** Active connections */
  active_connections: number
  /** Memory usage in MB */
  memory_usage_mb: number
  /** CPU usage percentage */
  cpu_usage_percent: number
}

/**
 * Generic key-value object type
 */
export type KeyValueObject = Record<string, any>

/**
 * UUID string type
 */
export type UUID = string

/**
 * ISO 8601 timestamp string type
 */
export type Timestamp = string