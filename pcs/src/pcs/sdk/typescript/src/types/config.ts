/**
 * @fileoverview Configuration types for PCS TypeScript SDK
 */

/**
 * Supported authentication methods
 */
export type AuthenticationMethod = 'api_key' | 'jwt_token' | 'none'

/**
 * Logging levels for SDK operations
 */
export type LogLevel = 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL'

/**
 * Retry configuration for failed requests
 */
export interface RetryConfig {
  /** Maximum number of retry attempts */
  max_retries?: number
  /** Base delay in milliseconds for exponential backoff */
  base_delay?: number
  /** Maximum delay between retries in milliseconds */
  max_delay?: number
  /** Multiplier for exponential backoff */
  backoff_factor?: number
  /** Whether to add random jitter to delays */
  jitter?: boolean
  /** Retry on connection errors */
  retry_on_connection_errors?: boolean
  /** Retry on timeout errors */
  retry_on_timeout?: boolean
  /** Retry on rate limit errors (429) */
  retry_on_rate_limit?: boolean
  /** Retry on server errors (5xx) */
  retry_on_server_errors?: boolean
}

/**
 * HTTP client configuration
 */
export interface HTTPConfig {
  /** Request timeout in milliseconds */
  timeout?: number
  /** Connection timeout in milliseconds */
  connect_timeout?: number
  /** Read timeout in milliseconds */
  read_timeout?: number
  /** Keep-alive enabled */
  keep_alive?: boolean
  /** Verify SSL certificates */
  verify_ssl?: boolean
  /** User-Agent header value */
  user_agent?: string
  /** Additional default headers */
  headers?: Record<string, string>
}

/**
 * Main configuration for PCS TypeScript SDK client
 */
export interface PCSClientConfig {
  /** Base URL for PCS API */
  base_url?: string
  /** API key for authentication */
  api_key?: string
  /** JWT token for authentication (alternative to API key) */
  jwt_token?: string
  /** API version to use */
  api_version?: string
  /** Environment name (for logging/debugging) */
  environment?: string
  
  /** Retry behavior configuration */
  retry_config?: RetryConfig
  /** HTTP client configuration */
  http_config?: HTTPConfig
  
  /** Logging level for SDK operations */
  log_level?: LogLevel
  /** Whether to log HTTP requests */
  log_requests?: boolean
  /** Whether to log HTTP responses */
  log_responses?: boolean
  
  /** Enable client-side rate limit tracking */
  rate_limit_tracking?: boolean
  /** Enable response caching (for read operations) */
  cache_enabled?: boolean
  /** Cache TTL in seconds */
  cache_ttl?: number
  
  /** Validate SSL certificates */
  validate_ssl?: boolean
  /** Enable debug mode with verbose logging */
  debug_mode?: boolean
}

/**
 * Default configuration values
 */
export const DEFAULT_CONFIG: Required<PCSClientConfig> = {
  base_url: 'http://localhost:8000',
  api_key: '',
  jwt_token: '',
  api_version: 'v1',
  environment: 'production',
  
  retry_config: {
    max_retries: 3,
    base_delay: 1000,
    max_delay: 60000,
    backoff_factor: 2.0,
    jitter: true,
    retry_on_connection_errors: true,
    retry_on_timeout: true,
    retry_on_rate_limit: true,
    retry_on_server_errors: true,
  },
  
  http_config: {
    timeout: 30000,
    connect_timeout: 10000,
    read_timeout: 30000,
    keep_alive: true,
    verify_ssl: true,
    user_agent: 'PCS-TypeScript-SDK/1.0.0',
    headers: {},
  },
  
  log_level: 'INFO',
  log_requests: false,
  log_responses: false,
  
  rate_limit_tracking: true,
  cache_enabled: false,
  cache_ttl: 300,
  
  validate_ssl: true,
  debug_mode: false,
}

/**
 * Create configuration with defaults
 */
export function createConfig(config: PCSClientConfig = {}): Required<PCSClientConfig> {
  return {
    ...DEFAULT_CONFIG,
    ...config,
    retry_config: { ...DEFAULT_CONFIG.retry_config, ...config.retry_config },
    http_config: { ...DEFAULT_CONFIG.http_config, ...config.http_config },
  }
}

/**
 * Create configuration from environment variables
 */
export function createConfigFromEnvironment(): PCSClientConfig {
  const env = typeof process !== 'undefined' ? process.env : {}
  
  return {
    base_url: env.PCS_BASE_URL,
    api_key: env.PCS_API_KEY,
    jwt_token: env.PCS_JWT_TOKEN,
    api_version: env.PCS_API_VERSION,
    environment: env.PCS_ENVIRONMENT,
    debug_mode: env.PCS_DEBUG?.toLowerCase() === 'true',
    log_level: (env.PCS_LOG_LEVEL as LogLevel) || undefined,
    
    retry_config: {
      max_retries: env.PCS_MAX_RETRIES ? parseInt(env.PCS_MAX_RETRIES, 10) : undefined,
    },
    
    http_config: {
      timeout: env.PCS_TIMEOUT ? parseFloat(env.PCS_TIMEOUT) * 1000 : undefined,
    },
  }
}
