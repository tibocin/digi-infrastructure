/**
 * @fileoverview Main PCS TypeScript SDK client
 */

import 'isomorphic-fetch'

import type {
  PCSClientConfig,
  RetryConfig,
  HTTPConfig,
} from '../types/config'
import type {
  PaginatedResponse,
  HealthResponse,
  MetricsResponse,
  KeyValueObject,
  UUID,
} from '../types/base'
import type {
  PromptCreate,
  PromptUpdate,
  PromptResponse,
  GeneratePromptRequest,
  GeneratedPromptResponse,
  PromptListFilters,
} from '../types/prompts'
import type {
  ContextCreate,
  ContextUpdate,
  ContextResponse,
  ContextMergeRequest,
  ContextMergeResponse,
  ContextListFilters,
} from '../types/contexts'
import type {
  ConversationCreate,
  ConversationUpdate,
  ConversationResponse,
  MessageCreate,
  MessageResponse,
  ConversationListFilters,
  MessageHistoryOptions,
} from '../types/conversations'
import type { ErrorResponseData } from '../types/errors'

import { createConfig, DEFAULT_CONFIG } from '../types/config'
import { createPCSErrorFromResponse, createConnectionError, createTimeoutError } from '../errors/errorFactory'
import { PCSSDKError } from '../errors/PCSErrors'

/**
 * HTTP response interface
 */
interface HTTPResponse {
  status: number
  statusText: string
  headers: Headers
  json(): Promise<any>
  text(): Promise<string>
}

/**
 * Request options interface
 */
interface RequestOptions {
  method: string
  headers: Record<string, string>
  body?: string
  signal?: AbortSignal
}

/**
 * Main PCS TypeScript SDK client
 * 
 * Provides comprehensive access to all PCS API endpoints with:
 * - Promise-based async operations
 * - Automatic retry with exponential backoff
 * - Type-safe request/response handling
 * - Structured error handling
 * - JWT and API key authentication
 * - Browser and Node.js compatibility
 * 
 * @example
 * ```typescript
 * import { PCSClient } from '@pcs/typescript-sdk'
 * 
 * const client = new PCSClient({
 *   base_url: 'https://api.pcs.example.com',
 *   api_key: 'your-api-key'
 * })
 * 
 * // Create a prompt
 * const prompt = await client.createPrompt({
 *   name: 'greeting',
 *   content: 'Hello {{name}}!',
 *   variables: { name: 'World' }
 * })
 * 
 * // Generate from template
 * const result = await client.generatePrompt('greeting', {
 *   context: { name: 'Alice' }
 * })
 * console.log(result.generated_prompt) // "Hello Alice!"
 * ```
 */
export class PCSClient {
  private readonly config: Required<PCSClientConfig>
  private readonly baseUrl: string

  constructor(config: PCSClientConfig = {}) {
    this.config = createConfig(config)
    
    // Normalize base URL
    let baseUrl = this.config.base_url.replace(/\/$/, '')
    if (!baseUrl.endsWith(`/api/${this.config.api_version}`)) {
      baseUrl = `${baseUrl}/api/${this.config.api_version}`
    }
    this.baseUrl = baseUrl

    // Validate authentication
    if (!this.config.api_key && !this.config.jwt_token) {
      throw new PCSSDKError(
        'No authentication configured. Set either api_key or jwt_token.',
        { type: 'authentication' }
      )
    }
  }

  /**
   * Make HTTP request with retry logic and error handling
   */
  private async makeRequest<T = any>(
    endpoint: string,
    options: {
      method?: string
      params?: Record<string, any>
      body?: any
      headers?: Record<string, string>
    } = {}
  ): Promise<T> {
    const {
      method = 'GET',
      params,
      body,
      headers: extraHeaders = {},
    } = options

    // Build URL with query parameters
    const url = new URL(endpoint.replace(/^\//, ''), this.baseUrl)
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          url.searchParams.append(key, String(value))
        }
      })
    }

    // Prepare headers
    const headers: Record<string, string> = {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
      'User-Agent': this.config.http_config.user_agent,
      ...this.config.http_config.headers,
      ...extraHeaders,
    }

    // Add authentication
    if (this.config.jwt_token) {
      headers['Authorization'] = `Bearer ${this.config.jwt_token}`
    } else if (this.config.api_key) {
      headers['X-API-Key'] = this.config.api_key
    }

    // Prepare request options
    const requestOptions: RequestOptions = {
      method,
      headers,
    }

    if (body && method !== 'GET' && method !== 'HEAD') {
      requestOptions.body = JSON.stringify(body)
    }

    // Setup timeout
    const controller = new AbortController()
    const timeoutId = setTimeout(() => {
      controller.abort()
    }, this.config.http_config.timeout)

    requestOptions.signal = controller.signal

    // Retry logic
    const retryConfig = this.config.retry_config
    let lastError: Error | undefined

    for (let attempt = 0; attempt <= retryConfig.max_retries; attempt++) {
      try {
        const startTime = Date.now()

        // Log request if enabled
        if (this.config.log_requests) {
          console.debug(`Making ${method} request to ${url.pathname}`, {
            method,
            url: url.toString(),
            attempt: attempt + 1,
          })
        }

        // Make HTTP request
        const response = await fetch(url.toString(), requestOptions)

        // Log response if enabled
        if (this.config.log_responses) {
          const elapsedMs = Date.now() - startTime
          console.debug(`Response: ${response.status} in ${elapsedMs}ms`, {
            status: response.status,
            elapsed_ms: elapsedMs,
            attempt: attempt + 1,
          })
        }

        // Clear timeout
        clearTimeout(timeoutId)

        // Handle response
        if (response.ok) {
          const contentType = response.headers.get('content-type')
          if (contentType?.includes('application/json')) {
            return await response.json()
          } else {
            return (await response.text()) as any
          }
        } else {
          // Parse error response
          let errorData: ErrorResponseData = {}
          try {
            errorData = await response.json()
          } catch {
            errorData = { message: response.statusText || 'Unknown error' }
          }

          const requestId = response.headers.get('X-Request-ID') || undefined
          const error = createPCSErrorFromResponse(
            response.status,
            errorData,
            requestId
          )

          // Check if we should retry
          if (attempt < retryConfig.max_retries && error.details.is_retryable) {
            lastError = error
            await this.waitForRetry(attempt, retryConfig, error)
            continue
          } else {
            throw error
          }
        }

      } catch (error) {
        clearTimeout(timeoutId)

        if (error instanceof PCSSDKError) {
          throw error
        }

        // Handle network errors
        if (error instanceof Error) {
          if (error.name === 'AbortError') {
            const timeoutError = createTimeoutError(this.config.http_config.timeout, error)
            if (attempt < retryConfig.max_retries && retryConfig.retry_on_timeout) {
              lastError = timeoutError
              await this.waitForRetry(attempt, retryConfig, timeoutError)
              continue
            } else {
              throw timeoutError
            }
          } else {
            const connectionError = createConnectionError(error.message, error)
            if (attempt < retryConfig.max_retries && retryConfig.retry_on_connection_errors) {
              lastError = connectionError
              await this.waitForRetry(attempt, retryConfig, connectionError)
              continue
            } else {
              throw connectionError
            }
          }
        }

        // Unexpected error
        throw new PCSSDKError(`Unexpected error: ${error}`, { type: 'unknown' })
      }
    }

    // If we get here, all retries failed
    if (lastError) {
      throw lastError
    } else {
      throw new PCSSDKError('Request failed after all retry attempts')
    }
  }

  /**
   * Wait before retrying request with exponential backoff
   */
  private async waitForRetry(
    attempt: number,
    retryConfig: RetryConfig,
    error: Error
  ): Promise<void> {
    let delay: number

    // Use server-specified retry delay for rate limit errors
    if (error instanceof Error && 'details' in error) {
      const details = (error as any).details
      if (details?.retry_after) {
        delay = details.retry_after * 1000 // Convert to milliseconds
      } else {
        // Exponential backoff with jitter
        delay = Math.min(
          retryConfig.base_delay * Math.pow(retryConfig.backoff_factor, attempt),
          retryConfig.max_delay
        )

        if (retryConfig.jitter) {
          // Add up to 25% jitter
          const jitter = delay * 0.25 * Math.random()
          delay += jitter
        }
      }
    } else {
      delay = retryConfig.base_delay
    }

    if (this.config.log_level === 'DEBUG') {
      console.debug(`Retrying in ${delay}ms after ${error.constructor.name}`, {
        delay,
        attempt: attempt + 1,
      })
    }

    await new Promise(resolve => setTimeout(resolve, delay))
  }

  // ========================================================================
  // Health and System Endpoints
  // ========================================================================

  /**
   * Get basic system health status
   */
  async getHealth(): Promise<HealthResponse> {
    return this.makeRequest<HealthResponse>('/health')
  }

  /**
   * Get system metrics and performance data
   */
  async getMetrics(): Promise<MetricsResponse> {
    return this.makeRequest<MetricsResponse>('/admin/metrics')
  }

  // ========================================================================
  // Prompt Template Management
  // ========================================================================

  /**
   * Create a new prompt template
   */
  async createPrompt(promptData: PromptCreate): Promise<PromptResponse> {
    return this.makeRequest<PromptResponse>('/prompts', {
      method: 'POST',
      body: promptData,
    })
  }

  /**
   * Get a prompt template by ID
   */
  async getPrompt(promptId: UUID): Promise<PromptResponse> {
    return this.makeRequest<PromptResponse>(`/prompts/${promptId}`)
  }

  /**
   * Get a prompt template by name
   */
  async getPromptByName(name: string): Promise<PromptResponse> {
    return this.makeRequest<PromptResponse>(`/prompts/by-name/${encodeURIComponent(name)}`)
  }

  /**
   * Update a prompt template
   */
  async updatePrompt(promptId: UUID, updates: PromptUpdate): Promise<PromptResponse> {
    return this.makeRequest<PromptResponse>(`/prompts/${promptId}`, {
      method: 'PUT',
      body: updates,
    })
  }

  /**
   * Delete a prompt template
   */
  async deletePrompt(promptId: UUID): Promise<boolean> {
    await this.makeRequest(`/prompts/${promptId}`, { method: 'DELETE' })
    return true
  }

  /**
   * List prompt templates with filtering and pagination
   */
  async listPrompts(filters: PromptListFilters = {}): Promise<PaginatedResponse<PromptResponse>> {
    return this.makeRequest<PaginatedResponse<PromptResponse>>('/prompts', {
      params: filters,
    })
  }

  /**
   * Generate a prompt from a template with context
   */
  async generatePrompt(
    templateNameOrOptions: string | GeneratePromptRequest,
    options: Partial<GeneratePromptRequest> = {}
  ): Promise<GeneratedPromptResponse> {
    let requestData: GeneratePromptRequest

    if (typeof templateNameOrOptions === 'string') {
      requestData = {
        template_name: templateNameOrOptions,
        ...options,
      }
    } else {
      requestData = templateNameOrOptions
    }

    return this.makeRequest<GeneratedPromptResponse>('/prompts/generate', {
      method: 'POST',
      body: requestData,
    })
  }

  // ========================================================================
  // Context Management
  // ========================================================================

  /**
   * Create a new context
   */
  async createContext(contextData: ContextCreate): Promise<ContextResponse> {
    return this.makeRequest<ContextResponse>('/contexts', {
      method: 'POST',
      body: contextData,
    })
  }

  /**
   * Get a context by ID
   */
  async getContext(contextId: UUID): Promise<ContextResponse> {
    return this.makeRequest<ContextResponse>(`/contexts/${contextId}`)
  }

  /**
   * Update a context
   */
  async updateContext(contextId: UUID, updates: ContextUpdate): Promise<ContextResponse> {
    return this.makeRequest<ContextResponse>(`/contexts/${contextId}`, {
      method: 'PUT',
      body: updates,
    })
  }

  /**
   * Delete a context
   */
  async deleteContext(contextId: UUID): Promise<boolean> {
    await this.makeRequest(`/contexts/${contextId}`, { method: 'DELETE' })
    return true
  }

  /**
   * List contexts with filtering and pagination
   */
  async listContexts(filters: ContextListFilters = {}): Promise<PaginatedResponse<ContextResponse>> {
    return this.makeRequest<PaginatedResponse<ContextResponse>>('/contexts', {
      params: filters,
    })
  }

  /**
   * Merge multiple contexts into one
   */
  async mergeContexts(mergeRequest: ContextMergeRequest): Promise<ContextMergeResponse> {
    return this.makeRequest<ContextMergeResponse>('/contexts/merge', {
      method: 'POST',
      body: mergeRequest,
    })
  }

  // ========================================================================
  // Conversation Management
  // ========================================================================

  /**
   * Create a new conversation
   */
  async createConversation(conversationData: ConversationCreate): Promise<ConversationResponse> {
    return this.makeRequest<ConversationResponse>('/conversations', {
      method: 'POST',
      body: conversationData,
    })
  }

  /**
   * Get a conversation by ID
   */
  async getConversation(conversationId: UUID): Promise<ConversationResponse> {
    return this.makeRequest<ConversationResponse>(`/conversations/${conversationId}`)
  }

  /**
   * Update a conversation
   */
  async updateConversation(
    conversationId: UUID,
    updates: ConversationUpdate
  ): Promise<ConversationResponse> {
    return this.makeRequest<ConversationResponse>(`/conversations/${conversationId}`, {
      method: 'PUT',
      body: updates,
    })
  }

  /**
   * Delete a conversation
   */
  async deleteConversation(conversationId: UUID): Promise<boolean> {
    await this.makeRequest(`/conversations/${conversationId}`, { method: 'DELETE' })
    return true
  }

  /**
   * List conversations with filtering and pagination
   */
  async listConversations(
    filters: ConversationListFilters = {}
  ): Promise<PaginatedResponse<ConversationResponse>> {
    return this.makeRequest<PaginatedResponse<ConversationResponse>>('/conversations', {
      params: filters,
    })
  }

  /**
   * Add a message to a conversation
   */
  async addMessage(conversationId: UUID, message: MessageCreate): Promise<MessageResponse> {
    return this.makeRequest<MessageResponse>(`/conversations/${conversationId}/messages`, {
      method: 'POST',
      body: message,
    })
  }

  /**
   * Get conversation message history
   */
  async getConversationHistory(
    conversationId: UUID,
    options: MessageHistoryOptions = {}
  ): Promise<MessageResponse[]> {
    return this.makeRequest<MessageResponse[]>(`/conversations/${conversationId}/messages`, {
      params: options,
    })
  }

  // ========================================================================
  // Utility Methods
  // ========================================================================

  /**
   * Get authentication information
   */
  getAuthInfo(): { method: string; authenticated: boolean } {
    return {
      method: this.config.jwt_token ? 'jwt_token' : this.config.api_key ? 'api_key' : 'none',
      authenticated: Boolean(this.config.jwt_token || this.config.api_key),
    }
  }

  /**
   * Update authentication credentials
   */
  updateAuthentication(credentials: { api_key?: string; jwt_token?: string }): void {
    if (credentials.api_key !== undefined) {
      this.config.api_key = credentials.api_key
      this.config.jwt_token = ''
    } else if (credentials.jwt_token !== undefined) {
      this.config.jwt_token = credentials.jwt_token
      this.config.api_key = ''
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): Readonly<Required<PCSClientConfig>> {
    return { ...this.config }
  }
}