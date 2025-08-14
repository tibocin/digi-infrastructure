/**
 * @fileoverview PCS TypeScript SDK - Main entry point
 * @version 1.0.0
 */

// Main client
export { PCSClient } from './client/PCSClient'

// Configuration and types
export type {
  PCSClientConfig,
  HTTPConfig,
  RetryConfig,
  AuthenticationMethod,
} from './types/config'

// Base types and responses
export type {
  PCSResponse,
  PaginatedResponse,
  HealthResponse,
  MetricsResponse,
} from './types/base'

// Prompt types
export type {
  PromptCreate,
  PromptUpdate,
  PromptResponse,
  PromptVersionResponse,
  GeneratePromptRequest,
  GeneratedPromptResponse,
  PromptStatus,
  RulePriority,
} from './types/prompts'

// Context types
export type {
  ContextCreate,
  ContextUpdate,
  ContextResponse,
  ContextMergeRequest,
  ContextMergeResponse,
  ContextTypeEnum,
  ContextScope,
  RelationshipType,
} from './types/contexts'

// Conversation types
export type {
  ConversationCreate,
  ConversationUpdate,
  ConversationResponse,
  MessageCreate,
  MessageResponse,
  ConversationStatus,
  ConversationPriority,
  MessageRole,
  MessageType,
} from './types/conversations'

// Exception types
export type {
  PCSError,
  PCSErrorType,
  PCSErrorDetails,
} from './types/errors'

// Exception classes
export {
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
} from './errors/PCSErrors'

// Utility functions
export { createPCSErrorFromResponse } from './errors/errorFactory'

// Version information
export const SDK_VERSION = '1.0.0'