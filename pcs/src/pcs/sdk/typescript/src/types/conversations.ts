/**
 * @fileoverview Conversation-related types for PCS TypeScript SDK
 */

import type { PCSResponse, KeyValueObject, UUID, Timestamp } from './base'

/**
 * Conversation status
 */
export type ConversationStatus = 'active' | 'paused' | 'completed' | 'archived'

/**
 * Conversation priority levels
 */
export type ConversationPriority = 'low' | 'normal' | 'high' | 'urgent'

/**
 * Message sender roles
 */
export type MessageRole = 'user' | 'assistant' | 'system'

/**
 * Message content types
 */
export type MessageType = 'text' | 'code' | 'error' | 'system_notification'

/**
 * Create conversation request
 */
export interface ConversationCreate {
  /** Conversation title */
  title: string
  /** Conversation description */
  description?: string
  /** User ID */
  user_id: string
  /** Project ID */
  project_id?: string
  /** Session ID */
  session_id?: string
  /** Priority level */
  priority?: ConversationPriority
  /** Conversation metadata */
  conversation_metadata?: KeyValueObject
  /** Conversation settings */
  settings?: KeyValueObject
  /** Associated context IDs */
  context_ids?: string[]
  /** Active prompt template */
  active_prompt_template_id?: UUID
}

/**
 * Update conversation request
 */
export interface ConversationUpdate {
  /** Conversation title */
  title?: string
  /** Conversation description */
  description?: string
  /** Status */
  status?: ConversationStatus
  /** Priority */
  priority?: ConversationPriority
  /** Conversation metadata */
  conversation_metadata?: KeyValueObject
  /** Conversation settings */
  settings?: KeyValueObject
  /** Context IDs */
  context_ids?: string[]
  /** Active prompt template */
  active_prompt_template_id?: UUID
}

/**
 * Conversation response
 */
export interface ConversationResponse extends PCSResponse {
  /** Conversation title */
  title: string
  /** Conversation description */
  description?: string
  /** User ID */
  user_id: string
  /** Project ID */
  project_id?: string
  /** Session ID */
  session_id?: string
  /** Current status */
  status: ConversationStatus
  /** Priority level */
  priority: ConversationPriority
  /** Conversation metadata */
  conversation_metadata: KeyValueObject
  /** Conversation settings */
  settings: KeyValueObject
  /** Associated context IDs */
  context_ids: string[]
  /** Active prompt template */
  active_prompt_template_id?: UUID
  /** Start time */
  started_at: Timestamp
  /** Last activity time */
  last_activity_at: Timestamp
  /** End time */
  ended_at?: Timestamp
  /** Number of messages */
  message_count: number
  /** Total tokens used */
  total_tokens: number
}

/**
 * Create message request
 */
export interface MessageCreate {
  /** Message sender role */
  role: MessageRole
  /** Message content */
  content: string
  /** Message type */
  message_type?: MessageType
  /** Raw content before processing */
  raw_content?: string
  /** Message metadata */
  message_metadata?: KeyValueObject
  /** Prompt template used */
  prompt_template_id?: UUID
  /** Context IDs used */
  context_ids?: string[]
  /** Parent message for threading */
  parent_message_id?: UUID
}

/**
 * Message response
 */
export interface MessageResponse extends PCSResponse {
  /** Parent conversation ID */
  conversation_id: UUID
  /** Message sequence number */
  sequence_number: number
  /** Message sender role */
  role: MessageRole
  /** Message type */
  message_type: MessageType
  /** Message content */
  content: string
  /** Raw content */
  raw_content?: string
  /** Message metadata */
  message_metadata: KeyValueObject
  /** Prompt template used */
  prompt_template_id?: UUID
  /** Context IDs used */
  context_ids: string[]
  /** Input tokens */
  input_tokens?: number
  /** Output tokens */
  output_tokens?: number
  /** Total tokens */
  total_tokens?: number
  /** Processing cost */
  cost?: number
  /** AI model used */
  model_used?: string
  /** Processing time */
  processing_time_ms?: number
  /** Edit status */
  is_edited: boolean
  /** Delete status */
  is_deleted: boolean
  /** Parent message ID */
  parent_message_id?: UUID
}

/**
 * Conversation list filters
 */
export interface ConversationListFilters {
  /** Page number (1-based) */
  page?: number
  /** Page size (1-100) */
  size?: number
  /** Filter by user ID */
  user_id?: string
  /** Filter by project ID */
  project_id?: string
  /** Filter by session ID */
  session_id?: string
  /** Filter by status */
  status?: ConversationStatus
  /** Filter by priority */
  priority?: ConversationPriority
  /** Search in title and description */
  search?: string
}

/**
 * Message history options
 */
export interface MessageHistoryOptions {
  /** Maximum number of messages */
  limit?: number
  /** Number of messages to skip */
  offset?: number
  /** Include deleted messages */
  include_deleted?: boolean
}
