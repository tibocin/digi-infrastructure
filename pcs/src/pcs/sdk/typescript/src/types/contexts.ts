/**
 * @fileoverview Context-related types for PCS TypeScript SDK
 */

import type { PCSResponse, KeyValueObject, UUID, Timestamp } from './base'

/**
 * Context type enumeration
 */
export type ContextTypeEnum = 'system' | 'user' | 'session' | 'project' | 'custom'

/**
 * Context scope levels
 */
export type ContextScope = 'global' | 'organization' | 'project' | 'user' | 'session'

/**
 * Context relationship types
 */
export type RelationshipType = 'depends_on' | 'extends' | 'includes' | 'references'

/**
 * Create context request
 */
export interface ContextCreate {
  /** Context name */
  name: string
  /** Context description */
  description?: string
  /** Context type reference */
  context_type_id: UUID
  /** Context scope */
  scope: ContextScope
  /** Owner ID */
  owner_id?: string
  /** Project ID */
  project_id?: string
  /** Context data payload */
  context_data: KeyValueObject
  /** Additional metadata */
  context_metadata?: KeyValueObject
  /** Context priority */
  priority?: number
  /** Vector embedding */
  vector_embedding?: number[]
  /** Embedding model used */
  embedding_model?: string
}

/**
 * Update context request
 */
export interface ContextUpdate {
  /** Context name */
  name?: string
  /** Context description */
  description?: string
  /** Context scope */
  scope?: ContextScope
  /** Owner ID */
  owner_id?: string
  /** Project ID */
  project_id?: string
  /** Context data */
  context_data?: KeyValueObject
  /** Context metadata */
  context_metadata?: KeyValueObject
  /** Context priority */
  priority?: number
  /** Vector embedding */
  vector_embedding?: number[]
  /** Embedding model */
  embedding_model?: string
  /** Active status */
  is_active?: boolean
}

/**
 * Context response
 */
export interface ContextResponse extends PCSResponse {
  /** Context name */
  name: string
  /** Context description */
  description?: string
  /** Context type reference */
  context_type_id: UUID
  /** Context scope */
  scope: ContextScope
  /** Owner ID */
  owner_id?: string
  /** Project ID */
  project_id?: string
  /** Context data */
  context_data: KeyValueObject
  /** Context metadata */
  context_metadata: KeyValueObject
  /** Context priority */
  priority: number
  /** Active status */
  is_active: boolean
  /** Vector embedding */
  vector_embedding?: number[]
  /** Embedding model */
  embedding_model?: string
  /** Number of times accessed */
  access_count: number
  /** Last access time */
  last_accessed_at?: Timestamp
}

/**
 * Context merge request
 */
export interface ContextMergeRequest {
  /** Base context to merge into */
  base_context_id: UUID
  /** Contexts to merge */
  merge_context_ids: UUID[]
  /** Merge strategy */
  merge_strategy?: string
  /** Conflict resolution */
  conflict_resolution?: string
  /** Preserve metadata */
  preserve_metadata?: boolean
}

/**
 * Context merge response
 */
export interface ContextMergeResponse {
  /** Resulting merged context */
  merged_context: ContextResponse
  /** Merge operation summary */
  merge_summary: KeyValueObject
  /** Conflicts that were resolved */
  conflicts_resolved: KeyValueObject[]
  /** Time taken to merge */
  merge_time_ms: number
}

/**
 * Context list filters
 */
export interface ContextListFilters {
  /** Page number (1-based) */
  page?: number
  /** Page size (1-100) */
  size?: number
  /** Filter by scope */
  scope?: ContextScope
  /** Filter by owner ID */
  owner_id?: string
  /** Filter by project ID */
  project_id?: string
  /** Search in name and description */
  search?: string
  /** Include only active contexts */
  active_only?: boolean
}
