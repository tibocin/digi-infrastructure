/**
 * @fileoverview Prompt-related types for PCS TypeScript SDK
 */

import type { PCSResponse, KeyValueObject, UUID } from './base'

/**
 * Prompt template status
 */
export type PromptStatus = 'draft' | 'active' | 'archived' | 'deprecated'

/**
 * Rule priority levels
 */
export type RulePriority = 'low' | 'medium' | 'high' | 'critical'

/**
 * Create prompt template request
 */
export interface PromptCreate {
  /** Unique prompt name */
  name: string
  /** Prompt description */
  description?: string
  /** Category for organization */
  category?: string
  /** Tags for filtering */
  tags?: string[]
  /** Author name */
  author?: string
  /** Template content with variables */
  content: string
  /** Variable definitions */
  variables?: KeyValueObject
  /** Processing rules */
  rules?: KeyValueObject[]
  /** System template flag */
  is_system?: boolean
}

/**
 * Update prompt template request
 */
export interface PromptUpdate {
  /** Prompt name */
  name?: string
  /** Prompt description */
  description?: string
  /** Category */
  category?: string
  /** Tags */
  tags?: string[]
  /** Author */
  author?: string
  /** Template content */
  content?: string
  /** Variable definitions */
  variables?: KeyValueObject
  /** Processing rules */
  rules?: KeyValueObject[]
  /** Status */
  status?: PromptStatus
}

/**
 * Prompt template response
 */
export interface PromptResponse extends PCSResponse {
  /** Prompt name */
  name: string
  /** Prompt description */
  description?: string
  /** Category */
  category?: string
  /** Tags */
  tags: string[]
  /** Author */
  author?: string
  /** Current status */
  status: PromptStatus
  /** System template flag */
  is_system: boolean
  /** Number of versions */
  version_count: number
  /** Current active version */
  current_version?: PromptVersionResponse
}

/**
 * Prompt version response
 */
export interface PromptVersionResponse extends PCSResponse {
  /** Parent template ID */
  template_id: UUID
  /** Version number */
  version_number: number
  /** Template content */
  content: string
  /** Variable definitions */
  variables: KeyValueObject
  /** Processing rules */
  rules: KeyValueObject[]
  /** Version changelog */
  changelog?: string
  /** Active version flag */
  is_active: boolean
}

/**
 * Generate prompt request
 */
export interface GeneratePromptRequest {
  /** Template name to use */
  template_name?: string
  /** Template ID to use */
  template_id?: UUID
  /** Context variables */
  context?: KeyValueObject
  /** Context IDs to merge */
  context_ids?: string[]
  /** Optimization level */
  optimization_level?: string
}

/**
 * Generated prompt response
 */
export interface GeneratedPromptResponse {
  /** Generated prompt text */
  generated_prompt: string
  /** Source template ID */
  template_id: UUID
  /** Source template name */
  template_name: string
  /** Applied context variables */
  context_applied: KeyValueObject
  /** Applied rules */
  rules_applied: KeyValueObject[]
  /** Generation time in milliseconds */
  generation_time_ms: number
  /** Whether result was cached */
  cache_hit: boolean
}

/**
 * Prompt list filters
 */
export interface PromptListFilters {
  /** Page number (1-based) */
  page?: number
  /** Page size (1-100) */
  size?: number
  /** Filter by category */
  category?: string
  /** Filter by status */
  status?: PromptStatus
  /** Search in name and description */
  search?: string
  /** Include version data */
  include_versions?: boolean
  /** Include rule data */
  include_rules?: boolean
}
