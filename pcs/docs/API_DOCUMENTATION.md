# PCS API Documentation

## Overview

The Prompt and Context Service (PCS) provides a comprehensive REST API for managing prompts, contexts, conversations, and system administration. This documentation covers all available endpoints with examples and usage guidelines.

## Base Information

- **Base URL**: `http://localhost:8000/api/v1`
- **Authentication**: Bearer Token (JWT)
- **Content Type**: `application/json`
- **API Version**: v1

## Authentication

All API endpoints require authentication via JWT tokens in the Authorization header:

```bash
Authorization: Bearer <your_jwt_token>
```

### Admin Endpoints

Endpoints under `/admin` require administrator privileges.

## API Endpoints Overview

| Endpoint Group | Base Path        | Description                    |
| -------------- | ---------------- | ------------------------------ |
| Health         | `/health`        | System health and status       |
| Prompts        | `/prompts`       | Prompt template management     |
| Contexts       | `/contexts`      | Context management and merging |
| Conversations  | `/conversations` | Conversation lifecycle         |
| Admin          | `/admin`         | System administration          |

---

## 1. Health Endpoints

### GET /health

Check basic system health status.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "environment": "production"
}
```

---

## 2. Prompt Management API

### List Prompt Templates

**GET** `/prompts/`

**Query Parameters:**

- `page` (int): Page number (default: 1)
- `size` (int): Page size (default: 20, max: 100)
- `category` (string): Filter by category
- `status` (string): Filter by status (draft, active, archived, deprecated)
- `search` (string): Search in name and description
- `include_versions` (bool): Include version data
- `include_rules` (bool): Include rule data

**Example Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/prompts/?page=1&size=10&status=active" \
  -H "Authorization: Bearer <token>"
```

**Example Response:**

```json
{
  "items": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "name": "code-review-template",
      "description": "Template for code review prompts",
      "category": "development",
      "tags": ["code", "review", "quality"],
      "status": "active",
      "is_system": false,
      "author": "admin",
      "version_count": 3,
      "created_at": "2024-01-01T12:00:00Z",
      "updated_at": "2024-01-01T12:00:00Z"
    }
  ],
  "total": 25,
  "page": 1,
  "size": 10,
  "pages": 3
}
```

### Create Prompt Template

**POST** `/prompts/`

**Request Body:**

```json
{
  "name": "new-template",
  "description": "A new prompt template",
  "category": "general",
  "tags": ["example", "test"],
  "author": "user123"
}
```

**Response:** `201 Created`

```json
{
  "id": "456e7890-e89b-12d3-a456-426614174000",
  "name": "new-template",
  "description": "A new prompt template",
  "category": "general",
  "tags": ["example", "test"],
  "status": "draft",
  "is_system": false,
  "author": "user123",
  "version_count": 0,
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z"
}
```

### Get Prompt Template

**GET** `/prompts/{template_id}`

**Path Parameters:**

- `template_id` (UUID): Template identifier

**Query Parameters:**

- `include_versions` (bool): Include version data
- `include_rules` (bool): Include rule data

### Update Prompt Template

**PUT** `/prompts/{template_id}`

**Request Body:**

```json
{
  "description": "Updated description",
  "status": "active",
  "tags": ["updated", "active"]
}
```

### Delete Prompt Template

**DELETE** `/prompts/{template_id}`

**Response:** `204 No Content`

### Create Prompt Version

**POST** `/prompts/{template_id}/versions`

**Query Parameters:**

- `make_active` (bool): Make this version active

**Request Body:**

```json
{
  "content": "You are a helpful assistant. Context: {{context}}\n\nUser: {{user_input}}\n\nAssistant:",
  "variables": {
    "context": {
      "type": "string",
      "required": true,
      "description": "Conversation context"
    },
    "user_input": {
      "type": "string",
      "required": true,
      "description": "User's input message"
    }
  },
  "changelog": "Initial version with basic structure"
}
```

### List Prompt Versions

**GET** `/prompts/{template_id}/versions`

### Generate Prompt

**POST** `/prompts/generate`

**Request Body:**

```json
{
  "template_name": "code-review-template",
  "context_data": {
    "language": "python",
    "file_type": "class",
    "complexity": "medium"
  },
  "variables": {
    "code_snippet": "class ExampleClass:\n    def method(self):\n        pass"
  },
  "optimization_level": "basic"
}
```

**Response:**

```json
{
  "request_id": "req_789",
  "generated_prompt": "Please review this Python class...",
  "status": "completed",
  "processing_time_ms": 45.2,
  "context_used": {
    "language": "python",
    "file_type": "class"
  },
  "rules_applied": ["python-best-practices"],
  "variables_resolved": {
    "code_snippet": "class ExampleClass:..."
  },
  "cache_hit": false,
  "metadata": {
    "optimization_level": "basic"
  },
  "created_at": "2024-01-01T12:00:00Z"
}
```

---

## 3. Context Management API

### List Context Types

**GET** `/contexts/types`

**Query Parameters:**

- `page` (int): Page number
- `size` (int): Page size
- `active_only` (bool): Filter active types only
- `supports_vectors` (bool): Filter by vector support

### Create Context Type

**POST** `/contexts/types`

**Request Body:**

```json
{
  "name": "user-preferences",
  "description": "User preference context type",
  "type_enum": "user_preference",
  "schema_definition": {
    "type": "object",
    "properties": {
      "theme": { "type": "string" },
      "language": { "type": "string" },
      "timezone": { "type": "string" }
    },
    "required": ["theme", "language"]
  },
  "default_scope": "user",
  "supports_vectors": false
}
```

### List Contexts

**GET** `/contexts/`

**Query Parameters:**

- `page` (int): Page number
- `size` (int): Page size
- `context_type_id` (UUID): Filter by context type
- `scope` (string): Filter by scope (global, project, user, session, private)
- `owner_id` (string): Filter by owner
- `project_id` (string): Filter by project
- `search` (string): Search in name and description
- `active_only` (bool): Filter active contexts only
- `include_type` (bool): Include context type data
- `include_relationships` (bool): Include relationship data

### Create Context

**POST** `/contexts/`

**Request Body:**

```json
{
  "context_type_id": "123e4567-e89b-12d3-a456-426614174000",
  "name": "user-john-preferences",
  "description": "John's user preferences",
  "scope": "private",
  "owner_id": "user_john",
  "project_id": "project_123",
  "context_data": {
    "theme": "dark",
    "language": "en",
    "timezone": "America/New_York",
    "notifications": {
      "email": true,
      "push": false
    }
  },
  "context_metadata": {
    "source": "user_settings_form",
    "version": "1.0"
  },
  "priority": 1
}
```

### Get Context

**GET** `/contexts/{context_id}`

### Update Context

**PUT** `/contexts/{context_id}`

### Delete Context

**DELETE** `/contexts/{context_id}`

### Merge Contexts

**POST** `/contexts/merge`

**Request Body:**

```json
{
  "source_context_ids": [
    "123e4567-e89b-12d3-a456-426614174000",
    "456e7890-e89b-12d3-a456-426614174001"
  ],
  "merge_strategy": "merge_deep",
  "conflict_resolution": {
    "theme": "use_first",
    "notifications": "merge"
  },
  "preserve_metadata": true,
  "create_new": true
}
```

**Response:**

```json
{
  "merge_id": "merge_abc123",
  "result_context_id": "789e0123-e89b-12d3-a456-426614174002",
  "source_context_ids": ["123e4567...", "456e7890..."],
  "merge_strategy": "merge_deep",
  "conflicts_resolved": {
    "theme": "used_first_value",
    "notifications": "merged_objects"
  },
  "processing_time_ms": 12.5,
  "merged_fields": ["theme", "language", "timezone", "notifications"],
  "preserved_fields": [],
  "metadata": {
    "total_fields": 4,
    "conflicts_count": 2,
    "preserve_metadata": true
  },
  "created_at": "2024-01-01T12:00:00Z"
}
```

### Search Contexts

**POST** `/contexts/search`

**Request Body:**

```json
{
  "query": "user preferences",
  "context_types": ["user_preference"],
  "scopes": ["private", "user"],
  "owner_ids": ["user_john"],
  "date_from": "2024-01-01T00:00:00Z",
  "date_to": "2024-12-31T23:59:59Z",
  "include_inactive": false
}
```

---

## 4. Conversation Management API

### List Conversations

**GET** `/conversations/`

**Query Parameters:**

- `page` (int): Page number
- `size` (int): Page size
- `user_id` (string): Filter by user ID (admin only)
- `project_id` (string): Filter by project ID
- `session_id` (string): Filter by session ID
- `status` (string): Filter by status (active, paused, completed, archived, error)
- `priority` (string): Filter by priority (low, normal, high, urgent)
- `search` (string): Search in title and description
- `include_messages` (bool): Include latest messages
- `include_archived` (bool): Include archived conversations

### Create Conversation

**POST** `/conversations/`

**Request Body:**

```json
{
  "title": "Code Review Session",
  "description": "Reviewing Python microservice code",
  "project_id": "project_123",
  "session_id": "session_456",
  "priority": "normal",
  "conversation_metadata": {
    "type": "code_review",
    "language": "python",
    "estimated_duration": "30m"
  },
  "settings": {
    "max_messages": 100,
    "auto_archive_after_days": 30
  },
  "context_ids": ["context_789"],
  "active_prompt_template_id": "template_012"
}
```

### Get Conversation

**GET** `/conversations/{conversation_id}`

**Query Parameters:**

- `include_messages` (bool): Include all messages
- `include_stats` (bool): Include conversation statistics

### Update Conversation

**PUT** `/conversations/{conversation_id}`

**Request Body:**

```json
{
  "title": "Updated Code Review Session",
  "status": "completed",
  "priority": "high"
}
```

### Delete Conversation

**DELETE** `/conversations/{conversation_id}`

**Query Parameters:**

- `force` (bool): Force delete even if conversation is active

### List Messages

**GET** `/conversations/{conversation_id}/messages`

**Query Parameters:**

- `page` (int): Page number
- `size` (int): Page size (max: 200)
- `role` (string): Filter by message role (user, assistant, system, function, tool)
- `message_type` (string): Filter by message type (text, code, image, file, command, error, system_notification)
- `search` (string): Search in message content

### Create Message

**POST** `/conversations/{conversation_id}/messages`

**Request Body:**

```json
{
  "role": "user",
  "content": "Please review this Python function for potential improvements.",
  "message_type": "text",
  "message_metadata": {
    "intent": "code_review_request",
    "confidence": 0.95
  },
  "context_ids": ["context_code_snippet"],
  "input_tokens": 50
}
```

**Response:**

```json
{
  "id": "msg_123",
  "conversation_id": "conv_456",
  "sequence_number": 1,
  "role": "user",
  "message_type": "text",
  "content": "Please review this Python function...",
  "message_metadata": {
    "intent": "code_review_request",
    "confidence": 0.95
  },
  "context_ids": ["context_code_snippet"],
  "input_tokens": 50,
  "output_tokens": null,
  "total_tokens": 50,
  "processing_time_ms": 0.0,
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z"
}
```

### Get Message

**GET** `/conversations/{conversation_id}/messages/{message_id}`

### Get Conversation Statistics

**GET** `/conversations/{conversation_id}/stats`

**Response:**

```json
{
  "conversation_id": "conv_456",
  "message_count": 24,
  "total_tokens": 5420,
  "average_tokens_per_message": 225.8,
  "duration_minutes": 45.5,
  "user_messages": 12,
  "assistant_messages": 11,
  "system_messages": 1,
  "message_types": {
    "text": 20,
    "code": 3,
    "system_notification": 1
  },
  "token_usage_by_role": {
    "user": 1200,
    "assistant": 4000,
    "system": 220
  },
  "activity_timeline": [
    {
      "sequence": 1,
      "timestamp": "2024-01-01T12:00:00Z",
      "role": "user",
      "type": "text",
      "tokens": 50
    }
  ]
}
```

### Search Conversations

**POST** `/conversations/search`

**Request Body:**

```json
{
  "query": "code review",
  "project_ids": ["project_123"],
  "statuses": ["active", "completed"],
  "priorities": ["normal", "high"],
  "date_from": "2024-01-01T00:00:00Z",
  "min_messages": 5,
  "include_archived": false
}
```

---

## 5. Admin API

### Get System Statistics

**GET** `/admin/stats`

**Query Parameters:**

- `include_detailed` (bool): Include detailed metrics

**Response:**

```json
{
  "system_info": {
    "platform": "Linux",
    "architecture": "x86_64",
    "processor": 8,
    "boot_time": "2024-01-01T00:00:00Z",
    "python_version": "3.11.0"
  },
  "database_stats": {
    "connections": 15,
    "size_bytes": 104857600,
    "table_statistics": [
      {
        "schemaname": "public",
        "tablename": "conversations",
        "n_tup_ins": 1500,
        "n_tup_upd": 300,
        "n_tup_del": 50
      }
    ]
  },
  "cache_stats": {
    "status": "available",
    "memory_usage": "512MB",
    "hit_rate": "85%",
    "keys_count": 15420
  },
  "application_stats": {
    "total_prompts": 45,
    "active_prompts": 32,
    "total_contexts": 156,
    "total_conversations": 89,
    "active_conversations": 12
  },
  "performance_metrics": {
    "response_time_ms": 2.5,
    "uptime_seconds": 86400
  },
  "resource_usage": {
    "cpu_percent": 25.4,
    "memory_total": 8589934592,
    "memory_available": 4294967296,
    "memory_percent": 50.0,
    "disk_total": 107374182400,
    "disk_used": 32212254720,
    "disk_percent": 30.0
  },
  "uptime_seconds": 86400,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Get Detailed Health Check

**GET** `/admin/health/detailed`

**Query Parameters:**

- `check_external` (bool): Check external dependencies

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "checks": {
    "database": {
      "status": "healthy",
      "response_time_ms": 2.1,
      "details": "Database connection successful"
    },
    "memory": {
      "status": "healthy",
      "usage_percent": 75.2,
      "available_gb": 2.0
    },
    "disk": {
      "status": "warning",
      "usage_percent": 85.5,
      "available_gb": 15.2
    }
  }
}
```

### Database Maintenance

**POST** `/admin/database/maintenance`

**Request Body:**

```json
{
  "operation": "analyze",
  "tables": ["conversations", "conversation_messages"],
  "dry_run": true,
  "force": false
}
```

### Cache Management

**POST** `/admin/cache/management`

**Request Body:**

```json
{
  "operation": "clear_pattern",
  "pattern": "pcs:context:*",
  "keys": null
}
```

### List Users

**GET** `/admin/users`

**Query Parameters:**

- `page` (int): Page number
- `size` (int): Page size
- `search` (string): Search users
- `active_only` (bool): Show only active users

### User Management

**POST** `/admin/users/manage`

**Request Body:**

```json
{
  "action": "reset_password",
  "user_ids": ["user_123", "user_456"],
  "parameters": {}
}
```

### Get System Configuration

**GET** `/admin/config`

**Query Parameters:**

- `section` (string): Configuration section

### Get System Logs

**GET** `/admin/logs`

**Query Parameters:**

- `level` (string): Log level filter (INFO, WARNING, ERROR, ALL)
- `lines` (int): Number of log lines (max: 1000)
- `search` (string): Search in log content

---

## Error Responses

All endpoints return consistent error responses:

```json
{
  "error": "ValidationError",
  "message": "Request validation failed",
  "details": {
    "field": ["Field is required"]
  }
}
```

### Common HTTP Status Codes

- `200 OK` - Successful request
- `201 Created` - Resource created successfully
- `204 No Content` - Successful deletion
- `400 Bad Request` - Invalid request data
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `409 Conflict` - Resource conflict (duplicate)
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error

## Rate Limiting

API requests are rate limited:

- **Standard endpoints**: 100 requests/minute
- **Admin endpoints**: 50 requests/minute
- **Generation endpoints**: 20 requests/minute

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## Pagination

List endpoints support pagination with consistent parameters:

**Query Parameters:**

- `page` (int): Page number (starts from 1)
- `size` (int): Items per page (max varies by endpoint)

**Response Format:**

```json
{
  "items": [...],
  "total": 100,
  "page": 1,
  "size": 20,
  "pages": 5
}
```

## OpenAPI Specification

The complete OpenAPI specification is available at:

- **JSON**: `http://localhost:8000/openapi.json`
- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## SDK Examples

### Python SDK Example

```python
import requests

class PCSClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def create_prompt_template(self, name, description, category=None):
        data = {
            "name": name,
            "description": description,
            "category": category
        }
        response = requests.post(
            f"{self.base_url}/prompts/",
            json=data,
            headers=self.headers
        )
        return response.json()

    def generate_prompt(self, template_name, context_data, variables):
        data = {
            "template_name": template_name,
            "context_data": context_data,
            "variables": variables
        }
        response = requests.post(
            f"{self.base_url}/prompts/generate",
            json=data,
            headers=self.headers
        )
        return response.json()

# Usage
client = PCSClient("http://localhost:8000/api/v1", "your_token")
template = client.create_prompt_template(
    name="test-template",
    description="A test template"
)
```

### JavaScript SDK Example

```javascript
class PCSClient {
  constructor(baseUrl, token) {
    this.baseUrl = baseUrl;
    this.headers = {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    };
  }

  async createConversation(title, description, projectId) {
    const response = await fetch(`${this.baseUrl}/conversations/`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify({
        title,
        description,
        project_id: projectId,
      }),
    });
    return response.json();
  }

  async addMessage(conversationId, role, content) {
    const response = await fetch(
      `${this.baseUrl}/conversations/${conversationId}/messages`,
      {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify({
          role,
          content,
          message_type: "text",
        }),
      }
    );
    return response.json();
  }
}

// Usage
const client = new PCSClient("http://localhost:8000/api/v1", "your_token");
const conversation = await client.createConversation(
  "My Conversation",
  "A test conversation",
  "project_123"
);
```

## Best Practices

### 1. Authentication

- Store JWT tokens securely
- Implement token refresh logic
- Use HTTPS in production

### 2. Error Handling

- Always check response status codes
- Parse error details for user feedback
- Implement retry logic for transient errors

### 3. Performance

- Use pagination for large datasets
- Implement caching where appropriate
- Monitor rate limits

### 4. Data Validation

- Validate data client-side before sending
- Handle validation errors gracefully
- Use the provided schemas for reference

### 5. Context Management

- Design context hierarchies thoughtfully
- Use appropriate scopes for data isolation
- Clean up unused contexts regularly

---

This documentation covers the complete PCS API. For additional support, please refer to the OpenAPI specification or contact the development team.
