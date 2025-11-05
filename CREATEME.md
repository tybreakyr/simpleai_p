# Application Recreation Guide

## Overview

Create a lightweight library for interacting with Large Language Model (LLM) providers. The library should provide a unified abstraction layer that allows applications to work with multiple LLM providers through a single interface, regardless of the underlying provider implementation.

## Core Architecture

### 1. Provider Interface

Design a provider interface that all LLM implementations must conform to. The interface should include:

- **Chat Method**: Accepts a chat request and returns a response. The request should include:
  - System prompt (optional instruction for the LLM)
  - Messages array (conversation history with role and content)
  - Optional structured output type (for parsing JSON responses into typed objects)
  
- **ListModels Method**: Returns available models for the provider

- **Name Method**: Returns the provider's identifier string

- **IsAvailable Method**: Checks if the provider is currently reachable and operational

- **SupportedFeatures Method**: Returns a feature descriptor indicating capabilities such as:
  - Structured output support
  - Streaming support
  - Vision/image analysis support
  - Maximum context window size
  - Supported message roles
  - Function calling support
  - Temperature parameter support
  - Top-p parameter support

### 2. Data Structures

Implement the following core data structures:

**Message**: Represents a single message in a conversation
- Role (e.g., "user", "assistant", "system")
- Content (the message text)

**SystemPrompt**: Represents system-level instructions
- Content (the system instruction text)

**ChatRequest**: The input structure for chat operations
- SystemPrompt (system-level instruction)
- Messages (array of conversation messages)
- Optional structured output type reference (for type-safe JSON parsing)

**ChatResponse**: The output structure from chat operations
- Message (the raw text response)
- Optional structured data (parsed JSON if structured output was requested)

**Model**: Represents an available LLM model
- Name (model identifier)

**ProviderFeatures**: Describes provider capabilities (see SupportedFeatures method above)

**ProviderConfig**: Configuration for a provider instance
- Host/endpoint URL
- API key (if required)
- Default model name
- Timeout in seconds
- Retry attempts count
- Rate limit (optional, requests per minute)
- Extra settings map (provider-specific configuration)

**FactoryConfig**: Configuration for the provider factory
- Default provider name
- Provider configurations map (keyed by provider name)
- Model preferences map (optional, task-specific model mappings)
- Fallback providers array (optional, ordered list for failover)

### 3. Factory Pattern

Implement a factory that manages provider creation and lifecycle:

**Factory Interface**:
- Create provider from configuration
- Create provider from factory's stored configuration
- List registered providers
- List available models for a provider
- Get default provider instance
- Set/get default provider name
- Check if provider is available
- Get provider features
- Load factory configuration
- Get current factory configuration

**Factory Implementation**:
- Provider registry: Maintain a registry of provider constructors (factory functions)
- Provider caching: Cache created provider instances for reuse
- Model caching: Cache model lists to avoid repeated API calls
- Thread-safe operations: Ensure concurrent access safety
- Configuration validation: Validate configuration before use

**Provider Registration**:
- Allow runtime registration of provider implementations
- Constructor pattern: Providers are created via factory functions that accept configuration maps

### 4. Error Handling System

Implement comprehensive error handling:

**Error Types**:
- Connection failed (network/service unavailable)
- Timeout (request exceeded time limit)
- Invalid response (malformed API response)
- JSON parse failed (structured output parsing failure)
- Model not available (requested model doesn't exist)
- Rate limit exceeded (API rate limiting)
- Invalid configuration (configuration errors)
- Operation failed (general failure)

**Error Structure**:
Each error should include:
- Error type/category
- Human-readable message
- Retryable flag (indicates if operation can be retried)
- Operation name (which operation failed)
- Retry count (how many attempts were made)
- Timestamp of last attempt
- Underlying cause (nested error if applicable)

**Error Classification**:
Implement error classification logic that:
- Analyzes error messages/status codes
- Determines if errors are retryable (network issues) vs non-retryable (auth failures)
- Provides helper functions to check if an error is retryable

**Validation Errors**:
Separate validation error type for configuration/input validation issues with field-level details.

### 5. Retry Mechanism

Implement configurable retry logic with exponential backoff:

**Retry Configuration**:
- Maximum retries (default: 3)
- Base delay (default: 2 seconds)
- Maximum delay cap (default: 30 seconds)
- Backoff factor (default: 2.0)

**Retry Behavior**:
- Only retry on retryable errors
- Exponential backoff: delay = base_delay × (backoff_factor ^ attempt_number)
- Cap delays at maximum delay
- Track retry attempts in errors
- Continue retrying until max retries or non-retryable error occurs

**Integration**:
- Retry logic should be integrated into provider Chat operations
- Each provider should use the shared retry configuration
- Retry should handle both API call failures and JSON parsing failures

### 6. Structured Output / JSON Extraction

Implement sophisticated JSON extraction from LLM responses:

**Purpose**: 
LLMs often return JSON wrapped in markdown, explanatory text, or with formatting issues. The system should robustly extract and parse JSON even when:
- Wrapped in markdown code blocks (```json ... ```)
- Prefixed/suffixed with explanatory text
- Contains unescaped newlines or other formatting issues
- Mixed with other content

**Extraction Strategies** (in order of attempt):

1. **Direct Validation**: Try parsing the response as-is after trimming whitespace

2. **JSON Repair**: Attempt to repair common JSON issues:
   - Fix unescaped newlines in string values (replace literal `\n` with `\\n`)
   - Fix unescaped carriage returns and tabs
   - Preserve escape sequences correctly
   - Track string boundaries to avoid fixing escaped characters

3. **Markdown Extraction**: 
   - Look for markdown code blocks with `json` language identifier
   - Extract content between ```json and ```
   - Fall back to generic code blocks (``` ... ```)

4. **Brace Counting**:
   - Use brace/brace matching to find JSON object boundaries
   - Track string state to avoid counting braces inside strings
   - Handle escape sequences correctly
   - Find the first complete JSON object/array

5. **Aggressive Cleanup**:
   - Remove common prefixes ("Here's the JSON:", "JSON:", etc.)
   - Remove common suffixes
   - Try brace counting again after cleanup

**Parsing Flow**:
- When structured output type is requested, extract JSON using above strategies
- Parse extracted JSON into the requested type
- If extraction or parsing fails, retry the request (if retryable)
- Return both raw message and parsed structured data in response
- If parsing fails after all retries, return raw message with null structured data

### 7. Provider Implementation Pattern

Each provider implementation should:

**Constructor Function**:
- Accept a configuration map (key-value pairs)
- Extract and validate required configuration (host, API key if needed, etc.)
- Set defaults for optional configuration (timeout, retry attempts, default model)
- Create and configure the provider-specific API client
- Initialize retry configuration
- Return a provider instance implementing the interface

**Chat Implementation**:
- Convert internal message format to provider's API format
- Handle system prompts appropriately (provider-specific)
- Execute chat request with timeout
- Handle streaming if provider supports it (accumulate response)
- Extract text response from provider's response format
- If structured output requested, attempt JSON extraction and parsing
- Classify errors and determine retryability
- Use retry mechanism with exponential backoff

**Error Classification**:
- Analyze provider-specific error messages/status codes
- Map to standard error types
- Determine retryability (network errors = retryable, auth errors = not retryable)

**Availability Check**:
- Implement lightweight health check (e.g., list models or simple test request)
- Use short timeout (5 seconds)
- Return boolean indicating availability

**Feature Detection**:
- Return accurate feature capabilities based on provider's actual capabilities
- Include realistic context window sizes
- List supported roles based on provider's API

### 8. Configuration System

**Configuration Loading**:
- Accept configuration via structured config object
- Validate configuration before use:
  - Default provider must be specified
  - At least one provider must be configured
  - Default provider must exist in provider configurations
  - Each provider must have required fields (default model, timeout)
  - Timeout must be positive
  - Retry attempts must be non-negative

**Configuration Migration** (optional):
- Support migration from legacy config formats
- Provide migration utilities if needed

### 9. Caching Strategy

**Provider Caching**:
- Cache provider instances after creation
- Clear cache when configuration changes
- Allow cache clearing for testing/refresh

**Model Caching**:
- Cache model lists per provider
- Avoid repeated API calls for model listing
- Clear cache on configuration changes
- Provide method to manually clear cache

### 10. Thread Safety

Ensure thread-safe operations:
- Use appropriate synchronization primitives for concurrent access
- Protect shared state (registry, cache, configuration)
- Use read/write locks where appropriate (read-heavy operations)
- Ensure provider creation is thread-safe

## Implementation Guidelines

### Language Agnostic Patterns

1. **Interface/Protocol Definition**: Define the provider interface clearly with all required methods

2. **Factory Pattern**: Use factory pattern for provider creation and management

3. **Configuration Object**: Use structured configuration objects rather than scattered parameters

4. **Error Wrapping**: Wrap errors with context (operation, retry count, etc.) while preserving original error

5. **Type Safety**: Use generics or type parameters where supported for structured output parsing

6. **Retry Abstraction**: Make retry logic reusable across providers

7. **JSON Extraction**: Implement as a standalone utility function that can be tested independently

### Testing Considerations

- Unit tests for JSON extraction logic with various malformed inputs
- Unit tests for error classification
- Unit tests for retry logic and backoff calculation
- Integration tests for provider implementations (if possible)
- Mock provider implementations for testing factory behavior

### Documentation

- Document the provider interface clearly
- Provide examples of creating custom providers
- Document configuration options
- Document error handling patterns
- Provide usage examples for structured output

## Key Design Principles

1. **Abstraction**: Hide provider-specific details behind a common interface

2. **Flexibility**: Support multiple providers with different capabilities

3. **Robustness**: Handle errors gracefully with retries and clear error messages

4. **Type Safety**: Provide structured output parsing when possible

5. **Performance**: Cache providers and models to reduce API calls

6. **Extensibility**: Make it easy to add new providers via registration

7. **Configuration**: Support flexible configuration for different deployment scenarios

