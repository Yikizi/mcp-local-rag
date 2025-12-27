// RAGServer implementation with MCP tools

import { randomUUID } from 'node:crypto'
import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js'
import { DocumentChunker } from '../chunker/index.js'
import { Embedder } from '../embedder/index.js'
import { DocumentParser } from '../parser/index.js'
import { type GroupingMode, type VectorChunk, VectorStore } from '../vectordb/index.js'

// ============================================
// Type Definitions
// ============================================

/**
 * RAGServer configuration
 */
export interface RAGServerConfig {
  /** LanceDB database path */
  dbPath: string
  /** Transformers.js model path */
  modelName: string
  /** Model cache directory */
  cacheDir: string
  /** Document base directory */
  baseDir: string
  /** Maximum file size (100MB) */
  maxFileSize: number
  /** Chunk size */
  chunkSize: number
  /** Chunk overlap */
  chunkOverlap: number
  /** Maximum distance threshold for quality filtering (optional) */
  maxDistance?: number
  /** Grouping mode for quality filtering (optional) */
  grouping?: GroupingMode
  /** Hybrid search weight for BM25 (0.0 = vector only, 1.0 = BM25 only, default 0.6) */
  hybridWeight?: number
}

/**
 * query_documents tool input
 */
export interface QueryDocumentsInput {
  /** Natural language query */
  query: string
  /** Number of results to retrieve (default 10) */
  limit?: number
  /** Filter by entry type */
  type?: 'all' | 'file' | 'memory'
  /** Filter by tags */
  tags?: string[]
  /** Filter by project */
  project?: string
  /** Minimum similarity score threshold */
  minScore?: number
}

/**
 * ingest_file tool input
 */
export interface IngestFileInput {
  /** File path */
  filePath: string
  /** Tags for categorization */
  tags?: string[]
  /** Project identifier */
  project?: string
  /** If true, memory is not associated with any project */
  global?: boolean
}

/**
 * delete_file tool input
 */
export interface DeleteFileInput {
  /** File path */
  filePath: string
}

/**
 * memorize_text tool input
 */
export interface MemorizeTextInput {
  /** Text content to memorize */
  text: string
  /** Optional label/identifier for snippet */
  label?: string
  /** Optional language hint (for code snippets) */
  language?: string
  /** Tags for categorization */
  tags?: string[]
  /** Type of memory */
  type?: 'memory' | 'lesson' | 'note'
  /** Time-to-live */
  ttl?: string
  /** Project identifier */
  project?: string
  /** If true, memory is not associated with any project */
  global?: boolean
}

/**
 * list_files tool input
 */
export interface ListFilesInput {
  /** Filter by entry type */
  type?: 'all' | 'file' | 'memory'
  /** Filter by tags */
  tags?: string[]
  /** Filter by project */
  project?: string
  /** Filter by label/filename substring */
  search?: string
  /** Maximum entries to return */
  limit?: number
}

/**
 * update_memory tool input
 */
export interface UpdateMemoryInput {
  /** Label of memory to update */
  label: string
  /** How to handle new content */
  mode?: 'replace' | 'append' | 'prepend'
  /** New text content */
  text?: string
  /** New tags (replaces existing) */
  tags?: string[]
  /** Tags to add to existing */
  addTags?: string[]
  /** Tags to remove from existing */
  removeTags?: string[]
}

/**
 * ingest_file tool output
 */
export interface IngestResult {
  /** File path */
  filePath: string
  /** Chunk count */
  chunkCount: number
  /** Timestamp */
  timestamp: string
}

/**
 * query_documents tool output
 */
export interface QueryResult {
  /** File path */
  filePath: string
  /** Chunk index */
  chunkIndex: number
  /** Text */
  text: string
  /** Similarity score */
  score: number
}

// ============================================
// RAGServer Class
// ============================================

/**
 * RAG server compliant with MCP Protocol
 *
 * Responsibilities:
 * - MCP tool integration (4 tools)
 * - Tool handler implementation
 * - Error handling
 * - Initialization (LanceDB, Transformers.js)
 */
// ============================================
// Helper Functions
// ============================================

/**
 * Parse TTL string and calculate expiration date
 *
 * @param ttl - TTL string ('1d', '7d', '30d', '1y', 'permanent')
 * @returns ISO 8601 timestamp or undefined for permanent
 */
function calculateExpiresAt(ttl?: string): string | undefined {
  if (!ttl || ttl === 'permanent') {
    return undefined
  }

  const now = new Date()
  const match = ttl.match(/^(\d+)([dhmy])$/)

  if (!match || !match[1] || !match[2]) {
    throw new Error(`Invalid TTL format: ${ttl}. Use format like "1d", "7d", "30d", "1y"`)
  }

  const value = Number.parseInt(match[1], 10)
  const unit = match[2]

  switch (unit) {
    case 'd': // days
      now.setDate(now.getDate() + value)
      break
    case 'h': // hours
      now.setHours(now.getHours() + value)
      break
    case 'm': // months
      now.setMonth(now.getMonth() + value)
      break
    case 'y': // years
      now.setFullYear(now.getFullYear() + value)
      break
    default:
      throw new Error(
        `Invalid TTL unit: ${unit}. Use 'd' (days), 'h' (hours), 'm' (months), 'y' (years)`
      )
  }

  return now.toISOString()
}

// Valid memory types
const VALID_MEMORY_TYPES = ['memory', 'lesson', 'note'] as const
type ValidMemoryType = (typeof VALID_MEMORY_TYPES)[number]

// Valid update modes
const VALID_UPDATE_MODES = ['replace', 'append', 'prepend'] as const
type ValidUpdateMode = (typeof VALID_UPDATE_MODES)[number]

// Valid type filters
const VALID_TYPE_FILTERS = ['all', 'file', 'memory'] as const
type ValidTypeFilter = (typeof VALID_TYPE_FILTERS)[number]

/**
 * Validate tags parameter
 *
 * @param tags - Tags to validate
 * @throws Error if tags are invalid
 */
function validateTags(tags: unknown): string[] {
  if (tags === undefined || tags === null) {
    return []
  }

  if (!Array.isArray(tags)) {
    throw new Error('Tags must be an array')
  }

  for (const tag of tags) {
    if (typeof tag !== 'string') {
      throw new Error('All tags must be strings')
    }
    if (tag.trim().length === 0) {
      throw new Error('Tags cannot be empty strings')
    }
  }

  return tags.map((t) => t.trim())
}

/**
 * Validate memory type parameter
 *
 * @param type - Memory type to validate
 * @throws Error if type is invalid
 */
function validateMemoryType(type: unknown): ValidMemoryType | undefined {
  if (type === undefined || type === null) {
    return undefined
  }

  if (typeof type !== 'string') {
    throw new Error(`Invalid memory type. Must be one of: ${VALID_MEMORY_TYPES.join(', ')}`)
  }

  if (!VALID_MEMORY_TYPES.includes(type as ValidMemoryType)) {
    throw new Error(
      `Invalid memory type: "${type}". Must be one of: ${VALID_MEMORY_TYPES.join(', ')}`
    )
  }

  return type as ValidMemoryType
}

/**
 * Validate update mode parameter
 *
 * @param mode - Update mode to validate
 * @throws Error if mode is invalid
 */
function validateUpdateMode(mode: unknown): ValidUpdateMode | undefined {
  if (mode === undefined || mode === null) {
    return undefined
  }

  if (typeof mode !== 'string') {
    throw new Error(`Invalid update mode. Must be one of: ${VALID_UPDATE_MODES.join(', ')}`)
  }

  if (!VALID_UPDATE_MODES.includes(mode as ValidUpdateMode)) {
    throw new Error(
      `Invalid update mode: "${mode}". Must be one of: ${VALID_UPDATE_MODES.join(', ')}`
    )
  }

  return mode as ValidUpdateMode
}

/**
 * Validate type filter parameter
 *
 * @param type - Type filter to validate
 * @throws Error if type filter is invalid
 */
function validateTypeFilter(type: unknown): ValidTypeFilter | undefined {
  if (type === undefined || type === null) {
    return undefined
  }

  if (typeof type !== 'string') {
    throw new Error(`Invalid type filter. Must be one of: ${VALID_TYPE_FILTERS.join(', ')}`)
  }

  if (!VALID_TYPE_FILTERS.includes(type as ValidTypeFilter)) {
    throw new Error(
      `Invalid type filter: "${type}". Must be one of: ${VALID_TYPE_FILTERS.join(', ')}`
    )
  }

  return type as ValidTypeFilter
}

/**
 * Validate minScore parameter
 *
 * @param minScore - Minimum score to validate
 * @throws Error if minScore is invalid
 */
function validateMinScore(minScore: unknown): number | undefined {
  if (minScore === undefined || minScore === null) {
    return undefined
  }

  if (typeof minScore !== 'number') {
    throw new Error('minScore must be a number')
  }

  if (minScore < 0) {
    throw new Error('minScore cannot be negative')
  }

  if (minScore > 2) {
    throw new Error('minScore must be <= 2 (LanceDB uses L2 distance, typical values are 0-2)')
  }

  return minScore
}

// ============================================
// RAGServer Class
// ============================================

export class RAGServer {
  private readonly server: Server
  private readonly vectorStore: VectorStore
  private readonly embedder: Embedder
  private readonly chunker: DocumentChunker
  private readonly parser: DocumentParser

  constructor(config: RAGServerConfig) {
    this.server = new Server(
      { name: 'rag-mcp-server', version: '1.0.0' },
      { capabilities: { tools: {} } }
    )

    // Component initialization
    // Only pass quality filter settings if they are defined
    const vectorStoreConfig: ConstructorParameters<typeof VectorStore>[0] = {
      dbPath: config.dbPath,
      tableName: 'chunks',
    }
    if (config.maxDistance !== undefined) {
      vectorStoreConfig.maxDistance = config.maxDistance
    }
    if (config.grouping !== undefined) {
      vectorStoreConfig.grouping = config.grouping
    }
    if (config.hybridWeight !== undefined) {
      vectorStoreConfig.hybridWeight = config.hybridWeight
    }
    this.vectorStore = new VectorStore(vectorStoreConfig)
    this.embedder = new Embedder({
      modelPath: config.modelName,
      batchSize: 8,
      cacheDir: config.cacheDir,
    })
    this.chunker = new DocumentChunker({
      chunkSize: config.chunkSize,
      chunkOverlap: config.chunkOverlap,
    })
    this.parser = new DocumentParser({
      baseDir: config.baseDir,
      maxFileSize: config.maxFileSize,
    })

    this.setupHandlers()
  }

  /**
   * Set up MCP handlers
   */
  private setupHandlers(): void {
    // Tool list
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'query_documents',
          description:
            'Search through previously ingested documents (PDF, DOCX, TXT, MD, code files, text snippets) using semantic search. Returns relevant passages from documents in the BASE_DIR. Documents must be ingested first using ingest_file or memorize_text. Supports filtering by type, tags, project, and minimum score.',
          inputSchema: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description:
                  'Natural language search query (e.g., "transformer architecture", "API documentation")',
              },
              limit: {
                type: 'number',
                description:
                  'Maximum number of results to return (default: 10). Recommended: 5 for precision, 10 for balance, 20 for broad exploration.',
              },
              type: {
                type: 'string',
                enum: ['all', 'file', 'memory'],
                description: 'Filter by entry type: "all" (default), "file", or "memory"',
              },
              tags: {
                type: 'array',
                items: { type: 'string' },
                description: 'Filter by tags (AND logic - results must have all specified tags)',
              },
              project: {
                type: 'string',
                description: 'Filter by project identifier',
              },
              minScore: {
                type: 'number',
                description: 'Minimum similarity score threshold (0-1)',
              },
            },
            required: ['query'],
          },
        },
        {
          name: 'ingest_file',
          description:
            'Ingest a document or code file (PDF, DOCX, TXT, MD, TypeScript, JavaScript, Python, Java, Go, Rust, C/C++, Ruby, PHP, C#, Shell, SQL) into the vector database for semantic search. File path must be an absolute path. Supports re-ingestion to update existing documents. Can be tagged and associated with a project.',
          inputSchema: {
            type: 'object',
            properties: {
              filePath: {
                type: 'string',
                description:
                  'Absolute path to the file to ingest. Example: "/Users/user/documents/manual.pdf"',
              },
              tags: {
                type: 'array',
                items: { type: 'string' },
                description: 'Tags for categorization (e.g., ["project-name", "documentation"])',
              },
              project: {
                type: 'string',
                description:
                  'Project identifier (auto-detected from cwd if not specified and global is false)',
              },
              global: {
                type: 'boolean',
                description: 'If true, file is not associated with any project (default: false)',
              },
            },
            required: ['filePath'],
          },
        },
        {
          name: 'delete_file',
          description:
            'Delete a previously ingested file from the vector database. Removes all chunks and embeddings associated with the specified file. File path must be an absolute path. This operation is idempotent - deleting a non-existent file completes without error.',
          inputSchema: {
            type: 'object',
            properties: {
              filePath: {
                type: 'string',
                description:
                  'Absolute path to the file to delete from the database. Example: "/Users/user/documents/manual.pdf"',
              },
            },
            required: ['filePath'],
          },
        },
        {
          name: 'memorize_text',
          description:
            'Store text snippets directly into the vector database for semantic search. Useful for memorizing code snippets, notes, lessons learned, or any text without file I/O. Stored with label identifier (default: auto-generated timestamp). Supports tags, types (memory/lesson/note), TTL expiration, and project association. Can be deleted using delete_file with path "memory://<label>".',
          inputSchema: {
            type: 'object',
            properties: {
              text: {
                type: 'string',
                description: 'Text content to memorize (e.g., code snippet, note, definition)',
              },
              label: {
                type: 'string',
                description: 'Optional label for snippet (default: "snippet-<timestamp>")',
              },
              language: {
                type: 'string',
                description:
                  'Optional language hint for code snippets (e.g., "python", "typescript")',
              },
              tags: {
                type: 'array',
                items: { type: 'string' },
                description:
                  'Tags for categorization (e.g., ["experiment", "workaround", "python"])',
              },
              type: {
                type: 'string',
                enum: ['memory', 'lesson', 'note'],
                description: 'Type of memory being stored: "memory" (default), "lesson", or "note"',
              },
              ttl: {
                type: 'string',
                description:
                  'Time-to-live: "1d", "7d", "30d", "1y", or "permanent" (default). Memories expire after this duration.',
              },
              project: {
                type: 'string',
                description:
                  'Project identifier (auto-detected from cwd if not specified and global is false)',
              },
              global: {
                type: 'boolean',
                description: 'If true, memory is not associated with any project (default: false)',
              },
            },
            required: ['text'],
          },
        },
        {
          name: 'list_files',
          description:
            'List all ingested files and memories in the vector database. Returns file paths, chunk counts, and metadata for each entry. Supports filtering by type, tags, project, and search string.',
          inputSchema: {
            type: 'object',
            properties: {
              type: {
                type: 'string',
                enum: ['all', 'file', 'memory'],
                description: 'Filter by entry type: "all" (default), "file", or "memory"',
              },
              tags: {
                type: 'array',
                items: { type: 'string' },
                description: 'Filter by tags (AND logic - entries must have all specified tags)',
              },
              project: {
                type: 'string',
                description: 'Filter by project identifier',
              },
              search: {
                type: 'string',
                description: 'Filter by label/filename substring match',
              },
              limit: {
                type: 'number',
                description: 'Maximum entries to return (default: 50)',
              },
            },
          },
        },
        {
          name: 'status',
          description:
            'Get system status including total documents, total chunks, database size, and configuration information.',
          inputSchema: { type: 'object', properties: {} },
        },
        {
          name: 'cleanup_expired',
          description:
            'Remove all expired memories from the database. Deletes entries where the TTL has elapsed. Returns the count of deleted entries.',
          inputSchema: { type: 'object', properties: {} },
        },
        {
          name: 'update_memory',
          description:
            "Update an existing memory's content or metadata. Allows replacing, appending, or prepending text, and managing tags. The memory is identified by its label.",
          inputSchema: {
            type: 'object',
            properties: {
              label: {
                type: 'string',
                description: 'Label of the memory to update (without "memory://" prefix)',
              },
              mode: {
                type: 'string',
                enum: ['replace', 'append', 'prepend'],
                description:
                  'How to handle the new text content: "replace" (default), "append", or "prepend"',
              },
              text: {
                type: 'string',
                description: 'New text content to apply based on mode',
              },
              tags: {
                type: 'array',
                items: { type: 'string' },
                description: 'New tags (replaces all existing tags)',
              },
              addTags: {
                type: 'array',
                items: { type: 'string' },
                description: 'Tags to add to existing tags',
              },
              removeTags: {
                type: 'array',
                items: { type: 'string' },
                description: 'Tags to remove from existing tags',
              },
            },
            required: ['label'],
          },
        },
      ],
    }))

    // Tool invocation
    this.server.setRequestHandler(
      CallToolRequestSchema,
      async (request: { params: { name: string; arguments?: unknown } }) => {
        switch (request.params.name) {
          case 'query_documents':
            return await this.handleQueryDocuments(
              request.params.arguments as unknown as QueryDocumentsInput
            )
          case 'ingest_file':
            return await this.handleIngestFile(
              request.params.arguments as unknown as IngestFileInput
            )
          case 'delete_file':
            return await this.handleDeleteFile(
              request.params.arguments as unknown as DeleteFileInput
            )
          case 'memorize_text':
            return await this.handleMemorizeText(
              request.params.arguments as unknown as MemorizeTextInput
            )
          case 'list_files':
            return await this.handleListFiles(request.params.arguments as unknown as ListFilesInput)
          case 'status':
            return await this.handleStatus()
          case 'cleanup_expired':
            return await this.handleCleanupExpired()
          case 'update_memory':
            return await this.handleUpdateMemory(
              request.params.arguments as unknown as UpdateMemoryInput
            )
          default:
            throw new Error(`Unknown tool: ${request.params.name}`)
        }
      }
    )
  }

  /**
   * Initialization
   */
  async initialize(): Promise<void> {
    await this.vectorStore.initialize()
    await this.chunker.initialize()
    console.error('RAGServer initialized')
  }

  /**
   * query_documents tool handler
   */
  async handleQueryDocuments(
    args: QueryDocumentsInput
  ): Promise<{ content: [{ type: 'text'; text: string }] }> {
    try {
      // Validate inputs
      const validatedType = validateTypeFilter(args.type)
      const validatedTags = validateTags(args.tags)
      const validatedMinScore = validateMinScore(args.minScore)

      // Generate query embedding
      const queryVector = await this.embedder.embed(args.query)

      // Hybrid search with filters - combines BM25 + vector search with metadata filtering
      const searchOptions: {
        queryText?: string
        limit?: number
        type?: 'all' | 'file' | 'memory'
        tags?: string[]
        project?: string
        minScore?: number
      } = {
        queryText: args.query, // Enable hybrid BM25 + vector search
        limit: args.limit || 10,
      }
      if (validatedType) searchOptions.type = validatedType
      if (validatedTags.length > 0) searchOptions.tags = validatedTags
      if (args.project) searchOptions.project = args.project
      if (validatedMinScore !== undefined) searchOptions.minScore = validatedMinScore

      const searchResults = await this.vectorStore.search(queryVector, searchOptions)

      // Format results
      const results: QueryResult[] = searchResults.map((result) => ({
        filePath: result.filePath,
        chunkIndex: result.chunkIndex,
        text: result.text,
        score: result.score,
      }))

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(results, null, 2),
          },
        ],
      }
    } catch (error) {
      console.error('Failed to query documents:', error)
      throw error
    }
  }

  /**
   * ingest_file tool handler (re-ingestion support, transaction processing, rollback capability)
   */
  async handleIngestFile(
    args: IngestFileInput
  ): Promise<{ content: [{ type: 'text'; text: string }] }> {
    let backup: VectorChunk[] | null = null

    try {
      // Parse file
      const parseResult = await this.parser.parseFile(args.filePath)
      const text = parseResult.text
      const language = parseResult.language

      // Split text into chunks
      const chunks = await this.chunker.chunkText(text)

      // Generate embeddings
      const embeddings = await this.embedder.embedBatch(chunks.map((chunk) => chunk.text))

      // Create backup (if existing data exists)
      try {
        const existingFiles = await this.vectorStore.listFiles()
        const existingFile = existingFiles.find((file) => file.filePath === args.filePath)
        if (existingFile && existingFile.chunkCount > 0) {
          // Backup existing data (retrieve via search)
          const queryVector = embeddings[0] || []
          if (queryVector.length > 0) {
            const allChunks = await this.vectorStore.search(queryVector, { limit: 20 }) // Retrieve max 20 items
            backup = allChunks
              .filter((chunk) => chunk.filePath === args.filePath)
              .map((chunk) => ({
                id: randomUUID(),
                filePath: chunk.filePath,
                chunkIndex: chunk.chunkIndex,
                text: chunk.text,
                vector: queryVector, // Use dummy vector since actual vector cannot be retrieved
                metadata: chunk.metadata,
                timestamp: new Date().toISOString(),
              }))
          }
          console.error(`Backup created: ${backup?.length || 0} chunks for ${args.filePath}`)
        }
      } catch (error) {
        // Backup creation failure is warning only (for new files)
        console.warn('Failed to create backup (new file?):', error)
      }

      // Delete existing data
      await this.vectorStore.deleteChunks(args.filePath)
      console.error(`Deleted existing chunks for: ${args.filePath}`)

      // Create vector chunks
      const timestamp = new Date().toISOString()
      const vectorChunks: VectorChunk[] = chunks.map((chunk, index) => {
        const embedding = embeddings[index]
        if (!embedding) {
          throw new Error(`Missing embedding for chunk ${index}`)
        }
        return {
          id: randomUUID(),
          filePath: args.filePath,
          chunkIndex: chunk.index,
          text: chunk.text,
          vector: embedding,
          metadata: {
            fileName: args.filePath.split('/').pop() || args.filePath,
            fileSize: text.length,
            fileType: args.filePath.split('.').pop() || '',
            language: language || null,
            memoryType: 'file',
            tags: args.tags || [],
            project: args.project && !args.global ? args.project : null,
            expiresAt: null,
            createdAt: timestamp,
            updatedAt: timestamp,
          },
          timestamp,
        }
      })

      // Insert vectors (transaction processing)
      try {
        await this.vectorStore.insertChunks(vectorChunks)
        console.error(`Inserted ${vectorChunks.length} chunks for: ${args.filePath}`)

        // Delete backup on success
        backup = null
      } catch (insertError) {
        // Rollback on error
        if (backup && backup.length > 0) {
          console.error('Ingestion failed, rolling back...', insertError)
          try {
            await this.vectorStore.insertChunks(backup)
            console.error(`Rollback completed: ${backup.length} chunks restored`)
          } catch (rollbackError) {
            console.error('Rollback failed:', rollbackError)
            throw new Error(
              `Failed to ingest file and rollback failed: ${(insertError as Error).message}`
            )
          }
        }
        throw insertError
      }

      // Result
      const result: IngestResult = {
        filePath: args.filePath,
        chunkCount: chunks.length,
        timestamp,
      }

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2),
          },
        ],
      }
    } catch (error) {
      // Error handling: suppress stack trace in production
      const errorMessage =
        process.env['NODE_ENV'] === 'production'
          ? (error as Error).message
          : (error as Error).stack || (error as Error).message

      console.error('Failed to ingest file:', errorMessage)

      throw new Error(`Failed to ingest file: ${errorMessage}`)
    }
  }

  /**
   * list_files tool handler
   */
  async handleListFiles(
    args?: ListFilesInput
  ): Promise<{ content: [{ type: 'text'; text: string }] }> {
    try {
      const filters: {
        type?: 'all' | 'file' | 'memory'
        tags?: string[]
        project?: string
        search?: string
        limit?: number
      } = {}

      if (args?.type) filters.type = args.type
      if (args?.tags) filters.tags = args.tags
      if (args?.project) filters.project = args.project
      if (args?.search) filters.search = args.search
      if (args?.limit) filters.limit = args.limit

      const files = await this.vectorStore.listFiles(
        Object.keys(filters).length > 0 ? filters : undefined
      )
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(files, null, 2),
          },
        ],
      }
    } catch (error) {
      console.error('Failed to list files:', error)
      throw error
    }
  }

  /**
   * status tool handler (Phase 1: basic implementation)
   */
  async handleStatus(): Promise<{ content: [{ type: 'text'; text: string }] }> {
    try {
      const status = await this.vectorStore.getStatus()
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(status, null, 2),
          },
        ],
      }
    } catch (error) {
      console.error('Failed to get status:', error)
      throw error
    }
  }

  /**
   * delete_file tool handler
   */
  async handleDeleteFile(
    args: DeleteFileInput
  ): Promise<{ content: [{ type: 'text'; text: string }] }> {
    try {
      // Validate and normalize file path (S-002 security requirement)
      this.parser.validateFilePath(args.filePath)

      // Delete chunks from vector database
      await this.vectorStore.deleteChunks(args.filePath)

      // Return success message
      const result = {
        filePath: args.filePath,
        deleted: true,
        timestamp: new Date().toISOString(),
      }

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2),
          },
        ],
      }
    } catch (error) {
      // Error handling: suppress stack trace in production
      const errorMessage =
        process.env['NODE_ENV'] === 'production'
          ? (error as Error).message
          : (error as Error).stack || (error as Error).message

      console.error('Failed to delete file:', errorMessage)

      throw new Error(`Failed to delete file: ${errorMessage}`)
    }
  }

  /**
   * memorize_text tool handler
   *
   * Stores text snippet directly without file I/O
   */
  async handleMemorizeText(
    args: MemorizeTextInput
  ): Promise<{ content: [{ type: 'text'; text: string }] }> {
    try {
      // Validate inputs
      const validatedTags = validateTags(args.tags)
      const validatedType = validateMemoryType(args.type)

      // Generate label and synthetic path
      const label = args.label || `snippet-${Date.now()}`
      const syntheticFilePath = `memory://${label}`

      // Chunk text
      const chunks = await this.chunker.chunkText(args.text)

      // Generate embeddings
      const embeddings = await this.embedder.embedBatch(chunks.map((chunk) => chunk.text))

      // Check for existing snippet (re-memorization support)
      try {
        const existingFiles = await this.vectorStore.listFiles()
        const existingFile = existingFiles.find((file) => file.filePath === syntheticFilePath)
        if (existingFile && existingFile.chunkCount > 0) {
          await this.vectorStore.deleteChunks(syntheticFilePath)
          console.log(`Deleted existing snippet: ${syntheticFilePath}`)
        }
      } catch (error) {
        console.warn('Failed to check for existing snippet:', error)
      }

      // Calculate expiration if TTL is specified
      const expiresAt = calculateExpiresAt(args.ttl)

      // Create vector chunks
      const timestamp = new Date().toISOString()
      const vectorChunks: VectorChunk[] = chunks.map((chunk, index) => {
        const embedding = embeddings[index]
        if (!embedding) {
          throw new Error(`Missing embedding for chunk ${index}`)
        }
        return {
          id: randomUUID(),
          filePath: syntheticFilePath,
          chunkIndex: chunk.index,
          text: chunk.text,
          vector: embedding,
          metadata: {
            fileName: label,
            fileSize: args.text.length,
            fileType: 'text-snippet',
            language: args.language || null,
            memoryType: validatedType || 'memory',
            tags: validatedTags,
            project: args.project && !args.global ? args.project : null,
            expiresAt: expiresAt || null,
            createdAt: timestamp,
            updatedAt: timestamp,
          },
          timestamp,
        }
      })

      // Insert into database
      await this.vectorStore.insertChunks(vectorChunks)
      console.log(`Inserted ${vectorChunks.length} chunks for snippet: ${syntheticFilePath}`)

      // Return result
      const result = {
        filePath: syntheticFilePath,
        label,
        chunkCount: chunks.length,
        timestamp,
        ...(expiresAt && { expiresAt }),
      }

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2),
          },
        ],
      }
    } catch (error) {
      const errorMessage =
        process.env['NODE_ENV'] === 'production'
          ? (error as Error).message
          : (error as Error).stack || (error as Error).message

      console.error('Failed to memorize text:', errorMessage)
      throw new Error(`Failed to memorize text: ${errorMessage}`)
    }
  }

  /**
   * cleanup_expired tool handler
   */
  async handleCleanupExpired(): Promise<{ content: [{ type: 'text'; text: string }] }> {
    try {
      const deletedCount = await this.vectorStore.cleanupExpired()
      const result = {
        deletedCount,
        timestamp: new Date().toISOString(),
      }

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2),
          },
        ],
      }
    } catch (error) {
      console.error('Failed to cleanup expired memories:', error)
      throw error
    }
  }

  /**
   * update_memory tool handler
   */
  async handleUpdateMemory(
    args: UpdateMemoryInput
  ): Promise<{ content: [{ type: 'text'; text: string }] }> {
    try {
      // Validate inputs
      const validatedMode = validateUpdateMode(args.mode)
      const validatedTags = args.tags !== undefined ? validateTags(args.tags) : undefined
      const validatedAddTags = args.addTags !== undefined ? validateTags(args.addTags) : undefined
      const validatedRemoveTags =
        args.removeTags !== undefined ? validateTags(args.removeTags) : undefined

      // Get existing memory
      const existingChunks = await this.vectorStore.getMemoryByLabel(args.label)

      if (existingChunks.length === 0) {
        throw new Error(`Memory not found: ${args.label}`)
      }

      // Reconstruct full text from chunks
      const sortedChunks = existingChunks.sort((a, b) => a.chunkIndex - b.chunkIndex)
      let fullText = sortedChunks.map((chunk) => chunk.text).join('\n')

      // Get existing metadata (we know sortedChunks has at least one element from the check above)
      const firstChunk = sortedChunks[0]
      if (!firstChunk) {
        throw new Error(`Memory not found: ${args.label}`)
      }
      const existingMetadata = firstChunk.metadata

      // Apply text changes based on mode
      if (args.text) {
        const mode = validatedMode || 'replace'
        switch (mode) {
          case 'replace':
            fullText = args.text
            break
          case 'append':
            fullText = `${fullText}\n${args.text}`
            break
          case 'prepend':
            fullText = `${args.text}\n${fullText}`
            break
        }
      }

      // Apply tag changes
      let newTags: string[] = Array.isArray(existingMetadata.tags) ? [...existingMetadata.tags] : []
      if (validatedTags !== undefined) {
        // Replace all tags
        newTags = validatedTags
      } else {
        // Add/remove tags incrementally
        if (validatedAddTags && validatedAddTags.length > 0) {
          newTags = [...new Set([...newTags, ...validatedAddTags])]
        }
        if (validatedRemoveTags && validatedRemoveTags.length > 0) {
          newTags = newTags.filter((tag) => !validatedRemoveTags.includes(tag))
        }
      }

      // Re-chunk and re-embed
      const chunks = await this.chunker.chunkText(fullText)
      const embeddings = await this.embedder.embedBatch(chunks.map((chunk) => chunk.text))

      // Update timestamp
      const timestamp = new Date().toISOString()

      // Create new vector chunks with updated metadata
      const syntheticFilePath = `memory://${args.label}`
      const vectorChunks: VectorChunk[] = chunks.map((chunk, index) => {
        const embedding = embeddings[index]
        if (!embedding) {
          throw new Error(`Missing embedding for chunk ${index}`)
        }
        return {
          id: randomUUID(),
          filePath: syntheticFilePath,
          chunkIndex: chunk.index,
          text: chunk.text,
          vector: embedding,
          metadata: {
            fileName: existingMetadata.fileName,
            fileSize: fullText.length,
            fileType: existingMetadata.fileType,
            language: existingMetadata.language || null,
            memoryType: existingMetadata.memoryType || 'memory',
            tags: newTags || [],
            project: existingMetadata.project || null,
            expiresAt: existingMetadata.expiresAt || null,
            createdAt: existingMetadata.createdAt || timestamp,
            updatedAt: timestamp,
          },
          timestamp,
        }
      })

      // Delete old chunks and insert new ones (transactional)
      await this.vectorStore.deleteChunks(syntheticFilePath)
      await this.vectorStore.insertChunks(vectorChunks)

      console.log(`Updated memory: ${args.label} (${vectorChunks.length} chunks)`)

      // Return result
      const result = {
        filePath: syntheticFilePath,
        label: args.label,
        chunkCount: chunks.length,
        timestamp,
        tags: newTags,
      }

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2),
          },
        ],
      }
    } catch (error) {
      const errorMessage =
        process.env['NODE_ENV'] === 'production'
          ? (error as Error).message
          : (error as Error).stack || (error as Error).message

      console.error('Failed to update memory:', errorMessage)
      throw new Error(`Failed to update memory: ${errorMessage}`)
    }
  }

  /**
   * Start the server
   */
  async run(): Promise<void> {
    const transport = new StdioServerTransport()
    await this.server.connect(transport)
    console.error('RAGServer running on stdio transport')
  }
}
