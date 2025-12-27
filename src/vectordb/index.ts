// VectorStore implementation with LanceDB integration

import { type Connection, Index, type Table, connect } from '@lancedb/lancedb'
import * as arrow from 'apache-arrow'

// ============================================
// Constants
// ============================================

/**
 * Standard deviation multiplier for detecting group boundaries.
 * A gap is considered a "boundary" if it exceeds mean + k*std.
 * Value of 1.5 means gaps > 1.5 standard deviations above mean are boundaries.
 */
const GROUPING_BOUNDARY_STD_MULTIPLIER = 1.5

/** Multiplier for candidate count in hybrid search (to allow reranking) */
const HYBRID_SEARCH_CANDIDATE_MULTIPLIER = 2

/** Maximum distance for dot product similarity (0 = identical, 2 = opposite) */
const DOT_PRODUCT_MAX_DISTANCE = 2

// ============================================
// Type Definitions
// ============================================

/**
 * Grouping mode for quality filtering
 * - 'similar': Only return the most similar group (stops at first distance jump)
 * - 'related': Include related groups (stops at second distance jump)
 */
export type GroupingMode = 'similar' | 'related'

/**
 * VectorStore configuration
 */
export interface VectorStoreConfig {
  /** LanceDB database path */
  dbPath: string
  /** Table name */
  tableName: string
  /** Maximum distance threshold for filtering results (optional) */
  maxDistance?: number
  /** Grouping mode for quality filtering (optional) */
  grouping?: GroupingMode
  /** Hybrid search weight for BM25 (0.0 = vector only, 1.0 = BM25 only, default 0.6) */
  hybridWeight?: number
}

/**
 * Document metadata
 */
export interface DocumentMetadata {
  /** File name */
  fileName: string
  /** File size in bytes */
  fileSize: number
  /** File type (extension) */
  fileType: string
  /** Programming language (optional, for code files) */
  language?: string | null | undefined
  /** Tags for categorization */
  tags?: string[]
  /** Associated project identifier */
  project?: string | null | undefined
  /** Memory type: 'file' | 'memory' | 'lesson' | 'note' */
  memoryType?: string | null | undefined
  /** Expiration timestamp (ISO 8601 format) */
  expiresAt?: string | null | undefined
  /** Creation timestamp (ISO 8601 format) */
  createdAt?: string | null | undefined
  /** Last update timestamp (ISO 8601 format) */
  updatedAt?: string | null | undefined
}

/**
 * Vector chunk
 */
export interface VectorChunk {
  /** Chunk ID (UUID) */
  id: string
  /** File path (absolute) */
  filePath: string
  /** Chunk index (zero-based) */
  chunkIndex: number
  /** Chunk text */
  text: string
  /** Embedding vector (dimension depends on model) */
  vector: number[]
  /** Metadata */
  metadata: DocumentMetadata
  /** Ingestion timestamp (ISO 8601 format) */
  timestamp: string
}

/**
 * Search result
 */
export interface SearchResult {
  /** File path */
  filePath: string
  /** Chunk index */
  chunkIndex: number
  /** Chunk text */
  text: string
  /** Distance score using dot product (0 = identical, 1 = orthogonal, 2 = opposite) */
  score: number
  /** Metadata */
  metadata: DocumentMetadata
}

// ============================================
// Error Classes
// ============================================

/**
 * Database error
 */
export class DatabaseError extends Error {
  constructor(
    message: string,
    public readonly cause?: Error
  ) {
    super(message)
    this.name = 'DatabaseError'
  }
}

// ============================================
// Helper Functions
// ============================================

/**
 * Validate file path to prevent SQL injection
 *
 * @param filePath - File path to validate
 * @throws Error if file path contains invalid characters
 */
function validateFilePath(filePath: string): void {
  if (!filePath || typeof filePath !== 'string') {
    throw new Error('Invalid file path: must be a non-empty string')
  }

  // For memory:// paths, validate label format (alphanumeric, hyphens, underscores, dots)
  if (filePath.startsWith('memory://')) {
    const label = filePath.substring(9) // Remove 'memory://' prefix
    if (!/^[\w.-]+$/.test(label)) {
      throw new Error(
        'Invalid memory label: must contain only alphanumeric characters, hyphens, underscores, and dots'
      )
    }
    return
  }

  // For regular file paths, ensure they are absolute paths
  // and don't contain SQL injection characters
  if (!filePath.startsWith('/')) {
    throw new Error('Invalid file path: must be an absolute path starting with /')
  }

  // Check for dangerous SQL characters that could be used for injection
  // Allow common path characters: alphanumeric, /, -, _, ., spaces, and common symbols
  // Disallow: ; ' " \ ` and control characters
  const dangerousCharsRegex = /[;'"\\`]/
  const hasControlChars = filePath.split('').some((char) => char.charCodeAt(0) < 32)
  if (dangerousCharsRegex.test(filePath) || hasControlChars) {
    throw new Error('Invalid file path: contains potentially dangerous characters')
  }
}

// ============================================
// VectorStore Class
// ============================================

/**
 * Vector storage class using LanceDB
 *
 * Responsibilities:
 * - LanceDB operations (insert, delete, search)
 * - Transaction handling (atomicity of delete→insert)
 * - Metadata management
 */
export class VectorStore {
  private db: Connection | null = null
  private table: Table | null = null
  private readonly config: VectorStoreConfig
  private ftsEnabled = false

  constructor(config: VectorStoreConfig) {
    this.config = config
  }

  /**
   * Get the schema for the chunks table
   */
  private getSchema(): arrow.Schema {
    return new arrow.Schema([
      new arrow.Field('id', new arrow.Utf8(), false),
      new arrow.Field('filePath', new arrow.Utf8(), false),
      new arrow.Field('chunkIndex', new arrow.Int32(), false),
      new arrow.Field('text', new arrow.Utf8(), false),
      new arrow.Field(
        'vector',
        new arrow.FixedSizeList(384, new arrow.Field('item', new arrow.Float32(), false)),
        false
      ),
      new arrow.Field(
        'metadata',
        new arrow.Struct([
          new arrow.Field('fileName', new arrow.Utf8(), false),
          new arrow.Field('fileSize', new arrow.Int32(), false),
          new arrow.Field('fileType', new arrow.Utf8(), false),
          new arrow.Field('language', new arrow.Utf8(), true),
          new arrow.Field('memoryType', new arrow.Utf8(), true),
          new arrow.Field(
            'tags',
            new arrow.List(new arrow.Field('item', new arrow.Utf8(), false)),
            true
          ),
          new arrow.Field('project', new arrow.Utf8(), true),
          new arrow.Field('expiresAt', new arrow.Utf8(), true),
          new arrow.Field('createdAt', new arrow.Utf8(), true),
          new arrow.Field('updatedAt', new arrow.Utf8(), true),
        ]),
        false
      ),
      new arrow.Field('timestamp', new arrow.Utf8(), false),
    ])
  }

  /**
   * Check if table schema needs migration
   * Returns true if migration is needed (missing new fields like createdAt, updatedAt)
   */
  private async needsMigration(table: Table): Promise<boolean> {
    try {
      const schema = await table.schema()
      const metadataField = schema.fields.find((f) => f.name === 'metadata')

      if (!metadataField) {
        return true // No metadata field at all - needs migration
      }

      // Check if metadata is a Struct type and has the new fields
      // Note: instanceof arrow.Struct doesn't work reliably, use typeId instead
      const metadataType = metadataField.type
      if (metadataType.typeId === arrow.Type.Struct && 'children' in metadataType) {
        const fieldNames = (metadataType as arrow.Struct).children.map((f) => f.name)
        // Check for the newer fields that may be missing in old schemas
        const hasCreatedAt = fieldNames.includes('createdAt')
        const hasUpdatedAt = fieldNames.includes('updatedAt')
        const hasMemoryType = fieldNames.includes('memoryType')
        const hasTags = fieldNames.includes('tags')

        if (!hasCreatedAt || !hasUpdatedAt || !hasMemoryType || !hasTags) {
          console.error(
            `VectorStore: Schema missing fields - createdAt: ${hasCreatedAt}, updatedAt: ${hasUpdatedAt}, memoryType: ${hasMemoryType}, tags: ${hasTags}`
          )
          return true
        }
      }

      return false
    } catch (error) {
      console.error('VectorStore: Error checking schema, assuming migration needed:', error)
      return true
    }
  }

  /**
   * Migrate data from old schema to new schema
   */
  private async migrateTable(): Promise<void> {
    if (!this.db || !this.table) {
      return
    }

    console.error('VectorStore: Starting schema migration...')

    try {
      // Read all existing data
      const allRecords = await this.table.query().toArray()
      console.error(`VectorStore: Read ${allRecords.length} records for migration`)

      if (allRecords.length === 0) {
        // No data to migrate, just drop and recreate
        await this.db.dropTable(this.config.tableName)
        this.table = null
        console.error('VectorStore: Dropped empty table, will recreate on first insert')
        return
      }

      // Transform records to new schema format
      const now = new Date().toISOString()
      const migratedRecords = allRecords.map((record) => {
        const rawMetadata = record.metadata as Record<string, unknown>

        // Normalize tags - handle Arrow vector types
        let tags: string[] = []
        const rawTags = rawMetadata['tags']
        if (rawTags) {
          if (Array.isArray(rawTags)) {
            tags = [...rawTags] as string[]
          } else if (typeof (rawTags as { toArray?: () => string[] }).toArray === 'function') {
            tags = (rawTags as { toArray: () => string[] }).toArray()
          }
        }

        // Normalize vector - handle Arrow FixedSizeList types
        // LanceDB returns vectors as Float32Array or similar typed arrays
        let vector: number[] = []
        if (record.vector) {
          if (Array.isArray(record.vector)) {
            vector = [...record.vector] as number[]
          } else if (record.vector instanceof Float32Array) {
            vector = Array.from(record.vector)
          } else if (
            typeof (record.vector as { toArray?: () => number[] }).toArray === 'function'
          ) {
            vector = Array.from((record.vector as { toArray: () => number[] }).toArray())
          } else if (typeof record.vector === 'object' && record.vector !== null) {
            // Handle other iterable types
            vector = Array.from(record.vector as Iterable<number>)
          }
        }

        // Build migrated metadata with all required fields
        const migratedMetadata: DocumentMetadata = {
          fileName: (rawMetadata['fileName'] as string) || 'unknown',
          fileSize: (rawMetadata['fileSize'] as number) || 0,
          fileType: (rawMetadata['fileType'] as string) || 'unknown',
          language: (rawMetadata['language'] as string | null) || null,
          memoryType: (rawMetadata['memoryType'] as string | null) || null,
          tags,
          project: (rawMetadata['project'] as string | null) || null,
          expiresAt: (rawMetadata['expiresAt'] as string | null) || null,
          createdAt:
            (rawMetadata['createdAt'] as string | null) || (record.timestamp as string) || now,
          updatedAt:
            (rawMetadata['updatedAt'] as string | null) || (record.timestamp as string) || now,
        }

        return {
          id: record.id as string,
          filePath: record.filePath as string,
          chunkIndex: record.chunkIndex as number,
          text: record.text as string,
          vector,
          metadata: migratedMetadata,
          timestamp: (record.timestamp as string) || now,
        }
      })

      // Drop old table
      await this.db.dropTable(this.config.tableName)
      console.error('VectorStore: Dropped old table')

      // Create new table with proper schema
      const schema = this.getSchema()
      this.table = await this.db.createTable(this.config.tableName, migratedRecords, { schema })
      console.error(
        `VectorStore: Created new table with ${migratedRecords.length} migrated records`
      )

      console.error('VectorStore: Migration completed successfully!')
    } catch (error) {
      console.error('VectorStore: Migration failed:', error)
      throw new DatabaseError('Failed to migrate table schema', error as Error)
    }
  }

  /**
   * Initialize LanceDB and create table
   */
  async initialize(): Promise<void> {
    try {
      // Connect to LanceDB
      this.db = await connect(this.config.dbPath)

      // Check table existence and create if needed
      const tableNames = await this.db.tableNames()
      if (tableNames.includes(this.config.tableName)) {
        // Open existing table
        this.table = await this.db.openTable(this.config.tableName)
        console.error(`VectorStore: Opened existing table "${this.config.tableName}"`)

        // Check if migration is needed
        if (await this.needsMigration(this.table)) {
          console.error('VectorStore: Schema migration required')
          await this.migrateTable()
        }

        // Ensure FTS index exists (migration for existing databases)
        await this.ensureFtsIndex()
      } else {
        // Create new table (schema auto-defined on first data insertion)
        console.error(
          `VectorStore: Table "${this.config.tableName}" will be created on first data insertion`
        )
      }

      console.error(`VectorStore initialized: ${this.config.dbPath}`)
    } catch (error) {
      throw new DatabaseError('Failed to initialize VectorStore', error as Error)
    }
  }

  /**
   * Delete all chunks for specified file path
   *
   * @param filePath - File path (absolute)
   */
  async deleteChunks(filePath: string): Promise<void> {
    // Validate file path to prevent SQL injection
    validateFilePath(filePath)

    if (!this.table) {
      // If table doesn't exist, no deletion targets, return normally
      console.error('VectorStore: Skipping deletion as table does not exist')
      return
    }

    try {
      // Use LanceDB delete API to remove records matching filePath
      // Escape single quotes as additional safety measure
      const escapedFilePath = filePath.replace(/'/g, "''")

      // LanceDB's delete method doesn't throw errors if targets don't exist,
      // so call delete directly
      // Note: Field names are case-sensitive, use backticks for camelCase fields
      await this.table.delete(`\`filePath\` = '${escapedFilePath}'`)
      console.error(`VectorStore: Deleted chunks for file "${filePath}"`)

      // Rebuild FTS index after deleting data
      await this.rebuildFtsIndex()
    } catch (error) {
      // If error occurs, output warning log
      console.warn(`VectorStore: Error occurred while deleting file "${filePath}":`, error)
      // Don't treat as error if deletion targets don't exist or table is empty
      // Otherwise throw exception
      const errorMessage = (error as Error).message.toLowerCase()
      if (
        !errorMessage.includes('not found') &&
        !errorMessage.includes('does not exist') &&
        !errorMessage.includes('no matching')
      ) {
        throw new DatabaseError(`Failed to delete chunks for file: ${filePath}`, error as Error)
      }
    }
  }

  /**
   * Batch insert vector chunks
   *
   * @param chunks - Array of vector chunks
   */
  async insertChunks(chunks: VectorChunk[]): Promise<void> {
    if (chunks.length === 0) {
      return
    }

    try {
      if (!this.table) {
        // Create table on first insertion with explicit schema
        if (!this.db) {
          throw new DatabaseError('VectorStore is not initialized. Call initialize() first.')
        }
        const records = chunks.map((chunk) => chunk as unknown as Record<string, unknown>)
        console.error(
          'Creating table with records:',
          JSON.stringify(records[0]?.['metadata'], null, 2)
        )
        const schema = this.getSchema()
        this.table = await this.db.createTable(this.config.tableName, records, { schema })
        console.error(`VectorStore: Created table "${this.config.tableName}"`)

        // Create FTS index for hybrid search
        await this.ensureFtsIndex()
      } else {
        // Add data to existing table
        const records = chunks.map((chunk) => chunk as unknown as Record<string, unknown>)
        console.error(
          'Adding records with metadata:',
          JSON.stringify(records[0]?.['metadata'], null, 2)
        )
        await this.table.add(records)

        // Rebuild FTS index after adding new data
        await this.rebuildFtsIndex()
      }

      console.error(`VectorStore: Inserted ${chunks.length} chunks`)
    } catch (error) {
      console.error('Insert error details:', error)
      throw new DatabaseError('Failed to insert chunks', error as Error)
    }
  }

  /**
   * Ensure FTS index exists for hybrid search
   * Creates the index if it doesn't exist, falls back gracefully on error
   */
  private async ensureFtsIndex(): Promise<void> {
    if (!this.table || this.ftsEnabled) {
      return
    }

    try {
      // Create FTS index on the 'text' column
      await this.table.createIndex('text', {
        config: Index.fts(),
      })
      this.ftsEnabled = true
      console.error('VectorStore: FTS index created successfully')
    } catch (error) {
      // FTS index creation failed, continue with vector-only search
      console.error('VectorStore: FTS index creation failed, using vector-only search:', error)
      this.ftsEnabled = false
    }
  }

  /**
   * Rebuild FTS index after data changes (insert/delete)
   * LanceDB OSS requires explicit optimize() call to update FTS index
   */
  private async rebuildFtsIndex(): Promise<void> {
    if (!this.table || !this.ftsEnabled) {
      return
    }

    try {
      // Optimize table to rebuild indexes including FTS
      await this.table.optimize()
      console.error('VectorStore: FTS index rebuilt successfully')
    } catch (error) {
      // Log warning but don't fail the operation
      console.warn('VectorStore: FTS index rebuild failed:', error)
    }
  }

  /**
   * Apply grouping algorithm to filter results by detecting group boundaries.
   *
   * Uses statistical threshold (mean + k*std) to identify significant gaps (group boundaries).
   * - 'similar': Returns only the first group (cuts at first boundary)
   * - 'related': Returns up to 2 groups (cuts at second boundary)
   *
   * @param results - Search results sorted by distance (ascending)
   * @param mode - Grouping mode ('similar' = 1 group, 'related' = 2 groups)
   * @returns Filtered results
   */
  private applyGrouping(results: SearchResult[], mode: GroupingMode): SearchResult[] {
    if (results.length <= 1) return results

    // Calculate gaps between consecutive results with their indices
    const gaps: { index: number; gap: number }[] = []
    for (let i = 0; i < results.length - 1; i++) {
      const current = results[i]
      const next = results[i + 1]
      if (current !== undefined && next !== undefined) {
        gaps.push({ index: i + 1, gap: next.score - current.score })
      }
    }

    if (gaps.length === 0) return results

    // Calculate statistical threshold to identify significant gaps (group boundaries)
    const gapValues = gaps.map((g) => g.gap)
    const mean = gapValues.reduce((a, b) => a + b, 0) / gapValues.length
    const variance = gapValues.reduce((a, b) => a + (b - mean) ** 2, 0) / gapValues.length
    const std = Math.sqrt(variance)
    const threshold = mean + GROUPING_BOUNDARY_STD_MULTIPLIER * std

    // Find all significant gaps (group boundaries)
    const boundaries = gaps.filter((g) => g.gap > threshold).map((g) => g.index)

    // If no boundaries found, return all results
    if (boundaries.length === 0) return results

    // Determine how many groups to include based on mode
    // 'similar': 1 group (cut at first boundary)
    // 'related': 2 groups (cut at second boundary, or return all if only 1 boundary)
    const groupsToInclude = mode === 'similar' ? 1 : 2
    const boundaryIndex = groupsToInclude - 1

    // If we don't have enough boundaries, return all results for 'related' mode
    if (boundaryIndex >= boundaries.length) {
      return mode === 'related' ? results : results.slice(0, boundaries[0])
    }

    // Cut at the appropriate boundary
    return results.slice(0, boundaries[boundaryIndex])
  }

  /**
   * Convert raw LanceDB result to SearchResult with proper metadata handling
   * Handles Arrow vector types (e.g., Utf8Vector) conversion to JS arrays
   */
  private toSearchResultWithMetadata(result: Record<string, unknown>): SearchResult {
    const rawMetadata = result['metadata'] as DocumentMetadata
    // Convert Arrow Utf8Vector to JS array
    let tags: string[] = []
    if (rawMetadata.tags) {
      if (Array.isArray(rawMetadata.tags)) {
        tags = [...rawMetadata.tags]
      } else if (
        typeof (rawMetadata.tags as unknown as { toArray?: () => string[] }).toArray === 'function'
      ) {
        tags = (rawMetadata.tags as unknown as { toArray: () => string[] }).toArray()
      }
    }
    return {
      filePath: result['filePath'] as string,
      chunkIndex: result['chunkIndex'] as number,
      text: result['text'] as string,
      score: (result['_distance'] as number) ?? 0,
      metadata: {
        fileName: rawMetadata.fileName,
        fileSize: rawMetadata.fileSize,
        fileType: rawMetadata.fileType,
        language: rawMetadata.language,
        memoryType: rawMetadata.memoryType,
        tags,
        project: rawMetadata.project,
        expiresAt: rawMetadata.expiresAt,
        createdAt: rawMetadata.createdAt,
        updatedAt: rawMetadata.updatedAt,
      },
    }
  }

  /**
   * Execute vector search with hybrid search and filtering support
   * Combines BM25 + vector search when FTS is enabled, plus metadata filtering
   *
   * @param queryVector - Query vector (dimension depends on model)
   * @param options - Search options including queryText for hybrid search and filters
   * @returns Array of search results (sorted by distance ascending, filtered by quality settings)
   */
  async search(
    queryVector: number[],
    options?: {
      /** Query text for hybrid BM25 + vector search */
      queryText?: string
      /** Maximum number of results (default 10, max 20) */
      limit?: number
      /** Filter by type: 'all', 'file', or 'memory' */
      type?: 'all' | 'file' | 'memory'
      /** Filter by tags (AND logic - must have all specified tags) */
      tags?: string[]
      /** Filter by project identifier */
      project?: string
      /** Maximum distance threshold (lower is more similar) */
      minScore?: number
    }
  ): Promise<SearchResult[]> {
    if (!this.table) {
      console.error('VectorStore: Returning empty results as table does not exist')
      return []
    }

    const limit = options?.limit ?? 10
    if (limit < 1 || limit > 20) {
      throw new DatabaseError(`Invalid limit: expected 1-20, got ${limit}`)
    }

    try {
      let rawResults: Record<string, unknown>[]
      const queryText = options?.queryText

      // Use hybrid search if FTS is enabled, query text is provided, and hybridWeight > 0
      const hybridWeight = this.config.hybridWeight ?? 0.6
      if (this.ftsEnabled && queryText && queryText.trim().length > 0 && hybridWeight > 0) {
        try {
          // Get more candidates for filtering and reranking
          const candidateLimit = limit * HYBRID_SEARCH_CANDIDATE_MULTIPLIER * 2

          // Hybrid search: combine FTS (BM25) and vector search
          const ftsResults = await this.table
            .search(queryText, 'fts', 'text')
            .select(['filePath', 'chunkIndex', 'text', 'metadata'])
            .limit(candidateLimit)
            .toArray()

          // Also perform vector search
          const vectorResults = await this.table
            .vectorSearch(queryVector)
            .distanceType('dot')
            .limit(candidateLimit)
            .toArray()

          // Combine and rerank
          rawResults = this.hybridRerank(ftsResults, vectorResults, candidateLimit)
        } catch (ftsError) {
          // FTS search failed, fall back to vector-only search
          console.error('VectorStore: FTS search failed, falling back to vector-only:', ftsError)
          this.ftsEnabled = false

          let query = this.table
            .vectorSearch(queryVector)
            .distanceType('dot')
            .limit(limit * 3)
          if (this.config.maxDistance !== undefined) {
            query = query.distanceRange(undefined, this.config.maxDistance)
          }
          rawResults = await query.toArray()
        }
      } else {
        // Vector-only search - get extra candidates for filtering
        let query = this.table
          .vectorSearch(queryVector)
          .distanceType('dot')
          .limit(limit * 3)

        // Apply distance threshold if configured
        if (this.config.maxDistance !== undefined) {
          query = query.distanceRange(undefined, this.config.maxDistance)
        }

        rawResults = await query.toArray()
      }

      // Convert to SearchResult format with proper metadata handling
      let searchResults: SearchResult[] = rawResults.map((result) =>
        this.toSearchResultWithMetadata(result)
      )

      // Apply maxDistance filter (for hybrid search results that weren't filtered at query level)
      if (this.config.maxDistance !== undefined) {
        searchResults = searchResults.filter((result) => result.score <= this.config.maxDistance!)
      }

      // Apply metadata filters
      if (options) {
        // Filter by type
        if (options.type === 'memory') {
          searchResults = searchResults.filter((r) => r.filePath.startsWith('memory://'))
        } else if (options.type === 'file') {
          searchResults = searchResults.filter((r) => !r.filePath.startsWith('memory://'))
        }

        // Filter by tags (AND logic - must have all specified tags)
        if (options.tags && options.tags.length > 0) {
          searchResults = searchResults.filter((r) =>
            options.tags!.every((tag) => r.metadata.tags?.includes(tag))
          )
        }

        // Filter by project
        if (options.project) {
          searchResults = searchResults.filter((r) => r.metadata.project === options.project)
        }

        // Filter by minimum score (distance-based: lower is better, so <= for filtering)
        if (options.minScore !== undefined) {
          searchResults = searchResults.filter((r) => r.score <= options.minScore!)
        }
      }

      // Apply grouping filter if configured
      if (this.config.grouping && searchResults.length > 1) {
        searchResults = this.applyGrouping(searchResults, this.config.grouping)
      }

      // Return up to limit results
      return searchResults.slice(0, limit)
    } catch (error) {
      throw new DatabaseError('Failed to search vectors', error as Error)
    }
  }

  /**
   * Rerank results from hybrid search using linear combination
   * Weight is configurable via hybridWeight (default 0.6 = 60% BM25, 40% vector)
   */
  private hybridRerank(
    ftsResults: Record<string, unknown>[],
    vectorResults: Record<string, unknown>[],
    limit: number
  ): Record<string, unknown>[] {
    const scoreMap = new Map<string, { result: Record<string, unknown>; score: number }>()

    // Get weight from config (default 0.6 = BM25 60%, vector 40%)
    const bm25Weight = this.config.hybridWeight ?? 0.6
    const vectorWeight = 1 - bm25Weight

    // Process vector results (lower distance = better, convert to 0-1 score)
    for (let i = 0; i < vectorResults.length; i++) {
      const result = vectorResults[i]
      if (!result) continue
      const key = `${result['filePath']}:${result['chunkIndex']}`
      const distance = (result['_distance'] as number) ?? 1
      // Convert distance to similarity score (0-1 range, higher is better)
      const vectorScore = Math.max(0, 1 - distance / DOT_PRODUCT_MAX_DISTANCE)
      scoreMap.set(key, { result, score: vectorScore * vectorWeight })
    }

    // Process FTS results (rank-based scoring)
    for (let i = 0; i < ftsResults.length; i++) {
      const result = ftsResults[i]
      if (!result) continue
      const key = `${result['filePath']}:${result['chunkIndex']}`
      // FTS score based on rank position (higher rank = higher score)
      const ftsScore = 1 - i / ftsResults.length

      if (scoreMap.has(key)) {
        // Combine scores
        const existing = scoreMap.get(key)
        if (existing) {
          existing.score += ftsScore * bm25Weight
        }
      } else {
        scoreMap.set(key, { result, score: ftsScore * bm25Weight })
      }
    }

    // Sort by combined score and return top results
    const sortedResults = Array.from(scoreMap.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map((item) => ({
        ...item.result,
        _distance: 1 - item.score, // Convert back to distance for consistency
      }))

    return sortedResults
  }

  /**
   * Get list of ingested files
   *
   * @param filters - Optional filters for file listing
   * @returns Array of file information
   */
  async listFiles(filters?: {
    type?: 'all' | 'file' | 'memory'
    tags?: string[]
    project?: string
    search?: string
    limit?: number
  }): Promise<
    {
      filePath: string
      chunkCount: number
      timestamp: string
      metadata?: DocumentMetadata
    }[]
  > {
    if (!this.table) {
      return [] // Return empty array if table doesn't exist
    }

    try {
      // Retrieve all records
      const allRecords = await this.table.query().toArray()

      // Group by file path
      const fileMap = new Map<
        string,
        { chunkCount: number; timestamp: string; metadata?: DocumentMetadata }
      >()

      for (const record of allRecords) {
        const filePath = record.filePath as string
        const timestamp = record.timestamp as string
        const rawMetadata = record.metadata as DocumentMetadata

        // Normalize metadata: LanceDB returns Arrow vector types (e.g., Utf8Vector)
        // which need to be converted to JS arrays using toArray()
        let tags: string[] = []
        if (rawMetadata.tags) {
          if (Array.isArray(rawMetadata.tags)) {
            tags = [...rawMetadata.tags]
          } else if (
            typeof (rawMetadata.tags as unknown as { toArray?: () => string[] }).toArray ===
            'function'
          ) {
            tags = (rawMetadata.tags as unknown as { toArray: () => string[] }).toArray()
          }
        }
        const metadata: DocumentMetadata = {
          fileName: rawMetadata.fileName,
          fileSize: rawMetadata.fileSize,
          fileType: rawMetadata.fileType,
          language: rawMetadata.language,
          memoryType: rawMetadata.memoryType,
          tags,
          project: rawMetadata.project,
          expiresAt: rawMetadata.expiresAt,
          createdAt: rawMetadata.createdAt,
          updatedAt: rawMetadata.updatedAt,
        }

        if (fileMap.has(filePath)) {
          const fileInfo = fileMap.get(filePath)
          if (fileInfo) {
            fileInfo.chunkCount += 1
            // Keep most recent timestamp
            if (timestamp > fileInfo.timestamp) {
              fileInfo.timestamp = timestamp
              fileInfo.metadata = metadata // Update metadata to latest
            }
          }
        } else {
          fileMap.set(filePath, { chunkCount: 1, timestamp, metadata })
        }
      }

      // Convert Map to array of objects
      let results: {
        filePath: string
        chunkCount: number
        timestamp: string
        metadata?: DocumentMetadata
      }[] = Array.from(fileMap.entries()).map(([filePath, info]) => {
        const result: {
          filePath: string
          chunkCount: number
          timestamp: string
          metadata?: DocumentMetadata
        } = {
          filePath,
          chunkCount: info.chunkCount,
          timestamp: info.timestamp,
        }
        if (info.metadata) {
          result.metadata = info.metadata
        }
        return result
      })

      // Apply filters
      if (filters) {
        // Filter by type
        if (filters.type === 'memory') {
          results = results.filter((r) => r.filePath.startsWith('memory://'))
        } else if (filters.type === 'file') {
          results = results.filter((r) => !r.filePath.startsWith('memory://'))
        }

        // Filter by tags (AND logic - must have all specified tags)
        if (filters.tags && filters.tags.length > 0) {
          results = results.filter((r) =>
            filters.tags!.every((tag) => r.metadata?.tags?.includes(tag))
          )
        }

        // Filter by project
        if (filters.project) {
          results = results.filter((r) => r.metadata?.project === filters.project)
        }

        // Filter by search string (label/filename substring match)
        if (filters.search) {
          const searchLower = filters.search.toLowerCase()
          results = results.filter(
            (r) =>
              r.filePath.toLowerCase().includes(searchLower) ||
              r.metadata?.fileName.toLowerCase().includes(searchLower)
          )
        }

        // Apply limit
        if (filters.limit && filters.limit > 0) {
          results = results.slice(0, filters.limit)
        }
      }

      return results
    } catch (error) {
      throw new DatabaseError('Failed to list files', error as Error)
    }
  }

  /**
   * Get system status
   *
   * @returns System status information
   */
  async getStatus(): Promise<{
    documentCount: number
    chunkCount: number
    memoryUsage: number
    uptime: number
    ftsIndexEnabled: boolean
    searchMode: 'hybrid' | 'vector-only'
  }> {
    if (!this.table) {
      return {
        documentCount: 0,
        chunkCount: 0,
        memoryUsage: 0,
        uptime: process.uptime(),
        ftsIndexEnabled: false,
        searchMode: 'vector-only',
      }
    }

    try {
      // Retrieve all records
      const allRecords = await this.table.query().toArray()
      const chunkCount = allRecords.length

      // Count unique file paths
      const uniqueFilePaths = new Set(allRecords.map((record) => record.filePath as string))
      const documentCount = uniqueFilePaths.size

      // Get memory usage (in MB)
      const memoryUsage = process.memoryUsage().heapUsed / 1024 / 1024

      // Get uptime (in seconds)
      const uptime = process.uptime()

      return {
        documentCount,
        chunkCount,
        memoryUsage,
        uptime,
        ftsIndexEnabled: this.ftsEnabled,
        searchMode:
          this.ftsEnabled && (this.config.hybridWeight ?? 0.6) > 0 ? 'hybrid' : 'vector-only',
      }
    } catch (error) {
      throw new DatabaseError('Failed to get status', error as Error)
    }
  }

  /**
   * Get memory by label
   *
   * @param label - Memory label
   * @returns Array of chunks for the specified memory
   */
  async getMemoryByLabel(label: string): Promise<VectorChunk[]> {
    if (!this.table) {
      return []
    }

    try {
      const filePath = `memory://${label}`
      const allRecords = await this.table.query().toArray()

      // Filter by file path
      const matchingRecords = allRecords.filter((record) => record.filePath === filePath)

      // Convert to VectorChunk format and normalize metadata
      // Note: LanceDB returns Arrow vector types (e.g., Utf8Vector) which need
      // to be converted to JS arrays using toArray()
      return matchingRecords.map((record) => {
        const rawMetadata = record.metadata as DocumentMetadata
        // Convert Arrow Utf8Vector to JS array
        let tags: string[] = []
        if (rawMetadata.tags) {
          if (Array.isArray(rawMetadata.tags)) {
            tags = [...rawMetadata.tags]
          } else if (
            typeof (rawMetadata.tags as unknown as { toArray?: () => string[] }).toArray ===
            'function'
          ) {
            tags = (rawMetadata.tags as unknown as { toArray: () => string[] }).toArray()
          }
        }
        return {
          id: record.id as string,
          filePath: record.filePath as string,
          chunkIndex: record.chunkIndex as number,
          text: record.text as string,
          vector: record.vector as number[],
          metadata: {
            fileName: rawMetadata.fileName,
            fileSize: rawMetadata.fileSize,
            fileType: rawMetadata.fileType,
            language: rawMetadata.language,
            memoryType: rawMetadata.memoryType,
            tags,
            project: rawMetadata.project,
            expiresAt: rawMetadata.expiresAt,
            createdAt: rawMetadata.createdAt,
            updatedAt: rawMetadata.updatedAt,
          },
          timestamp: record.timestamp as string,
        }
      })
    } catch (error) {
      throw new DatabaseError(`Failed to get memory by label: ${label}`, error as Error)
    }
  }

  /**
   * Cleanup expired memories
   *
   * @returns Number of deleted entries
   */
  async cleanupExpired(): Promise<number> {
    if (!this.table) {
      return 0
    }

    try {
      const now = new Date().toISOString()
      const allRecords = await this.table.query().toArray()

      // Find expired entries
      const expiredFilePaths = new Set<string>()
      for (const record of allRecords) {
        const metadata = record.metadata as DocumentMetadata
        if (metadata?.expiresAt && metadata.expiresAt < now) {
          expiredFilePaths.add(record.filePath as string)
        }
      }

      // Delete expired entries
      let deletedCount = 0
      for (const filePath of expiredFilePaths) {
        await this.deleteChunks(filePath)
        deletedCount++
      }

      console.error(`VectorStore: Cleaned up ${deletedCount} expired entries`)
      return deletedCount
    } catch (error) {
      throw new DatabaseError('Failed to cleanup expired memories', error as Error)
    }
  }
}
