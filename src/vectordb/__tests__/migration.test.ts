import * as fs from 'node:fs'
import * as path from 'node:path'
import { type Connection, type Table, connect } from '@lancedb/lancedb'
import * as arrow from 'apache-arrow'
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'
import { VectorStore } from '../index'

describe('VectorStore Schema Migration', () => {
  const testDbPath = './tmp/migration-test-db'
  const tableName = 'chunks'

  beforeAll(() => {
    if (!fs.existsSync('./tmp')) {
      fs.mkdirSync('./tmp', { recursive: true })
    }
  })

  afterEach(async () => {
    // Clean up test database
    if (fs.existsSync(testDbPath)) {
      fs.rmSync(testDbPath, { recursive: true, force: true })
    }
  })

  /**
   * Create a table with old schema (missing createdAt, updatedAt, memoryType, tags)
   */
  async function createOldSchemaTable(dbPath: string): Promise<void> {
    const db = await connect(dbPath)

    // Old schema without the new fields
    const oldSchema = new arrow.Schema([
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
          // Missing: memoryType, tags, project, expiresAt, createdAt, updatedAt
        ]),
        false
      ),
      new arrow.Field('timestamp', new arrow.Utf8(), false),
    ])

    // Create test record with old schema
    const testRecord = {
      id: 'test-id-1',
      filePath: '/test/file.txt',
      chunkIndex: 0,
      text: 'This is test content from old schema',
      vector: Array(384).fill(0.1),
      metadata: {
        fileName: 'file.txt',
        fileSize: 100,
        fileType: 'txt',
        language: null,
      },
      timestamp: '2024-01-01T00:00:00.000Z',
    }

    await db.createTable(tableName, [testRecord], { schema: oldSchema })
    await db.close()
  }

  it('should detect old schema and migrate data', async () => {
    // Create old schema table
    await createOldSchemaTable(testDbPath)

    // Initialize VectorStore - should detect and migrate
    const vectorStore = new VectorStore({ dbPath: testDbPath, tableName })
    await vectorStore.initialize()

    // Verify data was preserved
    const files = await vectorStore.listFiles()
    expect(files).toHaveLength(1)
    expect(files[0].filePath).toBe('/test/file.txt')
    expect(files[0].chunkCount).toBe(1)

    // Verify new fields are present with default values
    const metadata = files[0].metadata
    expect(metadata).toBeDefined()
    expect(metadata?.fileName).toBe('file.txt')
    expect(metadata?.fileSize).toBe(100)
    expect(metadata?.fileType).toBe('txt')
    // New fields should have default values
    expect(metadata?.createdAt).toBeDefined()
    expect(metadata?.updatedAt).toBeDefined()
    expect(metadata?.tags).toEqual([])
    expect(metadata?.memoryType).toBeNull()
    expect(metadata?.project).toBeNull()
    expect(metadata?.expiresAt).toBeNull()
  })

  it('should preserve text content during migration', async () => {
    // Create old schema table
    await createOldSchemaTable(testDbPath)

    // Initialize VectorStore - should detect and migrate
    const vectorStore = new VectorStore({ dbPath: testDbPath, tableName })
    await vectorStore.initialize()

    // Search for the migrated content
    const results = await vectorStore.search(Array(384).fill(0.1), 10)
    expect(results).toHaveLength(1)
    expect(results[0].text).toBe('This is test content from old schema')
  })

  it('should allow new inserts after migration', async () => {
    // Create old schema table
    await createOldSchemaTable(testDbPath)

    // Initialize VectorStore - should detect and migrate
    const vectorStore = new VectorStore({ dbPath: testDbPath, tableName })
    await vectorStore.initialize()

    // Insert new data with full schema
    const newChunk = {
      id: 'new-id-1',
      filePath: '/test/new-file.txt',
      chunkIndex: 0,
      text: 'This is new content after migration',
      vector: Array(384).fill(0.2),
      metadata: {
        fileName: 'new-file.txt',
        fileSize: 200,
        fileType: 'txt',
        language: null,
        memoryType: 'file',
        tags: ['test', 'migration'],
        project: 'test-project',
        expiresAt: null,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      },
      timestamp: new Date().toISOString(),
    }

    await vectorStore.insertChunks([newChunk])

    // Verify both old and new data exist
    const files = await vectorStore.listFiles()
    expect(files).toHaveLength(2)

    const newFile = files.find((f) => f.filePath === '/test/new-file.txt')
    expect(newFile).toBeDefined()
    expect(newFile?.metadata?.tags).toEqual(['test', 'migration'])
    expect(newFile?.metadata?.project).toBe('test-project')
  })

  it('should not migrate if schema is already current', async () => {
    // Create VectorStore and insert data with current schema
    const vectorStore = new VectorStore({ dbPath: testDbPath, tableName })
    await vectorStore.initialize()

    const chunk = {
      id: 'current-id-1',
      filePath: '/test/current-file.txt',
      chunkIndex: 0,
      text: 'Content with current schema',
      vector: Array(384).fill(0.3),
      metadata: {
        fileName: 'current-file.txt',
        fileSize: 300,
        fileType: 'txt',
        language: null,
        memoryType: 'file',
        tags: ['current'],
        project: null,
        expiresAt: null,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      },
      timestamp: new Date().toISOString(),
    }

    await vectorStore.insertChunks([chunk])

    // Re-initialize (should not migrate)
    const vectorStore2 = new VectorStore({ dbPath: testDbPath, tableName })
    await vectorStore2.initialize()

    // Verify data is still intact
    const files = await vectorStore2.listFiles()
    expect(files).toHaveLength(1)
    expect(files[0].metadata?.tags).toEqual(['current'])
  })
})
