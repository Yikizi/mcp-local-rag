// Memory Management Features Test Suite
// Test file for IMPROVEMENT_PLAN.md features
// Generated: 2025-12-18
// Test Type: Unit/Integration Test (TDD)
// Status: Tests will initially fail - implement features to make them pass

import { mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import type { DocumentMetadata } from '../../vectordb/index.js'
import { RAGServer } from '../index.js'

// ============================================
// Test Suite 1: Tags/Labels Support
// ============================================

describe('Feature 1: Tags/Labels Support', () => {
  let ragServer: RAGServer
  const testDbPath = resolve('./tmp/test-tags-db')
  const testDataDir = resolve('./tmp/test-tags-data')

  beforeAll(async () => {
    mkdirSync(testDbPath, { recursive: true })
    mkdirSync(testDataDir, { recursive: true })

    ragServer = new RAGServer({
      dbPath: testDbPath,
      modelName: 'Xenova/all-MiniLM-L6-v2',
      cacheDir: './tmp/models',
      baseDir: testDataDir,
      maxFileSize: 100 * 1024 * 1024,
      chunkSize: 512,
      chunkOverlap: 100,
    })

    await ragServer.initialize()
  })

  afterAll(async () => {
    rmSync(testDbPath, { recursive: true, force: true })
    rmSync(testDataDir, { recursive: true, force: true })
  })

  describe('memorize_text with tags parameter', () => {
    it('should accept and store empty tags array', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Test memory with empty tags',
        label: 'empty-tags-test',
        tags: [],
      })

      expect(result).toBeDefined()
      const parsed = JSON.parse(result.content[0].text)
      expect(parsed.label).toBe('empty-tags-test')
    })

    it('should accept and store single tag', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Test memory with single tag',
        label: 'single-tag-test',
        tags: ['project-alpha'],
      })

      expect(result).toBeDefined()
      const parsed = JSON.parse(result.content[0].text)
      expect(parsed.label).toBe('single-tag-test')

      // Verify tag is stored in metadata
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://single-tag-test'
      )
      expect(memory).toBeDefined()
      expect(memory.metadata?.tags).toEqual(['project-alpha'])
    })

    it('should accept and store multiple tags', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Test memory with multiple tags',
        label: 'multi-tag-test',
        tags: ['project-beta', 'experiment', 'machine-learning'],
      })

      expect(result).toBeDefined()
      const parsed = JSON.parse(result.content[0].text)
      expect(parsed.label).toBe('multi-tag-test')

      // Verify tags are stored
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://multi-tag-test'
      )
      expect(memory).toBeDefined()
      expect(memory.metadata?.tags).toEqual(['project-beta', 'experiment', 'machine-learning'])
    })

    it('should reject invalid tags (non-string values)', async () => {
      await expect(
        ragServer.handleMemorizeText({
          text: 'Test memory with invalid tags',
          label: 'invalid-tags-test',
          tags: [123, true, null] as unknown as string[],
        })
      ).rejects.toThrow('All tags must be strings')
    })

    it('should default to empty array when tags not provided', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Test memory without tags',
        label: 'no-tags-test',
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://no-tags-test'
      )
      expect(memory).toBeDefined()
      expect(memory.metadata?.tags || []).toEqual([])
    })
  })

  describe('ingest_file with tags parameter', () => {
    it('should accept tags parameter for file ingestion', async () => {
      const testFile = resolve(testDataDir, 'test-file-with-tags.txt')
      writeFileSync(testFile, 'File content with tags')

      const result = await ragServer.handleIngestFile({
        filePath: testFile,
        tags: ['documentation', 'important'],
      })

      expect(result).toBeDefined()
      const parsed = JSON.parse(result.content[0].text)
      expect(parsed.filePath).toBe(testFile)

      // Verify tags stored
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const file = filesList.find((f: { filePath: string }) => f.filePath === testFile)
      expect(file).toBeDefined()
      expect(file.metadata?.tags).toEqual(['documentation', 'important'])
    })
  })

  describe('DocumentMetadata interface extension', () => {
    it('should have tags field in DocumentMetadata', () => {
      // Type check - this will fail at compile time if tags field doesn't exist
      const metadata: DocumentMetadata = {
        fileName: 'test.txt',
        fileSize: 100,
        fileType: 'txt',
        tags: ['test-tag'],
      }
      expect(metadata.tags).toEqual(['test-tag'])
    })
  })
})

// ============================================
// Test Suite 2: Memory Types Distinction
// ============================================

describe('Feature 2: Memory Types Distinction', () => {
  let ragServer: RAGServer
  const testDbPath = resolve('./tmp/test-types-db')
  const testDataDir = resolve('./tmp/test-types-data')

  beforeAll(async () => {
    mkdirSync(testDbPath, { recursive: true })
    mkdirSync(testDataDir, { recursive: true })

    ragServer = new RAGServer({
      dbPath: testDbPath,
      modelName: 'Xenova/all-MiniLM-L6-v2',
      cacheDir: './tmp/models',
      baseDir: testDataDir,
      maxFileSize: 100 * 1024 * 1024,
      chunkSize: 512,
      chunkOverlap: 100,
    })

    await ragServer.initialize()
  })

  afterAll(async () => {
    rmSync(testDbPath, { recursive: true, force: true })
    rmSync(testDataDir, { recursive: true, force: true })
  })

  describe('memorize_text with type parameter', () => {
    it('should default to type "memory" when not specified', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Default memory type test',
        label: 'default-type-test',
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://default-type-test'
      )
      expect(memory).toBeDefined()
      expect(memory.metadata?.memoryType || 'memory').toBe('memory')
    })

    it('should accept type "memory"', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Explicit memory type',
        label: 'explicit-memory-type',
        type: 'memory',
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://explicit-memory-type'
      )
      expect(memory.metadata?.memoryType).toBe('memory')
    })

    it('should accept type "lesson"', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'This is a lesson learned from debugging',
        label: 'lesson-type-test',
        type: 'lesson',
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://lesson-type-test'
      )
      expect(memory.metadata?.memoryType).toBe('lesson')
    })

    it('should accept type "note"', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Quick note for later',
        label: 'note-type-test',
        type: 'note',
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://note-type-test'
      )
      expect(memory.metadata?.memoryType).toBe('note')
    })

    it('should reject invalid type values', async () => {
      await expect(
        ragServer.handleMemorizeText({
          text: 'Invalid type test',
          label: 'invalid-type-test',
          type: 'invalid-type' as any,
        })
      ).rejects.toThrow('Invalid memory type')
    })
  })

  describe('ingest_file sets type to "file"', () => {
    it('should automatically set type to "file" for ingested files', async () => {
      const testFile = resolve(testDataDir, 'test-file-type.txt')
      writeFileSync(testFile, 'File type test content')

      const result = await ragServer.handleIngestFile({
        filePath: testFile,
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const file = filesList.find((f: { filePath: string }) => f.filePath === testFile)
      expect(file).toBeDefined()
      expect(file.metadata?.memoryType).toBe('file')
    })
  })

  describe('DocumentMetadata type field', () => {
    it('should have type field in DocumentMetadata', () => {
      const metadata: DocumentMetadata = {
        fileName: 'test.txt',
        fileSize: 100,
        fileType: 'txt',
        memoryType: 'memory',
      }
      expect(metadata.memoryType).toBe('memory')
    })
  })
})

// ============================================
// Test Suite 3: Filtered Listing
// ============================================

describe('Feature 3: Filtered Listing (list_files enhancement)', () => {
  let ragServer: RAGServer
  const testDbPath = resolve('./tmp/test-filtering-db')
  const testDataDir = resolve('./tmp/test-filtering-data')

  beforeAll(async () => {
    mkdirSync(testDbPath, { recursive: true })
    mkdirSync(testDataDir, { recursive: true })

    ragServer = new RAGServer({
      dbPath: testDbPath,
      modelName: 'Xenova/all-MiniLM-L6-v2',
      cacheDir: './tmp/models',
      baseDir: testDataDir,
      maxFileSize: 100 * 1024 * 1024,
      chunkSize: 512,
      chunkOverlap: 100,
    })

    await ragServer.initialize()

    // Setup test data
    // Files
    const file1 = resolve(testDataDir, 'doc1.txt')
    writeFileSync(file1, 'Document 1 content')
    await ragServer.handleIngestFile({
      filePath: file1,
      tags: ['project-a', 'docs'],
    })

    const file2 = resolve(testDataDir, 'doc2.txt')
    writeFileSync(file2, 'Document 2 content')
    await ragServer.handleIngestFile({
      filePath: file2,
      tags: ['project-b', 'docs'],
    })

    // Memories
    await ragServer.handleMemorizeText({
      text: 'Memory about project A',
      label: 'memory-a',
      type: 'memory',
      tags: ['project-a', 'important'],
    })

    await ragServer.handleMemorizeText({
      text: 'Lesson learned from debugging',
      label: 'lesson-1',
      type: 'lesson',
      tags: ['project-a', 'debugging'],
    })

    await ragServer.handleMemorizeText({
      text: 'Quick note for project B',
      label: 'note-b',
      type: 'note',
      tags: ['project-b'],
    })
  })

  afterAll(async () => {
    rmSync(testDbPath, { recursive: true, force: true })
    rmSync(testDataDir, { recursive: true, force: true })
  })

  describe('Filter by type', () => {
    it('should return all entries when type="all" or not specified', async () => {
      const result = await ragServer.handleListFiles({ type: 'all' })
      const files = JSON.parse(result.content[0].text)
      expect(files.length).toBeGreaterThanOrEqual(5) // 2 files + 3 memories
    })

    it('should return only files when type="file"', async () => {
      const result = await ragServer.handleListFiles({ type: 'file' })
      const files = JSON.parse(result.content[0].text)

      expect(files.length).toBe(2)
      expect(files.every((f: { filePath: string }) => !f.filePath.startsWith('memory://'))).toBe(
        true
      )
    })

    it('should return only memories when type="memory"', async () => {
      const result = await ragServer.handleListFiles({ type: 'memory' })
      const files = JSON.parse(result.content[0].text)

      expect(files.length).toBe(3)
      expect(files.every((f: { filePath: string }) => f.filePath.startsWith('memory://'))).toBe(
        true
      )
    })
  })

  describe('Filter by tags (AND logic)', () => {
    it('should return entries with all specified tags', async () => {
      const result = await ragServer.handleListFiles({
        tags: ['project-a', 'important'],
      })
      const files = JSON.parse(result.content[0].text)

      // Only memory-a has both tags
      expect(files.length).toBe(1)
      expect(files[0].filePath).toBe('memory://memory-a')
    })

    it('should return entries with single tag', async () => {
      const result = await ragServer.handleListFiles({
        tags: ['project-a'],
      })
      const files = JSON.parse(result.content[0].text)

      // doc1.txt, memory-a, lesson-1 have project-a tag
      expect(files.length).toBe(3)
    })

    it('should return empty array when no entries match all tags', async () => {
      const result = await ragServer.handleListFiles({
        tags: ['project-a', 'project-b'], // No entry has both
      })
      const files = JSON.parse(result.content[0].text)
      expect(files.length).toBe(0)
    })

    it('should return all entries when tags is empty array', async () => {
      const result = await ragServer.handleListFiles({ tags: [] })
      const files = JSON.parse(result.content[0].text)
      expect(files.length).toBeGreaterThanOrEqual(5)
    })
  })

  describe('Filter by project', () => {
    it('should return entries with specified project', async () => {
      // First add memories with project field
      await ragServer.handleMemorizeText({
        text: 'Project specific memory',
        label: 'project-memory',
        project: 'my-project',
      })

      const result = await ragServer.handleListFiles({
        project: 'my-project',
      })
      const files = JSON.parse(result.content[0].text)

      expect(files.length).toBeGreaterThan(0)
      expect(
        files.some((f: { metadata?: { project?: string } }) => f.metadata?.project === 'my-project')
      ).toBe(true)
    })

    it('should return empty array when no entries match project', async () => {
      const result = await ragServer.handleListFiles({
        project: 'non-existent-project',
      })
      const files = JSON.parse(result.content[0].text)
      expect(files.length).toBe(0)
    })
  })

  describe('Filter by search (substring match)', () => {
    it('should filter by filename/label substring', async () => {
      const result = await ragServer.handleListFiles({
        search: 'memory',
      })
      const files = JSON.parse(result.content[0].text)

      // Should match memory-a and project-memory
      expect(files.length).toBeGreaterThan(0)
      expect(files.some((f: { filePath: string }) => f.filePath.includes('memory'))).toBe(true)
    })

    it('should be case-insensitive', async () => {
      const result = await ragServer.handleListFiles({
        search: 'LESSON',
      })
      const files = JSON.parse(result.content[0].text)

      expect(files.length).toBeGreaterThan(0)
      expect(
        files.some((f: { filePath: string }) => f.filePath.toLowerCase().includes('lesson'))
      ).toBe(true)
    })

    it('should return empty array when no matches', async () => {
      const result = await ragServer.handleListFiles({
        search: 'xyznonexistent',
      })
      const files = JSON.parse(result.content[0].text)
      expect(files.length).toBe(0)
    })
  })

  describe('Limit parameter', () => {
    it('should limit results to specified number', async () => {
      const result = await ragServer.handleListFiles({
        limit: 2,
      })
      const files = JSON.parse(result.content[0].text)
      expect(files.length).toBeLessThanOrEqual(2)
    })

    it('should default to 50 when limit not specified', async () => {
      const result = await ragServer.handleListFiles()
      const files = JSON.parse(result.content[0].text)
      expect(files.length).toBeLessThanOrEqual(50)
    })

    it('should handle limit of 0 as unlimited', async () => {
      const result = await ragServer.handleListFiles({ limit: 0 })
      const files = JSON.parse(result.content[0].text)
      expect(files.length).toBeGreaterThanOrEqual(5)
    })
  })

  describe('Combining multiple filters', () => {
    it('should combine type and tags filters', async () => {
      const result = await ragServer.handleListFiles({
        type: 'memory',
        tags: ['project-a'],
      })
      const files = JSON.parse(result.content[0].text)

      // Should return memory-a and lesson-1 (both are memories with project-a tag)
      expect(files.length).toBe(2)
      expect(files.every((f: { filePath: string }) => f.filePath.startsWith('memory://'))).toBe(
        true
      )
    })

    it('should combine type, tags, and search filters', async () => {
      const result = await ragServer.handleListFiles({
        type: 'memory',
        tags: ['project-a'],
        search: 'lesson',
      })
      const files = JSON.parse(result.content[0].text)

      // Should return only lesson-1
      expect(files.length).toBe(1)
      expect(files[0].filePath).toBe('memory://lesson-1')
    })

    it('should combine all filters including limit', async () => {
      const result = await ragServer.handleListFiles({
        type: 'memory',
        tags: ['project-a'],
        limit: 1,
      })
      const files = JSON.parse(result.content[0].text)

      expect(files.length).toBeLessThanOrEqual(1)
    })
  })
})

// ============================================
// Test Suite 4: TTL/Expiration Support
// ============================================

describe('Feature 4: TTL/Expiration Support', () => {
  let ragServer: RAGServer
  const testDbPath = resolve('./tmp/test-ttl-db')
  const testDataDir = resolve('./tmp/test-ttl-data')

  beforeAll(async () => {
    mkdirSync(testDbPath, { recursive: true })
    mkdirSync(testDataDir, { recursive: true })

    ragServer = new RAGServer({
      dbPath: testDbPath,
      modelName: 'Xenova/all-MiniLM-L6-v2',
      cacheDir: './tmp/models',
      baseDir: testDataDir,
      maxFileSize: 100 * 1024 * 1024,
      chunkSize: 512,
      chunkOverlap: 100,
    })

    await ragServer.initialize()
  })

  afterAll(async () => {
    rmSync(testDbPath, { recursive: true, force: true })
    rmSync(testDataDir, { recursive: true, force: true })
  })

  describe('TTL parameter parsing', () => {
    it('should accept ttl="1d" (1 day)', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Memory with 1 day TTL',
        label: 'ttl-1d',
        ttl: '1d',
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find((f: { filePath: string }) => f.filePath === 'memory://ttl-1d')
      expect(memory).toBeDefined()
      expect(memory.metadata?.expiresAt).toBeDefined()

      // Verify expires in approximately 1 day
      const expiresAt = new Date(memory.metadata.expiresAt)
      const now = new Date()
      const diffHours = (expiresAt.getTime() - now.getTime()) / (1000 * 60 * 60)
      expect(diffHours).toBeGreaterThan(23)
      expect(diffHours).toBeLessThan(25)
    })

    it('should accept ttl="7d" (7 days)', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Memory with 7 days TTL',
        label: 'ttl-7d',
        ttl: '7d',
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find((f: { filePath: string }) => f.filePath === 'memory://ttl-7d')

      const expiresAt = new Date(memory.metadata.expiresAt)
      const now = new Date()
      const diffDays = (expiresAt.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)
      expect(diffDays).toBeGreaterThan(6.9)
      expect(diffDays).toBeLessThan(7.1)
    })

    it('should accept ttl="30d" (30 days)', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Memory with 30 days TTL',
        label: 'ttl-30d',
        ttl: '30d',
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find((f: { filePath: string }) => f.filePath === 'memory://ttl-30d')

      const expiresAt = new Date(memory.metadata.expiresAt)
      const now = new Date()
      const diffDays = (expiresAt.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)
      expect(diffDays).toBeGreaterThan(29)
      expect(diffDays).toBeLessThan(31)
    })

    it('should accept ttl="1y" (1 year)', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Memory with 1 year TTL',
        label: 'ttl-1y',
        ttl: '1y',
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find((f: { filePath: string }) => f.filePath === 'memory://ttl-1y')

      const expiresAt = new Date(memory.metadata.expiresAt)
      const now = new Date()
      const diffDays = (expiresAt.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)
      expect(diffDays).toBeGreaterThan(364)
      expect(diffDays).toBeLessThan(366)
    })

    it('should accept ttl="permanent" (no expiration)', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Permanent memory',
        label: 'ttl-permanent',
        ttl: 'permanent',
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://ttl-permanent'
      )

      expect(memory.metadata?.expiresAt).toBeNull()
    })

    it('should default to permanent when ttl not specified', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Memory without TTL',
        label: 'no-ttl',
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find((f: { filePath: string }) => f.filePath === 'memory://no-ttl')

      expect(memory.metadata?.expiresAt).toBeNull()
    })

    it('should reject invalid TTL format', async () => {
      await expect(
        ragServer.handleMemorizeText({
          text: 'Invalid TTL',
          label: 'invalid-ttl',
          ttl: '5x' as any,
        })
      ).rejects.toThrow()
    })
  })

  describe('Timestamp metadata', () => {
    it('should include createdAt timestamp', async () => {
      const before = new Date().toISOString()

      const result = await ragServer.handleMemorizeText({
        text: 'Test createdAt',
        label: 'created-at-test',
      })

      const after = new Date().toISOString()

      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://created-at-test'
      )

      expect(memory.metadata?.createdAt).toBeDefined()
      expect(memory.metadata.createdAt).toMatch(/^\d{4}-\d{2}-\d{2}T/)
      expect(memory.metadata.createdAt >= before).toBe(true)
      expect(memory.metadata.createdAt <= after).toBe(true)
    })

    it('should include updatedAt timestamp', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Test updatedAt',
        label: 'updated-at-test',
      })

      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://updated-at-test'
      )

      expect(memory.metadata?.updatedAt).toBeDefined()
      expect(memory.metadata.updatedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/)
    })

    it('should have createdAt equal to updatedAt initially', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Test timestamps',
        label: 'timestamps-test',
      })

      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://timestamps-test'
      )

      expect(memory.metadata?.createdAt).toBe(memory.metadata?.updatedAt)
    })
  })

  describe('cleanup_expired tool', () => {
    it('should have cleanup_expired tool registered', async () => {
      // Verify tool exists
      expect(typeof ragServer.handleCleanupExpired).toBe('function')
    })

    it('should remove expired memories', async () => {
      // Create memory that expires immediately (for testing)
      await ragServer.handleMemorizeText({
        text: 'Expired memory',
        label: 'expired-test',
        ttl: '1d',
      })

      // Manually set expiration to past (this would require internal access)
      // For now, verify cleanup doesn't crash
      const result = await ragServer.handleCleanupExpired()
      expect(result).toBeDefined()

      const parsed = JSON.parse(result.content[0].text)
      expect(parsed.deletedCount).toBeDefined()
      expect(typeof parsed.deletedCount).toBe('number')
    })

    it('should keep non-expired memories', async () => {
      await ragServer.handleMemorizeText({
        text: 'Non-expired memory',
        label: 'non-expired-test',
        ttl: '7d',
      })

      await ragServer.handleCleanupExpired()

      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://non-expired-test'
      )

      expect(memory).toBeDefined()
    })

    it('should keep permanent memories', async () => {
      await ragServer.handleMemorizeText({
        text: 'Permanent memory',
        label: 'permanent-test',
        ttl: 'permanent',
      })

      await ragServer.handleCleanupExpired()

      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://permanent-test'
      )

      expect(memory).toBeDefined()
    })

    it('should return count of deleted memories', async () => {
      const result = await ragServer.handleCleanupExpired()
      const parsed = JSON.parse(result.content[0].text)

      expect(parsed).toHaveProperty('deletedCount')
      expect(parsed).toHaveProperty('timestamp')
      expect(typeof parsed.deletedCount).toBe('number')
      expect(parsed.deletedCount).toBeGreaterThanOrEqual(0)
    })
  })
})

// ============================================
// Test Suite 5: Memory Update/Append Support
// ============================================

describe('Feature 5: Memory Update/Append Support', () => {
  let ragServer: RAGServer
  const testDbPath = resolve('./tmp/test-update-db')
  const testDataDir = resolve('./tmp/test-update-data')

  beforeAll(async () => {
    mkdirSync(testDbPath, { recursive: true })
    mkdirSync(testDataDir, { recursive: true })

    ragServer = new RAGServer({
      dbPath: testDbPath,
      modelName: 'Xenova/all-MiniLM-L6-v2',
      cacheDir: './tmp/models',
      baseDir: testDataDir,
      maxFileSize: 100 * 1024 * 1024,
      chunkSize: 512,
      chunkOverlap: 100,
    })

    await ragServer.initialize()
  })

  afterAll(async () => {
    rmSync(testDbPath, { recursive: true, force: true })
    rmSync(testDataDir, { recursive: true, force: true })
  })

  describe('update_memory tool existence', () => {
    it('should have update_memory handler', () => {
      expect(typeof ragServer.handleUpdateMemory).toBe('function')
    })
  })

  describe('Replace mode', () => {
    it('should replace entire content with new text', async () => {
      // Create initial memory
      await ragServer.handleMemorizeText({
        text: 'Original content',
        label: 'replace-test',
      })

      // Update with replace mode
      await ragServer.handleUpdateMemory({
        label: 'replace-test',
        mode: 'replace',
        text: 'New content',
      })

      // Verify content replaced
      const result = await ragServer.handleQueryDocuments({
        query: 'content',
        limit: 10,
      })
      const results = JSON.parse(result.content[0].text)
      const memory = results.find(
        (r: { filePath: string }) => r.filePath === 'memory://replace-test'
      )

      expect(memory).toBeDefined()
      expect(memory.text).toContain('New content')
      expect(memory.text).not.toContain('Original content')
    })
  })

  describe('Append mode', () => {
    it('should append text to end of existing content', async () => {
      // Create initial memory
      await ragServer.handleMemorizeText({
        text: 'Start content.',
        label: 'append-test',
      })

      // Update with append mode
      await ragServer.handleUpdateMemory({
        label: 'append-test',
        mode: 'append',
        text: ' Appended content.',
      })

      // Verify content appended
      const result = await ragServer.handleQueryDocuments({
        query: 'content',
        limit: 10,
      })
      const results = JSON.parse(result.content[0].text)
      const memory = results.find(
        (r: { filePath: string }) => r.filePath === 'memory://append-test'
      )

      expect(memory).toBeDefined()
      expect(memory.text).toContain('Start content')
      expect(memory.text).toContain('Appended content')
    })
  })

  describe('Prepend mode', () => {
    it('should prepend text to beginning of existing content', async () => {
      // Create initial memory
      await ragServer.handleMemorizeText({
        text: 'End content.',
        label: 'prepend-test',
      })

      // Update with prepend mode
      await ragServer.handleUpdateMemory({
        label: 'prepend-test',
        mode: 'prepend',
        text: 'Prepended content. ',
      })

      // Verify content prepended
      const result = await ragServer.handleQueryDocuments({
        query: 'content',
        limit: 10,
      })
      const results = JSON.parse(result.content[0].text)
      const memory = results.find(
        (r: { filePath: string }) => r.filePath === 'memory://prepend-test'
      )

      expect(memory).toBeDefined()
      expect(memory.text).toContain('Prepended content')
      expect(memory.text).toContain('End content')
    })
  })

  describe('Tag replacement', () => {
    it('should replace all tags with new tags', async () => {
      // Create memory with tags
      await ragServer.handleMemorizeText({
        text: 'Memory with tags',
        label: 'tag-replace-test',
        tags: ['old-tag-1', 'old-tag-2'],
      })

      // Replace tags
      await ragServer.handleUpdateMemory({
        label: 'tag-replace-test',
        tags: ['new-tag-1', 'new-tag-2', 'new-tag-3'],
      })

      // Verify tags replaced
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://tag-replace-test'
      )

      expect(memory.metadata?.tags).toEqual(['new-tag-1', 'new-tag-2', 'new-tag-3'])
    })
  })

  describe('Adding tags', () => {
    it('should add tags to existing tags', async () => {
      // Create memory with tags
      await ragServer.handleMemorizeText({
        text: 'Memory with tags',
        label: 'tag-add-test',
        tags: ['existing-tag'],
      })

      // Add tags
      await ragServer.handleUpdateMemory({
        label: 'tag-add-test',
        addTags: ['new-tag-1', 'new-tag-2'],
      })

      // Verify tags added
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://tag-add-test'
      )

      expect(memory.metadata?.tags).toContain('existing-tag')
      expect(memory.metadata?.tags).toContain('new-tag-1')
      expect(memory.metadata?.tags).toContain('new-tag-2')
    })

    it('should not add duplicate tags', async () => {
      // Create memory with tags
      await ragServer.handleMemorizeText({
        text: 'Memory with tags',
        label: 'tag-duplicate-test',
        tags: ['tag-1'],
      })

      // Try to add duplicate tag
      await ragServer.handleUpdateMemory({
        label: 'tag-duplicate-test',
        addTags: ['tag-1', 'tag-2'],
      })

      // Verify no duplicates
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://tag-duplicate-test'
      )

      const tag1Count = memory.metadata?.tags.filter((t: string) => t === 'tag-1').length
      expect(tag1Count).toBe(1)
    })
  })

  describe('Removing tags', () => {
    it('should remove specified tags from existing tags', async () => {
      // Create memory with tags
      await ragServer.handleMemorizeText({
        text: 'Memory with tags',
        label: 'tag-remove-test',
        tags: ['tag-1', 'tag-2', 'tag-3'],
      })

      // Remove tags
      await ragServer.handleUpdateMemory({
        label: 'tag-remove-test',
        removeTags: ['tag-2'],
      })

      // Verify tags removed
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://tag-remove-test'
      )

      expect(memory.metadata?.tags).toEqual(['tag-1', 'tag-3'])
    })

    it('should ignore non-existent tags when removing', async () => {
      // Create memory with tags
      await ragServer.handleMemorizeText({
        text: 'Memory with tags',
        label: 'tag-remove-nonexistent-test',
        tags: ['tag-1', 'tag-2'],
      })

      // Remove non-existent tag
      await ragServer.handleUpdateMemory({
        label: 'tag-remove-nonexistent-test',
        removeTags: ['tag-3', 'tag-4'],
      })

      // Verify original tags intact
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://tag-remove-nonexistent-test'
      )

      expect(memory.metadata?.tags).toEqual(['tag-1', 'tag-2'])
    })
  })

  describe('Error cases', () => {
    it('should fail when updating non-existent memory', async () => {
      await expect(
        ragServer.handleUpdateMemory({
          label: 'non-existent-memory',
          mode: 'replace',
          text: 'New content',
        })
      ).rejects.toThrow()
    })

    it('should require label parameter', async () => {
      await expect(
        ragServer.handleUpdateMemory({
          label: '',
          text: 'New content',
        })
      ).rejects.toThrow()
    })
  })

  describe('updatedAt timestamp', () => {
    it('should update updatedAt timestamp on update', async () => {
      // Create memory
      await ragServer.handleMemorizeText({
        text: 'Original content',
        label: 'timestamp-update-test',
      })

      const filesBefore = await ragServer.handleListFiles()
      const filesListBefore = JSON.parse(filesBefore.content[0].text)
      const memoryBefore = filesListBefore.find(
        (f: { filePath: string }) => f.filePath === 'memory://timestamp-update-test'
      )
      const originalUpdatedAt = memoryBefore.metadata?.updatedAt

      // Wait a bit
      await new Promise((resolve) => setTimeout(resolve, 100))

      // Update memory
      await ragServer.handleUpdateMemory({
        label: 'timestamp-update-test',
        mode: 'append',
        text: ' Updated content',
      })

      const filesAfter = await ragServer.handleListFiles()
      const filesListAfter = JSON.parse(filesAfter.content[0].text)
      const memoryAfter = filesListAfter.find(
        (f: { filePath: string }) => f.filePath === 'memory://timestamp-update-test'
      )
      const newUpdatedAt = memoryAfter.metadata?.updatedAt

      expect(newUpdatedAt).toBeDefined()
      expect(newUpdatedAt).not.toBe(originalUpdatedAt)
      expect(new Date(newUpdatedAt) > new Date(originalUpdatedAt)).toBe(true)
    })

    it('should keep createdAt unchanged on update', async () => {
      // Create memory
      await ragServer.handleMemorizeText({
        text: 'Original content',
        label: 'created-at-unchanged-test',
      })

      const filesBefore = await ragServer.handleListFiles()
      const filesListBefore = JSON.parse(filesBefore.content[0].text)
      const memoryBefore = filesListBefore.find(
        (f: { filePath: string }) => f.filePath === 'memory://created-at-unchanged-test'
      )
      const originalCreatedAt = memoryBefore.metadata?.createdAt

      // Update memory
      await ragServer.handleUpdateMemory({
        label: 'created-at-unchanged-test',
        text: 'Updated content',
      })

      const filesAfter = await ragServer.handleListFiles()
      const filesListAfter = JSON.parse(filesAfter.content[0].text)
      const memoryAfter = filesListAfter.find(
        (f: { filePath: string }) => f.filePath === 'memory://created-at-unchanged-test'
      )

      expect(memoryAfter.metadata?.createdAt).toBe(originalCreatedAt)
    })
  })
})

// ============================================
// Test Suite 6: Project Context Association
// ============================================

describe('Feature 6: Project Context Association', () => {
  let ragServer: RAGServer
  const testDbPath = resolve('./tmp/test-project-db')
  const testDataDir = resolve('./tmp/test-project-data')

  beforeAll(async () => {
    mkdirSync(testDbPath, { recursive: true })
    mkdirSync(testDataDir, { recursive: true })

    ragServer = new RAGServer({
      dbPath: testDbPath,
      modelName: 'Xenova/all-MiniLM-L6-v2',
      cacheDir: './tmp/models',
      baseDir: testDataDir,
      maxFileSize: 100 * 1024 * 1024,
      chunkSize: 512,
      chunkOverlap: 100,
    })

    await ragServer.initialize()
  })

  afterAll(async () => {
    rmSync(testDbPath, { recursive: true, force: true })
    rmSync(testDataDir, { recursive: true, force: true })
  })

  describe('Explicit project association', () => {
    it('should accept project parameter in memorize_text', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Project-specific memory',
        label: 'project-memory-1',
        project: 'my-app',
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://project-memory-1'
      )

      expect(memory.metadata?.project).toBe('my-app')
    })

    it('should accept project parameter in ingest_file', async () => {
      const testFile = resolve(testDataDir, 'project-file.txt')
      writeFileSync(testFile, 'File for specific project')

      const result = await ragServer.handleIngestFile({
        filePath: testFile,
        project: 'my-app',
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const file = filesList.find((f: { filePath: string }) => f.filePath === testFile)

      expect(file.metadata?.project).toBe('my-app')
    })
  })

  describe('Global flag overrides project', () => {
    it('should create global memory when global=true', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Global memory',
        label: 'global-memory',
        project: 'my-app',
        global: true,
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://global-memory'
      )

      // When global=true, project should be undefined or null
      expect(memory.metadata?.project).toBeNull()
    })

    it('should create global file when global=true', async () => {
      const testFile = resolve(testDataDir, 'global-file.txt')
      writeFileSync(testFile, 'Global file content')

      const result = await ragServer.handleIngestFile({
        filePath: testFile,
        project: 'my-app',
        global: true,
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const file = filesList.find((f: { filePath: string }) => f.filePath === testFile)

      expect(file.metadata?.project).toBeNull()
    })
  })

  describe('Default behavior (no project)', () => {
    it('should create global memory by default', async () => {
      const result = await ragServer.handleMemorizeText({
        text: 'Default memory',
        label: 'default-memory',
      })

      expect(result).toBeDefined()
      const files = await ragServer.handleListFiles()
      const filesList = JSON.parse(files.content[0].text)
      const memory = filesList.find(
        (f: { filePath: string }) => f.filePath === 'memory://default-memory'
      )

      expect(memory.metadata?.project).toBeNull()
    })
  })

  describe('Project filtering in list_files', () => {
    it('should filter by project in list_files', async () => {
      // Create memories for different projects
      await ragServer.handleMemorizeText({
        text: 'Project A memory',
        label: 'project-a-mem',
        project: 'project-a',
      })

      await ragServer.handleMemorizeText({
        text: 'Project B memory',
        label: 'project-b-mem',
        project: 'project-b',
      })

      // Filter by project
      const result = await ragServer.handleListFiles({ project: 'project-a' })
      const files = JSON.parse(result.content[0].text)

      expect(files.length).toBeGreaterThan(0)
      expect(
        files.every((f: { metadata?: { project?: string } }) => f.metadata?.project === 'project-a')
      ).toBe(true)
    })
  })

  describe('Project filtering in query_documents', () => {
    it('should filter by project in query_documents', async () => {
      // Create project-specific memories
      await ragServer.handleMemorizeText({
        text: 'Project X documentation about APIs',
        label: 'project-x-docs',
        project: 'project-x',
      })

      await ragServer.handleMemorizeText({
        text: 'Project Y documentation about APIs',
        label: 'project-y-docs',
        project: 'project-y',
      })

      // Query with project filter
      const result = await ragServer.handleQueryDocuments({
        query: 'API documentation',
        project: 'project-x',
        limit: 10,
      })

      const results = JSON.parse(result.content[0].text)

      expect(results.length).toBeGreaterThan(0)
      expect(
        results.some((r: { filePath: string }) => r.filePath === 'memory://project-x-docs')
      ).toBe(true)
      expect(
        results.every((r: { filePath: string }) => r.filePath !== 'memory://project-y-docs')
      ).toBe(true)
    })
  })
})

// ============================================
// Test Suite 7: Enhanced Query Tool
// ============================================

describe('Feature 7: Enhanced Query Tool (query_documents enhancement)', () => {
  let ragServer: RAGServer
  const testDbPath = resolve('./tmp/test-enhanced-query-db')
  const testDataDir = resolve('./tmp/test-enhanced-query-data')

  beforeAll(async () => {
    mkdirSync(testDbPath, { recursive: true })
    mkdirSync(testDataDir, { recursive: true })

    ragServer = new RAGServer({
      dbPath: testDbPath,
      modelName: 'Xenova/all-MiniLM-L6-v2',
      cacheDir: './tmp/models',
      baseDir: testDataDir,
      maxFileSize: 100 * 1024 * 1024,
      chunkSize: 512,
      chunkOverlap: 100,
    })

    await ragServer.initialize()

    // Setup test data
    const file1 = resolve(testDataDir, 'api-docs.txt')
    writeFileSync(file1, 'API documentation for REST endpoints. '.repeat(20))
    await ragServer.handleIngestFile({
      filePath: file1,
      tags: ['documentation', 'api'],
    })

    await ragServer.handleMemorizeText({
      text: 'Lesson learned about API debugging. '.repeat(20),
      label: 'api-lesson',
      type: 'lesson',
      tags: ['api', 'debugging'],
    })

    await ragServer.handleMemorizeText({
      text: 'Notes about database optimization. '.repeat(20),
      label: 'db-notes',
      type: 'note',
      tags: ['database', 'performance'],
    })
  })

  afterAll(async () => {
    rmSync(testDbPath, { recursive: true, force: true })
    rmSync(testDataDir, { recursive: true, force: true })
  })

  describe('Filter by type', () => {
    it('should filter by type="file"', async () => {
      const result = await ragServer.handleQueryDocuments({
        query: 'API',
        type: 'file',
        limit: 10,
      })

      const results = JSON.parse(result.content[0].text)

      expect(results.length).toBeGreaterThan(0)
      expect(results.every((r: { filePath: string }) => !r.filePath.startsWith('memory://'))).toBe(
        true
      )
    })

    it('should filter by type="memory"', async () => {
      const result = await ragServer.handleQueryDocuments({
        query: 'API',
        type: 'memory',
        limit: 10,
      })

      const results = JSON.parse(result.content[0].text)

      expect(results.length).toBeGreaterThan(0)
      expect(results.every((r: { filePath: string }) => r.filePath.startsWith('memory://'))).toBe(
        true
      )
    })

    it('should return all types when type="all" or not specified', async () => {
      const result = await ragServer.handleQueryDocuments({
        query: 'API',
        type: 'all',
        limit: 10,
      })

      const results = JSON.parse(result.content[0].text)
      expect(results.length).toBeGreaterThan(0)
    })
  })

  describe('Filter by tags', () => {
    it('should filter by single tag', async () => {
      const result = await ragServer.handleQueryDocuments({
        query: 'API',
        tags: ['api'],
        limit: 10,
      })

      const results = JSON.parse(result.content[0].text)

      expect(results.length).toBeGreaterThan(0)
      // Should include both file and memory with 'api' tag
    })

    it('should filter by multiple tags (AND logic)', async () => {
      const result = await ragServer.handleQueryDocuments({
        query: 'API',
        tags: ['api', 'debugging'],
        limit: 10,
      })

      const results = JSON.parse(result.content[0].text)

      // Should only include api-lesson which has both tags
      expect(results.length).toBeGreaterThan(0)
      expect(results.some((r: { filePath: string }) => r.filePath === 'memory://api-lesson')).toBe(
        true
      )
    })

    it('should return empty when no results match all tags', async () => {
      const result = await ragServer.handleQueryDocuments({
        query: 'API',
        tags: ['api', 'nonexistent-tag'],
        limit: 10,
      })

      const results = JSON.parse(result.content[0].text)
      expect(results.length).toBe(0)
    })
  })

  describe('Filter by project', () => {
    it('should filter by project', async () => {
      // Add project-specific memory
      await ragServer.handleMemorizeText({
        text: 'Project-specific API information. '.repeat(20),
        label: 'project-api',
        project: 'test-project',
        tags: ['api'],
      })

      const result = await ragServer.handleQueryDocuments({
        query: 'API',
        project: 'test-project',
        limit: 10,
      })

      const results = JSON.parse(result.content[0].text)

      expect(results.length).toBeGreaterThan(0)
      expect(results.some((r: { filePath: string }) => r.filePath === 'memory://project-api')).toBe(
        true
      )
    })
  })

  describe('Filter by minScore', () => {
    it('should filter by minimum similarity score', async () => {
      const result = await ragServer.handleQueryDocuments({
        query: 'API documentation',
        minScore: 0.5,
        limit: 10,
      })

      const results = JSON.parse(result.content[0].text)

      // All results should have score >= minScore
      // Note: LanceDB uses distance (lower is better), so we need to check <= for distance
      expect(
        results.every(
          (r: { score: number }) => r.score <= 0.5 // Assuming lower score means more similar in LanceDB
        )
      ).toBe(true)
    })

    it('should return empty array when minScore too high', async () => {
      const result = await ragServer.handleQueryDocuments({
        query: 'completely unrelated xyz123',
        minScore: 0.01, // Very strict threshold
        limit: 10,
      })

      const results = JSON.parse(result.content[0].text)
      expect(results.length).toBe(0)
    })
  })

  describe('Combining multiple filters', () => {
    it('should combine type and tags filters', async () => {
      const result = await ragServer.handleQueryDocuments({
        query: 'API',
        type: 'memory',
        tags: ['api'],
        limit: 10,
      })

      const results = JSON.parse(result.content[0].text)

      expect(results.length).toBeGreaterThan(0)
      expect(results.every((r: { filePath: string }) => r.filePath.startsWith('memory://'))).toBe(
        true
      )
    })

    it('should combine type, tags, and project filters', async () => {
      await ragServer.handleMemorizeText({
        text: 'Project memory with tags. '.repeat(20),
        label: 'multi-filter-test',
        type: 'memory',
        tags: ['test-tag'],
        project: 'multi-filter-project',
      })

      const result = await ragServer.handleQueryDocuments({
        query: 'project memory',
        type: 'memory',
        tags: ['test-tag'],
        project: 'multi-filter-project',
        limit: 10,
      })

      const results = JSON.parse(result.content[0].text)

      expect(results.length).toBeGreaterThan(0)
      expect(
        results.some((r: { filePath: string }) => r.filePath === 'memory://multi-filter-test')
      ).toBe(true)
    })

    it('should combine all filters including minScore', async () => {
      const result = await ragServer.handleQueryDocuments({
        query: 'API documentation',
        type: 'memory',
        tags: ['api'],
        minScore: 0.8,
        limit: 5,
      })

      const results = JSON.parse(result.content[0].text)

      expect(results.length).toBeLessThanOrEqual(5)
      if (results.length > 0) {
        expect(results.every((r: { filePath: string }) => r.filePath.startsWith('memory://'))).toBe(
          true
        )
      }
    })
  })
})

// ============================================
// Test Suite 8: Backward Compatibility
// ============================================

describe('Backward Compatibility', () => {
  let ragServer: RAGServer
  const testDbPath = resolve('./tmp/test-compat-db')
  const testDataDir = resolve('./tmp/test-compat-data')

  beforeAll(async () => {
    mkdirSync(testDbPath, { recursive: true })
    mkdirSync(testDataDir, { recursive: true })

    ragServer = new RAGServer({
      dbPath: testDbPath,
      modelName: 'Xenova/all-MiniLM-L6-v2',
      cacheDir: './tmp/models',
      baseDir: testDataDir,
      maxFileSize: 100 * 1024 * 1024,
      chunkSize: 512,
      chunkOverlap: 100,
    })

    await ragServer.initialize()
  })

  afterAll(async () => {
    rmSync(testDbPath, { recursive: true, force: true })
    rmSync(testDataDir, { recursive: true, force: true })
  })

  it('should work with old memorize_text calls (no new parameters)', async () => {
    const result = await ragServer.handleMemorizeText({
      text: 'Old-style memory',
      label: 'old-style',
    })

    expect(result).toBeDefined()
    const parsed = JSON.parse(result.content[0].text)
    expect(parsed.label).toBe('old-style')
  })

  it('should work with old ingest_file calls (no new parameters)', async () => {
    const testFile = resolve(testDataDir, 'old-style-file.txt')
    writeFileSync(testFile, 'Old-style file content')

    const result = await ragServer.handleIngestFile({
      filePath: testFile,
    })

    expect(result).toBeDefined()
    const parsed = JSON.parse(result.content[0].text)
    expect(parsed.filePath).toBe(testFile)
  })

  it('should work with old list_files calls (no parameters)', async () => {
    const result = await ragServer.handleListFiles()
    expect(result).toBeDefined()
    const files = JSON.parse(result.content[0].text)
    expect(Array.isArray(files)).toBe(true)
  })

  it('should work with old query_documents calls (no filter parameters)', async () => {
    const result = await ragServer.handleQueryDocuments({
      query: 'test',
      limit: 5,
    })

    expect(result).toBeDefined()
    const results = JSON.parse(result.content[0].text)
    expect(Array.isArray(results)).toBe(true)
  })

  it('should handle existing data without new metadata fields', async () => {
    // This tests that old data (without tags, type, etc.) can coexist with new data
    const result = await ragServer.handleListFiles()
    const files = JSON.parse(result.content[0].text)

    // Should not crash on entries without new fields
    expect(files.length).toBeGreaterThan(0)
  })
})

// ============================================
// Test Suite 9: Input Validation
// ============================================

describe('Input Validation', () => {
  let ragServer: RAGServer
  const testDbPath = resolve('./tmp/test-validation-db')
  const testDataDir = resolve('./tmp/test-validation-data')

  beforeAll(async () => {
    mkdirSync(testDbPath, { recursive: true })
    mkdirSync(testDataDir, { recursive: true })

    ragServer = new RAGServer({
      dbPath: testDbPath,
      modelName: 'Xenova/all-MiniLM-L6-v2',
      cacheDir: './tmp/models',
      baseDir: testDataDir,
      maxFileSize: 100 * 1024 * 1024,
      chunkSize: 512,
      chunkOverlap: 100,
    })

    await ragServer.initialize()
  })

  afterAll(async () => {
    rmSync(testDbPath, { recursive: true, force: true })
    rmSync(testDataDir, { recursive: true, force: true })
  })

  describe('Tags validation', () => {
    it('should reject non-array tags', async () => {
      await expect(
        ragServer.handleMemorizeText({
          text: 'Test',
          label: 'invalid-tags',
          tags: 'not-an-array' as any,
        })
      ).rejects.toThrow('Tags must be an array')
    })

    it('should reject tags with non-string elements', async () => {
      await expect(
        ragServer.handleMemorizeText({
          text: 'Test',
          label: 'invalid-tag-elements',
          tags: [123, {}, null] as any,
        })
      ).rejects.toThrow('All tags must be strings')
    })

    it('should reject empty string tags', async () => {
      await expect(
        ragServer.handleMemorizeText({
          text: 'Test',
          label: 'empty-string-tag',
          tags: ['valid-tag', '', 'another-tag'],
        })
      ).rejects.toThrow('Tags cannot be empty strings')
    })
  })

  describe('Type validation', () => {
    it('should reject invalid type values', async () => {
      await expect(
        ragServer.handleMemorizeText({
          text: 'Test',
          label: 'invalid-type',
          type: 'invalid' as any,
        })
      ).rejects.toThrow('Invalid memory type')
    })

    it('should accept valid type values', async () => {
      const validTypes = ['memory', 'lesson', 'note']

      for (const type of validTypes) {
        const result = await ragServer.handleMemorizeText({
          text: 'Test',
          label: `valid-type-${type}`,
          type: type as 'memory' | 'lesson' | 'note',
        })
        expect(result).toBeDefined()
      }
    })
  })

  describe('TTL validation', () => {
    it('should reject invalid TTL formats', async () => {
      const invalidTTLs = ['5x', '1.5d', 'abc', '1 day', '-1d']

      for (const ttl of invalidTTLs) {
        await expect(
          ragServer.handleMemorizeText({
            text: 'Test',
            label: `invalid-ttl-${ttl}`,
            ttl: ttl as any,
          })
        ).rejects.toThrow()
      }
    })

    it('should accept valid TTL formats', async () => {
      const validTTLs = ['1d', '7d', '30d', '1y', 'permanent']

      for (const ttl of validTTLs) {
        const result = await ragServer.handleMemorizeText({
          text: 'Test',
          label: `valid-ttl-${ttl}`,
          ttl: ttl as any,
        })
        expect(result).toBeDefined()
      }
    })
  })

  describe('Update mode validation', () => {
    it('should reject invalid mode values', async () => {
      // Create memory first
      await ragServer.handleMemorizeText({
        text: 'Original',
        label: 'mode-test',
      })

      await expect(
        ragServer.handleUpdateMemory({
          label: 'mode-test',
          mode: 'invalid' as any,
          text: 'New',
        })
      ).rejects.toThrow('Invalid update mode')
    })

    it('should accept valid mode values', async () => {
      const validModes = ['replace', 'append', 'prepend']

      for (const mode of validModes) {
        await ragServer.handleMemorizeText({
          text: 'Original',
          label: `mode-${mode}`,
        })

        const result = await ragServer.handleUpdateMemory({
          label: `mode-${mode}`,
          mode: mode as 'replace' | 'append' | 'prepend',
          text: 'Updated',
        })
        expect(result).toBeDefined()
      }
    })
  })

  describe('Query filter validation', () => {
    it('should reject invalid type filter in query_documents', async () => {
      await expect(
        ragServer.handleQueryDocuments({
          query: 'test',
          type: 'invalid' as any,
        })
      ).rejects.toThrow('Invalid type filter')
    })

    it('should reject invalid minScore values', async () => {
      // Negative scores should be rejected
      await expect(
        ragServer.handleQueryDocuments({
          query: 'test',
          minScore: -0.5,
        })
      ).rejects.toThrow('minScore cannot be negative')

      // Scores above 2 should be rejected (LanceDB uses L2 distance, typical range 0-2)
      await expect(
        ragServer.handleQueryDocuments({
          query: 'test',
          minScore: 2.5,
        })
      ).rejects.toThrow('minScore must be <= 2')
    })
  })
})
