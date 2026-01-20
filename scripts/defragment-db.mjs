#!/usr/bin/env node
/**
 * Database defragmentation script for mcp-local-rag
 *
 * Rebuilds the LanceDB table to consolidate all data and indices
 * into a single optimized structure.
 *
 * Usage: node scripts/defragment-db.mjs [--db-path /path/to/lancedb]
 */

import { connect, Index } from '@lancedb/lancedb';
import * as arrow from 'apache-arrow';
import { parseArgs } from 'node:util';

const DEFAULT_DB_PATH = process.env.BASE_DIR
  ? `${process.env.BASE_DIR}/lancedb`
  : `${process.env.HOME}/lancedb`;

const TABLE_NAME = 'chunks';

// Parse command line arguments
const { values } = parseArgs({
  options: {
    'db-path': { type: 'string', default: DEFAULT_DB_PATH },
    'dry-run': { type: 'boolean', default: false },
  },
});

const DB_PATH = values['db-path'];
const DRY_RUN = values['dry-run'];

function getSchema() {
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
  ]);
}

function normalizeRecord(record) {
  const rawMetadata = record.metadata;

  // Normalize tags
  let tags = [];
  if (rawMetadata?.tags) {
    if (Array.isArray(rawMetadata.tags)) {
      tags = [...rawMetadata.tags];
    } else if (typeof rawMetadata.tags.toArray === 'function') {
      tags = rawMetadata.tags.toArray();
    }
  }

  // Normalize vector
  let vector = [];
  if (record.vector) {
    if (Array.isArray(record.vector)) {
      vector = [...record.vector];
    } else if (record.vector instanceof Float32Array) {
      vector = Array.from(record.vector);
    } else if (typeof record.vector.toArray === 'function') {
      vector = Array.from(record.vector.toArray());
    } else if (typeof record.vector === 'object') {
      vector = Array.from(record.vector);
    }
  }

  return {
    id: record.id,
    filePath: record.filePath,
    chunkIndex: record.chunkIndex,
    text: record.text,
    vector,
    metadata: {
      fileName: rawMetadata?.fileName || 'unknown',
      fileSize: rawMetadata?.fileSize || 0,
      fileType: rawMetadata?.fileType || 'unknown',
      language: rawMetadata?.language || null,
      memoryType: rawMetadata?.memoryType || null,
      tags,
      project: rawMetadata?.project || null,
      expiresAt: rawMetadata?.expiresAt || null,
      createdAt: rawMetadata?.createdAt || record.timestamp,
      updatedAt: rawMetadata?.updatedAt || record.timestamp,
    },
    timestamp: record.timestamp,
  };
}

async function main() {
  console.log(`Database path: ${DB_PATH}`);
  console.log(`Dry run: ${DRY_RUN}`);
  console.log('');

  // Connect
  const db = await connect(DB_PATH);
  const tableNames = await db.tableNames();

  if (!tableNames.includes(TABLE_NAME)) {
    console.log(`Table '${TABLE_NAME}' not found. Nothing to defragment.`);
    return;
  }

  const table = await db.openTable(TABLE_NAME);

  // Get stats before
  const rowCount = await table.countRows();
  console.log(`Current row count: ${rowCount}`);

  // Read all data
  console.log('Reading all records...');
  const allRecords = await table.query().toArray();
  console.log(`Read ${allRecords.length} records`);

  // Normalize records
  console.log('Normalizing records...');
  const normalizedRecords = allRecords.map(normalizeRecord);

  if (DRY_RUN) {
    console.log('');
    console.log('DRY RUN - would perform:');
    console.log(`  1. Drop table '${TABLE_NAME}'`);
    console.log(`  2. Create new table with ${normalizedRecords.length} records`);
    console.log('  3. Create FTS index on text column');
    return;
  }

  // Drop old table
  console.log('Dropping old table...');
  await db.dropTable(TABLE_NAME);

  // Create new table with all data at once
  console.log('Creating new table with consolidated data...');
  const schema = getSchema();
  const newTable = await db.createTable(TABLE_NAME, normalizedRecords, { schema });

  // Create FTS index
  console.log('Creating FTS index...');
  await newTable.createIndex('text', {
    config: Index.fts(),
  });

  // Verify
  const newRowCount = await newTable.countRows();
  console.log('');
  console.log('=== Defragmentation Complete ===');
  console.log(`Rows: ${rowCount} -> ${newRowCount}`);

  if (newRowCount !== rowCount) {
    console.error('WARNING: Row count mismatch!');
  }
}

main().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
