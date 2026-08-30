import { test } from 'node:test';
import assert from 'node:assert/strict';
import { check } from '../src/index.ts';
test('cross-workspace typescript import resolves', () => {
  assert.equal(check(), 'cross-package-ts-import-works');
});
