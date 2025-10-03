#!/usr/bin/env node
/*
 Simple guard: block "\{" or "\}" in TSX/JSX files (invalid JSX escape).
 Scans staged files on pre-commit and fails if patterns are found.
*/
const { execSync } = require('child_process')
const fs = require('fs')
const path = require('path')

function getStagedFiles() {
  const out = execSync('git diff --cached --name-only --diff-filter=ACM', { encoding: 'utf8' })
  return out.split(/\r?\n/).map(s => s.trim()).filter(Boolean)
}

function isTsxLike(file) {
  return /\.(tsx|jsx)$/i.test(file)
}

function scanFile(file) {
  const content = fs.readFileSync(file, 'utf8')
  const lines = content.split(/\r?\n/)
  const issues = []
  const re = /\\[{}]/g // matches \{ or \}
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    let m
    while ((m = re.exec(line)) !== null) {
      issues.push({ line: i + 1, col: m.index + 1, match: m[0] })
    }
  }
  return issues
}

function main() {
  const staged = getStagedFiles().filter(isTsxLike)
  let hadError = false
  for (const file of staged) {
    if (!fs.existsSync(file)) continue
    const issues = scanFile(file)
    if (issues.length) {
      hadError = true
      console.error(`\n[pre-commit] Invalid JSX escape in ${file}:`)
      for (const it of issues) {
        console.error(`  ${file}:${it.line}:${it.col} contains '${it.match}' (remove the backslash)`) 
      }
    }
  }
  if (hadError) {
    console.error('\nFix: In JSX/TSX use {...} expressions directly, do NOT write \\{ or \\}.')
    process.exit(1)
  }
}

try { main() } catch (e) {
  console.error('[pre-commit] guard failed to run:', e?.message || e)
  // do not block commit on tool failure — comment the next line to enforce
  // process.exit(1)
}

