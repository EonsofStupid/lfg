import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { BoundaryPanel, readProjectPreview } from './BoundaryPanel'
import type { BoundaryCatalog } from './types'

const catalog: BoundaryCatalog = { digest: 'catalog-digest', catalog: {
  schemaVersion: 'lfg.boundary-question-banks.v1', revision: 'draft', sourceNotice: 'Draft',
  banks: [{ id: 'core', title: 'Core questions', status: 'draft', questions: [
    { id: 'one', section: 'A', number: 1, question: 'Which module?', sourceTerm: 'module', binding: 'module inventory' },
    { id: 'two', section: 'A', number: 2, question: 'Who decides?', sourceTerm: 'decision', binding: null }
  ] }, { id: 'other', title: 'Other questions', status: 'draft', questions: [
    { id: 'three', section: 'B', number: 1, question: 'What is missing?', sourceTerm: 'test', binding: 'tests' }
  ] }]
} }
const preview = { schemaVersion: 'lfg.project-attunement-preview.v1', configRevision: 'local', configSha256: 'abc',
  sources: [{ binding: 'module inventory', path: 'src', status: 'collected' }], preview: {
    schemaVersion: 'lfg.boundary-attunement-preview.v1', catalogDigest: 'catalog-digest', sourceApp: 'project', questions: [
      { bankId: 'core', question: { id: 'one', question: 'This supplied text must never replace the catalog' }, bindingStatus: 'supplied', artifact: ['src/main.rs'], artifactSource: 'app_artifact' },
      { bankId: 'other', question: { id: 'three' }, bindingStatus: 'missing', artifact: null, artifactSource: null }
    ]
  }
}
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

test('shows catalog questions and locally imported project evidence with missing and unselected states', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(catalog))))
  render(<BoundaryPanel maxFileBytes={10000} />)
  fireEvent.click(screen.getByRole('button', { name: 'Open questions' }))
  expect(await screen.findByText(/Which module/)).toBeInTheDocument()
  const input = screen.getByLabelText('Project preview JSON')
  fireEvent.change(input, { target: { files: [new File([JSON.stringify(preview)], 'project.json', { type: 'application/json' })] } })
  expect(await screen.findByText('Imported evidence (app_artifact)')).toBeInTheDocument()
  expect(screen.getByText(/src\/main.rs/)).toBeInTheDocument()
  expect(screen.getByText('Not included in this project preview')).toBeInTheDocument()
  expect(screen.queryByText(/This supplied text/)).not.toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('Question bank'), { target: { value: 'other' } })
  expect(screen.getByText('Required artifact missing from preview')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Clear evidence' }))
  expect(screen.queryByText('Required artifact missing from preview')).not.toBeInTheDocument()
  expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1)
})

test('rejects mismatched catalogs, duplicated and mismatched question identities', () => {
  expect(() => readProjectPreview(JSON.stringify({ ...preview, preview: { ...preview.preview, catalogDigest: 'different' } }), catalog)).toThrow(/different question catalog/)
  expect(() => readProjectPreview(JSON.stringify({ ...preview, preview: { ...preview.preview, questions: [preview.preview.questions[0], preview.preview.questions[0]] } }), catalog)).toThrow(/Invalid question evidence/)
  expect(() => readProjectPreview(JSON.stringify({ ...preview, preview: { ...preview.preview, questions: [{ ...preview.preview.questions[0], bankId: 'other' }] } }), catalog)).toThrow(/Invalid question evidence/)
})

test('rejects oversized project imports without submitting their contents', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(catalog))))
  render(<BoundaryPanel maxFileBytes={4} />)
  fireEvent.click(screen.getByRole('button', { name: 'Open questions' }))
  await screen.findByLabelText('Project preview JSON')
  fireEvent.change(screen.getByLabelText('Project preview JSON'), { target: { files: [new File(['12345'], 'large.json')] } })
  await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('exceeds the configured browser import limit'))
  expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1)
})
