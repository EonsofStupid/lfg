import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { SupervisionPanel } from './SupervisionPanel'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

test('connects to a local review session and displays human decisions without claiming training', async () => {
  const fetcher = vi.fn(async () => new Response(JSON.stringify({ reviewer: 'reviewer', decisions: { pending: 4, approved: 2, rejected: 1 } })))
  vi.stubGlobal('fetch', fetcher)
  render(<SupervisionPanel />)
  fireEvent.change(screen.getByLabelText('Local review URL'), { target: { value: 'http://127.0.0.1:12345/#secret-token' } })
  fireEvent.click(screen.getByRole('button', { name: 'Connect review workspace' }))
  expect(await screen.findByText('Awaiting review')).toBeInTheDocument()
  expect(screen.getByText('4')).toBeInTheDocument()
  expect(screen.getByText(/not model accuracy/)).toBeInTheDocument()
  expect(fetcher).toHaveBeenCalledWith('http://127.0.0.1:12345/api/status', expect.objectContaining({
    headers: { Authorization: 'Bearer secret-token' }, credentials: 'omit',
  }))
  expect(screen.getByRole('link', { name: 'Open review workspace' })).toHaveAttribute('href', 'http://127.0.0.1:12345/#secret-token')
  fireEvent.click(screen.getByRole('button', { name: 'Disconnect' }))
  expect(screen.getByLabelText('Local review URL')).toHaveValue('')
})

test('rejects a remote review URL before sending any request', () => {
  const fetcher = vi.fn()
  vi.stubGlobal('fetch', fetcher)
  render(<SupervisionPanel />)
  fireEvent.change(screen.getByLabelText('Local review URL'), { target: { value: 'http://example.com:12345/#secret-token' } })
  fireEvent.click(screen.getByRole('button', { name: 'Connect review workspace' }))
  expect(screen.getByRole('alert')).toHaveTextContent('complete local review URL')
  expect(fetcher).not.toHaveBeenCalled()
})
