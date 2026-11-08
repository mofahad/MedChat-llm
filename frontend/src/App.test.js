import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from './App';
beforeEach(() => { global.fetch = jest.fn(); });
afterEach(() => { jest.restoreAllMocks(); });

test('submits to the local backend and displays cited excerpts', async () => {
  global.fetch.mockResolvedValue({ ok: true, json: async () => ({ reply: '[model] BioMistral project note.', sources: [{ id: 'model', source: 'Archived notebook' }] }) });
  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: 'Get Started' }));
  fireEvent.change(screen.getByLabelText('Question'), { target: { value: 'What model is used?' } });
  fireEvent.click(screen.getByRole('button', { name: 'Send question' }));
  expect(await screen.findByText('[model] BioMistral project note.')).toBeInTheDocument();
  expect(screen.getByText('[model] Archived notebook')).toBeInTheDocument();
  expect(global.fetch).toHaveBeenCalledWith('http://127.0.0.1:5005/chat', expect.objectContaining({ method: 'POST', body: JSON.stringify({ question: 'What model is used?' }) }));
});
test('shows an actionable failure and allows retry', async () => {
  global.fetch.mockResolvedValue({ ok: false, status: 503, json: async () => ({ error: 'Unavailable' }) });
  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: 'Get Started' }));
  fireEvent.change(screen.getByLabelText('Question'), { target: { value: 'model' } });
  fireEvent.click(screen.getByRole('button', { name: 'Send question' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Check that the local backend is running');
  await waitFor(() => expect(screen.getByRole('button', { name: 'Clear chat' })).toBeEnabled());
  fireEvent.change(screen.getByLabelText('Question'), { target: { value: 'retry' } });
  expect(screen.getByRole('button', { name: 'Send question' })).toBeEnabled();
});
test('clears messages without a stale remote clear endpoint', () => {
  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: 'Get Started' }));
  fireEvent.click(screen.getByRole('button', { name: 'Clear chat' }));
  expect(screen.getByRole('button', { name: 'Get Started' })).toBeInTheDocument();
  expect(global.fetch).not.toHaveBeenCalled();
});
