import { fireEvent, render, screen } from '@testing-library/react';
import App from './App';

jest.mock('react-type-animation', () => ({
  TypeAnimation: ({ sequence }) => <span>{typeof sequence[0] === 'string' ? sequence[0] : ''}</span>
}));

beforeEach(() => {
  global.fetch = jest.fn();
});
afterEach(() => jest.restoreAllMocks());

function startAndAsk() {
  fireEvent.click(screen.getByText(/Get Started/i));
  const input = screen.getByRole('textbox');
  fireEvent.change(input, { target: { value: 'Test source question' } });
  fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
}

test('shows backend source response', async () => {
  global.fetch.mockResolvedValue({ ok: true, json: async () => ({ reply: 'Retrieved source records: fixture' }) });
  render(<App />);
  startAndAsk();
  expect(await screen.findByText('Retrieved source records: fixture')).toBeInTheDocument();
  expect(JSON.parse(global.fetch.mock.calls[0][1].body)).toEqual({ question: 'Test source question' });
});

test('shows backend errors instead of a silent failed chat', async () => {
  global.fetch.mockResolvedValue({ ok: false, json: async () => ({ error: 'Dataset missing' }) });
  render(<App />);
  startAndAsk();
  expect(await screen.findByText('Request failed: Dataset missing')).toBeInTheDocument();
});

test('shows a network failure', async () => {
  global.fetch.mockRejectedValue(new Error('Connection unavailable'));
  render(<App />);
  startAndAsk();
  expect(await screen.findByText('Request failed: Connection unavailable')).toBeInTheDocument();
});
