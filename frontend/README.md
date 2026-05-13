# RAG App Frontend (React + Vite)

## Run

Install dependencies:

```bash
cd frontend
npm install
```

Start development server:

```bash
npm run dev
```

## API Base URL

By default frontend calls:

- `/api`

Override with env variable:

- `VITE_API_BASE_URL`

Example `.env` inside `frontend`:

```env
VITE_API_BASE_URL=/api
```

In development, Vite proxies `/api` to `http://127.0.0.1:8000`.
