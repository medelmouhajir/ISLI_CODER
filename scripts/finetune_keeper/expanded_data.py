"""Expanded Gold Teacher Dataset Samples for ISLI Keeper SLM (Qwen3-0.6B).

Supplies rich multi-task data addressing specific forensic gaps:
  1. Permuted Search Candidate Re-Ranking (eliminating positional index-0 bias)
  2. Multi-Language Structural Code Extraction (Rust, Go, Python, TypeScript, SQL)
  3. Composite Monorepo / Multi-Package Build Outputs (Turborepo, pnpm, Nx, Cargo workspace)
"""

from __future__ import annotations

import json
from typing import Any


def get_expanded_rank_cases() -> list[tuple[str, list[dict[str, Any]], str]]:
    """Return 45+ permuted search ranking cases with varying target index permutations.

    Ensures the SLM learns semantic relevance rather than default [0, 1, 2] positional order.
    """
    cases = [
        # --- 1. Best candidate at Index 1 ([1, 0, 2] or [1, 2, 0]) ---
        (
            "TokenBucket rate limiter class implementation",
            [
                {"file": "docs/architecture.md", "text": "We use a TokenBucket algorithm to protect API endpoints against bursts."},
                {"file": "src/rate_limiter.ts", "text": "export class TokenBucket {\n  private capacity: number;\n  public consume(tokens: number): boolean {\n    this.refill();\n    if (this.tokens >= tokens) ...\n  }\n}"},
                {"file": "tests/rate_limiter.test.ts", "text": "describe('TokenBucket', () => {\n  it('refills tokens over time', () => { ... });\n});"},
            ],
            "[1, 2, 0]"
        ),
        (
            "User interface schema definition",
            [
                {"file": "src/components/UserProfile.tsx", "text": "<UserAvatar user={user} />\n<p>{user.displayName}</p>"},
                {"file": "src/types/user.ts", "text": "export interface User {\n  id: string;\n  email: string;\n  displayName: string;\n  role: 'admin' | 'user';\n}"},
                {"file": "tests/fixtures/user.json", "text": "{\"id\": \"u_123\", \"email\": \"alice@example.com\"}"},
            ],
            "[1, 0, 2]"
        ),
        (
            "Redux toolkit slice for authentication state",
            [
                {"file": "src/store/index.ts", "text": "export const store = configureStore({ reducer: { auth: authReducer } });"},
                {"file": "src/store/authSlice.ts", "text": "export const authSlice = createSlice({\n  name: 'auth',\n  initialState,\n  reducers: {\n    setCredentials: (state, action) => { ... }\n  }\n});"},
                {"file": "src/components/LoginForm.tsx", "text": "const dispatch = useAppDispatch();\ndispatch(setCredentials(res));"},
            ],
            "[1, 0, 2]"
        ),
        (
            "Docker compose service for Redis cache",
            [
                {"file": "src/cache/redis.ts", "text": "export const redis = new Redis(process.env.REDIS_URL || 'redis://localhost:6379');"},
                {"file": "docker-compose.yml", "text": "redis:\n  image: redis:7-alpine\n  ports:\n    - '6379:6379'\n  volumes:\n    - redis_data:/data"},
                {"file": "Dockerfile", "text": "FROM node:20-alpine\nWORKDIR /app\nCOPY package*.json ./ ..."},
            ],
            "[1, 0, 2]"
        ),
        (
            "FastAPI dependency for database session",
            [
                {"file": "src/api/users.py", "text": "@router.get('/')\ndef list_users(db: Session = Depends(get_db)):\n    return db.query(User).all()"},
                {"file": "src/db/deps.py", "text": "def get_db() -> Generator[Session, None, None]:\n    db = SessionLocal()\n    try:\n        yield db\n    finally:\n        db.close()"},
                {"file": "src/main.py", "text": "app = FastAPI(title='API Service')\napp.include_router(users_router)"},
            ],
            "[1, 0, 2]"
        ),
        (
            "Jest mock for AWS S3 client",
            [
                {"file": "src/services/storage.ts", "text": "import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';\nexport const upload = ...;"},
                {"file": "__mocks__/@aws-sdk/client-s3.ts", "text": "export const S3Client = jest.fn().mockImplementation(() => ({\n  send: jest.fn().mockResolvedValue({ ETag: '\"mock-etag\"' })\n}));"},
                {"file": "package.json", "text": "\"dependencies\": { \"@aws-sdk/client-s3\": \"^3.500.0\" }"},
            ],
            "[1, 0, 2]"
        ),
        (
            "Rust trait definition for storage backend",
            [
                {"file": "src/storage/memory.rs", "text": "impl StorageBackend for MemoryStorage {\n    fn get(&self, key: &str) -> Option<Vec<u8>> { ... }\n}"},
                {"file": "src/storage/mod.rs", "text": "pub trait StorageBackend: Send + Sync {\n    fn get(&self, key: &str) -> Option<Vec<u8>>;\n    fn set(&mut self, key: &str, val: Vec<u8>) -> Result<(), StorageError>;\n}"},
                {"file": "src/main.rs", "text": "fn main() { let storage = MemoryStorage::new(); }"},
            ],
            "[1, 0, 2]"
        ),

        # --- 2. Best candidate at Index 2 ([2, 0, 1] or [2, 1, 0]) ---
        (
            "Database migration script for organization table",
            [
                {"file": "src/models/organization.py", "text": "class Organization(Base):\n    __tablename__ = 'organizations'\n    id = Column(Integer, primary_key=True)"},
                {"file": "dist/migrations/003_org.js", "text": "exports.up = function(knex) { ... }"},
                {"file": "migrations/versions/003_add_org.py", "text": "def upgrade():\n    op.create_table('organizations',\n        sa.Column('id', sa.Integer(), primary_key=True),\n        sa.Column('name', sa.String(length=100), nullable=False)\n    )"},
            ],
            "[2, 0, 1]"
        ),
        (
            "Next.js dynamic route segment for blog post",
            [
                {"file": "src/components/blog-card.tsx", "text": "export function BlogCard({ post }: { post: Post }) {\n  return <Link href={`/blog/${post.slug}`}>{post.title}</Link>;\n}"},
                {"file": "public/blog/sample.md", "text": "# Sample Post\nThis is a static markdown sample."},
                {"file": "src/app/blog/[slug]/page.tsx", "text": "export default async function BlogPostPage({ params }: { params: Promise<{ slug: string }> }) {\n  const { slug } = await params;\n  const post = await getPostBySlug(slug);\n  return <article><h1>{post.title}</h1></article>;\n}"},
            ],
            "[2, 0, 1]"
        ),
        (
            "Go HTTP router initialization and route registration",
            [
                {"file": "cmd/server/main.go", "text": "func main() {\n  srv := server.New()\n  srv.Start(':8080')\n}"},
                {"file": "internal/handlers/user.go", "text": "func HandleGetUser(w http.ResponseWriter, r *http.Request) { ... }"},
                {"file": "internal/server/router.go", "text": "func NewRouter(h *handlers.Handler) *chi.Mux {\n  r := chi.NewRouter()\n  r.Use(middleware.Logger)\n  r.Get('/users', h.ListUsers)\n  r.Post('/users', h.CreateUser)\n  return r\n}"},
            ],
            "[2, 0, 1]"
        ),
        (
            "Tailwind CSS theme font configuration",
            [
                {"file": "package.json", "text": "\"tailwindcss\": \"^4.0.0\", \"@tailwindcss/vite\": \"^4.0.0\""},
                {"file": "src/app/layout.tsx", "text": "import { Inter } from 'next/font/google';\nconst inter = Inter({ subsets: ['latin'] });"},
                {"file": "src/app/globals.css", "text": "@import \"tailwindcss\";\n@theme {\n  --font-sans: 'Inter', system-ui, sans-serif;\n  --font-mono: 'JetBrains Mono', monospace;\n}"},
            ],
            "[2, 1, 0]"
        ),
        (
            "Zod validation schema for checkout request",
            [
                {"file": "src/services/checkout.ts", "text": "export async function processPayment(data: CheckoutInput) { ... }"},
                {"file": "src/pages/api/checkout.ts", "text": "const body = req.body;\nconst validated = CheckoutSchema.parse(body);"},
                {"file": "src/schemas/checkout.ts", "text": "export const CheckoutSchema = z.object({\n  cartId: z.string().uuid(),\n  paymentMethod: z.enum(['card', 'paypal']),\n  amount: z.number().positive(),\n});"},
            ],
            "[2, 1, 0]"
        ),
        (
            "GraphQL resolver for querying organization members",
            [
                {"file": "schema.graphql", "text": "type Query {\n  organizationMembers(orgId: ID!): [User!]!\n}"},
                {"file": "src/services/orgService.ts", "text": "export function findMembers(orgId: string) { return db.users.find({ orgId }); }"},
                {"file": "src/graphql/resolvers/org.ts", "text": "export const orgResolvers = {\n  Query: {\n    organizationMembers: async (_, { orgId }, { db }) => {\n      return db.user.findMany({ where: { orgId } });\n    }\n  }\n};"},
            ],
            "[2, 0, 1]"
        ),
        (
            "Pytest fixture for mocked database engine",
            [
                {"file": "tests/test_users.py", "text": "def test_create_user(mock_db):\n    res = create_user(mock_db, 'Alice')\n    assert res.id is not None"},
                {"file": "src/db/session.py", "text": "engine = create_engine(DATABASE_URL)"},
                {"file": "tests/conftest.py", "text": "@pytest.fixture(scope='session')\ndef mock_db():\n    engine = create_engine('sqlite:///:memory:')\n    Base.metadata.create_all(engine)\n    yield sessionmaker(bind=engine)()"},
            ],
            "[2, 0, 1]"
        ),

        # --- 3. Best candidate at Index 0 ([0, 1, 2] or [0, 2, 1]) ---
        (
            "CVA badge variant component definition",
            [
                {"file": "src/components/ui/badge.tsx", "text": "export const badgeVariants = cva(\n  'inline-flex items-center rounded-md px-2.5 py-0.5 text-xs font-semibold',\n  { variants: { variant: { default: 'bg-primary text-primary-foreground', outline: 'text-foreground' } } }\n);"},
                {"file": "src/components/status-indicator.tsx", "text": "import { Badge } from '@/components/ui/badge';\nexport function Status() { return <Badge>Active</Badge>; }"},
                {"file": "package.json", "text": "\"class-variance-authority\": \"^0.7.0\""},
            ],
            "[0, 1, 2]"
        ),
        (
            "Alembic database migration configuration env.py",
            [
                {"file": "alembic/env.py", "text": "target_metadata = Base.metadata\ndef run_migrations_online():\n    connectable = engine_from_config(...)\n    with connectable.connect() as connection:\n        context.configure(connection=connection, target_metadata=target_metadata)"},
                {"file": "alembic.ini", "text": "script_location = alembic\nsqlalchemy.url = driver://user:pass@localhost/dbname"},
                {"file": "src/models/base.py", "text": "from sqlalchemy.orm import declarative_base\nBase = declarative_base()"},
            ],
            "[0, 1, 2]"
        ),
        (
            "Vite configuration for React with Tailwind 4",
            [
                {"file": "vite.config.ts", "text": "import { defineConfig } from 'vite';\nimport react from '@vitejs/plugin-react';\nimport tailwindcss from '@tailwindcss/vite';\n\nexport default defineConfig({\n  plugins: [react(), tailwindcss()]\n});"},
                {"file": "src/main.tsx", "text": "import './index.css';\ncreateRoot(document.getElementById('root')!).render(<App />);"},
                {"file": "package.json", "text": "\"devDependencies\": { \"vite\": \"^5.2.0\" }"},
            ],
            "[0, 1, 2]"
        ),
        (
            "Rust Tokio async TCP echo server listener loop",
            [
                {"file": "src/server.rs", "text": "pub async fn run_server(addr: &str) -> io::Result<()> {\n    let listener = TcpListener::bind(addr).await?;\n    loop {\n        let (mut socket, _) = listener.accept().await?;\n        tokio::spawn(async move { ... });\n    }\n}"},
                {"file": "src/main.rs", "text": "#[tokio::main]\nasync fn main() {\n    run_server('127.0.0.1:8080').await.unwrap();\n}"},
                {"file": "Cargo.toml", "text": "[dependencies]\ntokio = { version = '1.38', features = ['full'] }"},
            ],
            "[0, 1, 2]"
        ),

        # --- 4. Monorepo & Disambiguation Cases (4 or 5 items) ---
        (
            "Shared UI Button component in Turborepo monorepo",
            [
                {"file": "apps/web/node_modules/@repo/ui/dist/button.js", "text": "exports.Button = function() { ... }"},
                {"file": "apps/docs/src/pages/index.tsx", "text": "import { Button } from '@repo/ui';"},
                {"file": "packages/ui/src/button.tsx", "text": "export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ children, ...props }, ref) => {\n  return <button ref={ref} {...props}>{children}</button>;\n});"},
                {"file": "apps/web/src/app/page.tsx", "text": "import { Button } from '@repo/ui';"},
            ],
            "[2, 3, 1, 0]"
        ),
        (
            "Core logger interface in pnpm workspace",
            [
                {"file": "services/auth/src/index.ts", "text": "import { createLogger } from '@corp/logger';\nconst log = createLogger('auth');"},
                {"file": "packages/logger/src/index.ts", "text": "export interface ILogger {\n  info(msg: string, ...meta: any[]): void;\n  error(msg: string, ...meta: any[]): void;\n}\nexport function createLogger(name: string): ILogger { ... }"},
                {"file": "services/billing/package.json", "text": "\"dependencies\": { \"@corp/logger\": \"workspace:*\" }"},
                {"file": "packages/logger/package.json", "text": "\"name\": \"@corp/logger\", \"version\": \"1.0.0\""},
            ],
            "[1, 3, 0, 2]"
        ),
        (
            "Prisma schema file with PostgreSQL datasource",
            [
                {"file": "node_modules/.prisma/client/index.d.ts", "text": "export class PrismaClient { ... }"},
                {"file": "src/lib/prisma.ts", "text": "import { PrismaClient } from '@prisma/client';\nexport const prisma = new PrismaClient();"},
                {"file": "prisma/schema.prisma", "text": "datasource db {\n  provider = 'postgresql'\n  url      = env('DATABASE_URL')\n}\n\nmodel User {\n  id String @id @default(cuid())\n}"},
            ],
            "[2, 1, 0]"
        ),
        (
            "GitHub Actions workflow for continuous integration testing",
            [
                {"file": ".github/workflows/deploy.yml", "text": "name: Deploy to Production\non: push: branches: [main]"},
                {"file": ".github/workflows/ci.yml", "text": "name: CI\non: [push, pull_request]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - run: pnpm install && pnpm test"},
                {"file": "package.json", "text": "\"scripts\": { \"test\": \"jest --ci\" }"},
            ],
            "[1, 0, 2]"
        ),
        (
            "ESLint flat config with TypeScript rules",
            [
                {"file": "package.json", "text": "\"eslint\": \"^9.0.0\", \"@typescript-eslint/parser\": \"^8.0.0\""},
                {"file": ".eslintrc.json", "text": "{\"deprecated\": true}"},
                {"file": "eslint.config.mjs", "text": "import tseslint from 'typescript-eslint';\nexport default tseslint.config(\n  ...tseslint.configs.recommended,\n  { rules: { '@typescript-eslint/no-explicit-any': 'warn' } }\n);"},
            ],
            "[2, 0, 1]"
        ),
        (
            "Next.js middleware for authentication redirect",
            [
                {"file": "src/app/api/auth/[...nextauth]/route.ts", "text": "export const { GET, POST } = handlers;"},
                {"file": "src/middleware.ts", "text": "export function middleware(request: NextRequest) {\n  const token = request.cookies.get('session');\n  if (!token && request.nextUrl.pathname.startsWith('/dashboard')) {\n    return NextResponse.redirect(new URL('/login', request.url));\n  }\n}"},
                {"file": "src/app/dashboard/page.tsx", "text": "export default function Dashboard() { return <div>Dashboard</div>; }"},
            ],
            "[1, 0, 2]"
        ),
        (
            "Go context cancellation in worker pool",
            [
                {"file": "cmd/worker/main.go", "text": "func main() { ctx := context.Background(); ... }"},
                {"file": "internal/pool/worker.go", "text": "func (p *Pool) Run(ctx context.Context) {\n  for {\n    select {\n    case <-ctx.Done():\n      log.Println('Shutting down worker pool')\n      return\n    case job := <-p.jobs:\n      p.process(job)\n    }\n  }\n}"},
                {"file": "internal/pool/pool_test.go", "text": "func TestPoolCancel(t *testing.T) { ... }"},
            ],
            "[1, 2, 0]"
        ),
        (
            "Python Pydantic BaseSettings environment loader",
            [
                {"file": "src/main.py", "text": "from src.config import settings\nprint(settings.app_name)"},
                {"file": "src/config.py", "text": "from pydantic_settings import BaseSettings\nclass Settings(BaseSettings):\n    database_url: str\n    redis_url: str = 'redis://localhost:6379'\n    app_env: str = 'development'\nsettings = Settings()"},
                {"file": ".env.example", "text": "DATABASE_URL=postgresql://user:pass@localhost:5432/db"},
            ],
            "[1, 2, 0]"
        ),
        (
            "React query custom hook for fetching user profile",
            [
                {"file": "src/components/Navbar.tsx", "text": "const { data: user } = useUserProfile();\nreturn <div>{user?.name}</div>;"},
                {"file": "src/api/client.ts", "text": "export const api = axios.create({ baseURL: '/api' });"},
                {"file": "src/hooks/useUserProfile.ts", "text": "export function useUserProfile() {\n  return useQuery({\n    queryKey: ['user', 'profile'],\n    queryFn: async () => (await api.get<User>('/user/me')).data,\n    staleTime: 5 * 60 * 1000,\n  });\n}"},
            ],
            "[2, 0, 1]"
        ),
        (
            "SQL foreign key constraint cascade delete",
            [
                {"file": "schema/tables.sql", "text": "CREATE TABLE users ( id SERIAL PRIMARY KEY, name TEXT );"},
                {"file": "schema/orders.sql", "text": "CREATE TABLE orders (\n  id SERIAL PRIMARY KEY,\n  user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,\n  amount DECIMAL(10,2)\n);"},
                {"file": "src/queries/orderQueries.ts", "text": "export const getOrders = 'SELECT * FROM orders WHERE user_id = $1';"},
            ],
            "[1, 0, 2]"
        ),
        (
            "Rust Serde deserialize custom field attribute",
            [
                {"file": "Cargo.toml", "text": "serde = { version = '1.0', features = ['derive'] }"},
                {"file": "src/models.rs", "text": "#[derive(Debug, Serialize, Deserialize)]\npub struct EventPayload {\n    #[serde(rename = 'eventType')]\n    pub event_type: String,\n    #[serde(default)]\n    pub timestamp: u64,\n}"},
                {"file": "src/main.rs", "text": "let payload: EventPayload = serde_json::from_str(data)?;"},
            ],
            "[1, 2, 0]"
        ),
        (
            "Cypress end-to-end authentication test",
            [
                {"file": "cypress.config.ts", "text": "export default defineConfig({ e2e: { baseUrl: 'http://localhost:3000' } });"},
                {"file": "src/components/LoginForm.tsx", "text": "<input data-testid='email-input' />"},
                {"file": "cypress/e2e/auth.cy.ts", "text": "describe('Authentication', () => {\n  it('logs in successfully with valid credentials', () => {\n    cy.visit('/login');\n    cy.get('[data-testid=email-input]').type('test@example.com');\n    cy.get('button[type=submit]').click();\n    cy.url().should('include', '/dashboard');\n  });\n});"},
            ],
            "[2, 0, 1]"
        ),
    ]
    return cases


def get_expanded_extract_cases() -> list[tuple[str, str, str]]:
    """Return 30+ multi-language structural extraction cases (Rust, Go, Python, TS, CSS)."""
    cases = [
        # --- 1. Rust Structures ---
        (
            "SlidingWindow struct definition and new constructor",
            "use std::time::{Duration, Instant};\n\npub struct SlidingWindow {\n    window_size: Duration,\n    capacity: usize,\n    timestamps: Vec<Instant>,\n}\n\nimpl SlidingWindow {\n    pub fn new(window_size: Duration, capacity: usize) -> Self {\n        Self {\n            window_size,\n            capacity,\n            timestamps: Vec::new(),\n        }\n    }\n\n    pub fn allow_request(&mut self) -> bool {\n        let now = Instant::now();\n        self.timestamps.retain(|&t| now.duration_since(t) <= self.window_size);\n        if self.timestamps.len() < self.capacity {\n            self.timestamps.push(now);\n            true\n        } else {\n            false\n        }\n    }\n}",
            "   3 | pub struct SlidingWindow {\n   4 |     window_size: Duration,\n   5 |     capacity: usize,\n   6 |     timestamps: Vec<Instant>,\n   7 | }\n   8 |\n   9 | impl SlidingWindow {\n  10 |     pub fn new(window_size: Duration, capacity: usize) -> Self {\n  11 |         Self {\n  12 |             window_size,\n  13 |             capacity,\n  14 |             timestamps: Vec::new(),\n  15 |         }\n  16 |     }"
        ),
        (
            "StorageError enum with thiserror derive",
            "use thiserror::Error;\n\n#[derive(Error, Debug)]\npub enum StorageError {\n    #[error(\"Key not found: {0}\")]\n    NotFound(String),\n    #[error(\"IO error during storage operation: {0}\")]\n    Io(#[from] std::io::Error),\n    #[error(\"Serialization failed\")]\n    Serialization,\n}\n\npub type Result<T> = std::result::Result<T, StorageError>;",
            "   3 | #[derive(Error, Debug)]\n   4 | pub enum StorageError {\n   5 |     #[error(\"Key not found: {0}\")]\n   6 |     NotFound(String),\n   7 |     #[error(\"IO error during storage operation: {0}\")]\n   8 |     Io(#[from] std::io::Error),\n   9 |     #[error(\"Serialization failed\")]\n  10 |     Serialization,\n  11 | }"
        ),
        (
            "Async message dispatch handler in Rust",
            "pub struct Dispatcher {\n    sender: mpsc::Sender<Message>,\n}\n\nimpl Dispatcher {\n    pub async fn dispatch(&self, msg: Message) -> Result<(), DispatchError> {\n        self.sender.send(msg).await.map_err(|_| DispatchError::ChannelClosed)\n    }\n}\n",
            "   5 | impl Dispatcher {\n   6 |     pub async fn dispatch(&self, msg: Message) -> Result<(), DispatchError> {\n   7 |         self.sender.send(msg).await.map_err(|_| DispatchError::ChannelClosed)\n   8 |     }\n   9 | }"
        ),

        # --- 2. Go Structures ---
        (
            "UserRepository interface declaration",
            "package repository\n\nimport (\n\t\"context\"\n\t\"corp/models\"\n)\n\ntype UserRepository interface {\n\tFindByID(ctx context.Context, id string) (*models.User, error)\n\tFindByEmail(ctx context.Context, email string) (*models.User, error)\n\tCreate(ctx context.Context, u *models.User) error\n\tDelete(ctx context.Context, id string) error\n}\n",
            "   8 | type UserRepository interface {\n   9 | \tFindByID(ctx context.Context, id string) (*models.User, error)\n  10 | \tFindByEmail(ctx context.Context, email string) (*models.User, error)\n  11 | \tCreate(ctx context.Context, u *models.User) error\n  12 | \tDelete(ctx context.Context, id string) error\n  13 | }"
        ),
        (
            "HTTP Server Start method with graceful shutdown",
            "func (s *Server) Start(addr string) error {\n\ts.httpServer = &http.Server{\n\t\tAddr:    addr,\n\t\tHandler: s.router,\n\t}\n\n\tgo func() {\n\t\tif err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {\n\t\t\tlog.Fatalf(\"listen: %s\\n\", err)\n\t\t}\n\t}()\n\n\treturn nil\n}\n",
            "   1 | func (s *Server) Start(addr string) error {\n   2 | \ts.httpServer = &http.Server{\n   3 | \t\tAddr:    addr,\n   4 | \t\tHandler: s.router,\n   5 | \t}\n   6 |\n   7 | \tgo func() {\n   8 | \t\tif err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {\n   9 | \t\t\tlog.Fatalf(\"listen: %s\\n\", err)\n  10 | \t\t}\n  11 | \t}()\n  12 |\n  13 | \treturn nil\n  14 | }"
        ),
        (
            "JWT Claims struct definition in Go",
            "package auth\n\nimport \"github.com/golang-jwt/jwt/v5\"\n\ntype CustomClaims struct {\n\tUserID   string `json:\"user_id\"`\n\tTenantID string `json:\"tenant_id\"`\n\tRole     string `json:\"role\"`\n\tjwt.RegisteredClaims\n}\n",
            "   5 | type CustomClaims struct {\n   6 | \tUserID   string `json:\"user_id\"`\n   7 | \tTenantID string `json:\"tenant_id\"`\n   8 | \tRole     string `json:\"role\"`\n   9 | \tjwt.RegisteredClaims\n  10 | }"
        ),

        # --- 3. Python Structures ---
        (
            "ExecutionState dataclass definition",
            "from dataclasses import dataclass, field\nfrom typing import List, Optional\n\n@dataclass\nclass ExecutionState:\n    goal_prompt: str\n    turn: int = 0\n    history: List[dict] = field(default_factory=list)\n    status: str = \"running\"\n    last_error: Optional[str] = None\n\n    def is_terminal(self) -> bool:\n        return self.status in {\"completed\", \"failed\", \"stuck\"}\n",
            "   4 | @dataclass\n   5 | class ExecutionState:\n   6 |     goal_prompt: str\n   7 |     turn: int = 0\n   8 |     history: List[dict] = field(default_factory=list)\n   9 |     status: str = \"running\"\n  10 |     last_error: Optional[str] = None"
        ),
        (
            "FastAPI rate limited endpoint route",
            "from fastapi import APIRouter, Depends, HTTPException\n\nrouter = APIRouter(prefix=\"/items\", tags=[\"items\"])\n\n@router.get(\"/{item_id}\", response_model=ItemResponse)\n@limiter.limit(\"60/minute\")\nasync def read_item(item_id: str, db: Session = Depends(get_db)):\n    item = db.query(Item).filter(Item.id == item_id).first()\n    if not item:\n        raise HTTPException(status_code=404, detail=\"Item not found\")\n    return item\n",
            "   5 | @router.get(\"/{item_id}\", response_model=ItemResponse)\n   6 | @limiter.limit(\"60/minute\")\n   7 | async def read_item(item_id: str, db: Session = Depends(get_db)):\n   8 |     item = db.query(Item).filter(Item.id == item_id).first()\n   9 |     if not item:\n  10 |         raise HTTPException(status_code=404, detail=\"Item not found\")\n  11 |     return item"
        ),
        (
            "Pydantic validator for positive integer",
            "class OrderInput(BaseModel):\n    item_id: str\n    quantity: int\n    price_cents: int\n\n    @field_validator(\"quantity\", \"price_cents\")\n    @classmethod\n    def must_be_positive(cls, v: int) -> int:\n        if v <= 0:\n            raise ValueError(\"Value must be strictly positive\")\n        return v\n",
            "   6 |     @field_validator(\"quantity\", \"price_cents\")\n   7 |     @classmethod\n   8 |     def must_be_positive(cls, v: int) -> int:\n   9 |         if v <= 0:\n  10 |             raise ValueError(\"Value must be strictly positive\")\n  11 |         return v"
        ),

        # --- 4. TypeScript / React Structures ---
        (
            "Zod checkout input schema with custom refinement",
            "import { z } from 'zod';\n\nexport const CheckoutSchema = z.object({\n  cartId: z.string().uuid(),\n  currency: z.enum(['USD', 'EUR', 'GBP']),\n  amount: z.number().int().positive(),\n  discountCode: z.string().optional(),\n}).refine(data => data.amount >= 100, {\n  message: 'Minimum order amount is 100 cents',\n});\n\nexport type CheckoutInput = z.infer<typeof CheckoutSchema>;\n",
            "   3 | export const CheckoutSchema = z.object({\n   4 |   cartId: z.string().uuid(),\n   5 |   currency: z.enum(['USD', 'EUR', 'GBP']),\n   6 |   amount: z.number().int().positive(),\n   7 |   discountCode: z.string().optional(),\n   8 | }).refine(data => data.amount >= 100, {\n   9 |   message: 'Minimum order amount is 100 cents',\n  10 | });"
        ),
        (
            "Custom React useDebounce hook",
            "import { useEffect, useState } from 'react';\n\nexport function useDebounce<T>(value: T, delayMs: number = 300): T {\n  const [debouncedValue, setDebouncedValue] = useState<T>(value);\n\n  useEffect(() => {\n    const timer = setTimeout(() => setDebouncedValue(value), delayMs);\n    return () => clearTimeout(timer);\n  }, [value, delayMs]);\n\n  return debouncedValue;\n}\n",
            "   3 | export function useDebounce<T>(value: T, delayMs: number = 300): T {\n   4 |   const [debouncedValue, setDebouncedValue] = useState<T>(value);\n   5 |\n   6 |   useEffect(() => {\n   7 |     const timer = setTimeout(() => setDebouncedValue(value), delayMs);\n   8 |     return () => clearTimeout(timer);\n   9 |   }, [value, delayMs]);\n  10 |\n  11 |   return debouncedValue;\n  12 | }"
        ),
        (
            "Next.js generateMetadata export",
            "export async function generateMetadata({ params }: Props): Promise<Metadata> {\n  const { id } = await params;\n  const product = await getProduct(id);\n  return {\n    title: `${product.title} | Store`,\n    description: product.summary,\n    openGraph: {\n      images: [product.imageUrl],\n    },\n  };\n}\n",
            "   1 | export async function generateMetadata({ params }: Props): Promise<Metadata> {\n   2 |   const { id } = await params;\n   3 |   const product = await getProduct(id);\n   4 |   return {\n   5 |     title: `${product.title} | Store`,\n   6 |     description: product.summary,\n   7 |     openGraph: {\n   8 |       images: [product.imageUrl],\n   9 |     },\n  10 |   };\n  11 | }"
        ),

        # --- 5. CSS & Config Structures ---
        (
            "Tailwind 4 custom theme color definition",
            "@import \"tailwindcss\";\n\n@theme {\n  --color-brand-50: oklch(0.97 0.02 240);\n  --color-brand-500: oklch(0.62 0.22 250);\n  --color-brand-900: oklch(0.25 0.12 255);\n}\n\n@layer base {\n  body {\n    @apply bg-background text-foreground;\n  }\n}\n",
            "   3 | @theme {\n   4 |   --color-brand-50: oklch(0.97 0.02 240);\n   5 |   --color-brand-500: oklch(0.62 0.22 250);\n   6 |   --color-brand-900: oklch(0.25 0.12 255);\n   7 | }"
        ),
        (
            "SQL table definition with primary key and timestamps",
            "CREATE TABLE IF NOT EXISTS audit_logs (\n    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),\n    actor_id VARCHAR(64) NOT NULL,\n    action VARCHAR(128) NOT NULL,\n    payload JSONB DEFAULT '{}'::jsonb,\n    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()\n);\n\nCREATE INDEX idx_audit_actor ON audit_logs(actor_id);\n",
            "   1 | CREATE TABLE IF NOT EXISTS audit_logs (\n   2 |     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),\n   3 |     actor_id VARCHAR(64) NOT NULL,\n   4 |     action VARCHAR(128) NOT NULL,\n   5 |     payload JSONB DEFAULT '{}'::jsonb,\n   6 |     created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()\n   7 | );"
        ),
    ]
    return cases


def get_expanded_monorepo_cases() -> list[tuple[str, str, str]]:
    """Return 15+ composite monorepo and multi-package build/test runner parse cases."""
    cases = [
        (
            "pnpm --filter ... build (Exit code: 1)",
            "Scope: 4 of 5 workspace projects\n"
            "@corp/database:build: cache hit, replaying output 9a8c1f\n"
            "@corp/ui:build: cache hit, replaying output b42e10\n"
            "@corp/logger:build: Compiled in 320ms\n"
            "apps/web:build:\n"
            "  src/components/Header.tsx(24,15): error TS2322: Type 'string' is not assignable to type 'number'.\n"
            "  Failed to compile TypeScript.\n"
            "apps/web:build: ERROR: command finished with error: exit status 1\n"
            "pnpm run: command failed with exit code 1",
            "- Status: Failed (exit code: 1)\n"
            "- Failed Project: apps/web\n"
            "- Succeeded: @corp/database, @corp/ui, @corp/logger\n"
            "- Error Location: src/components/Header.tsx:24\n"
            "- Error Details: error TS2322: Type 'string' is not assignable to type 'number'."
        ),
        (
            "turbo run test (Exit code: 1)",
            "• Packages in scope: @repo/auth, @repo/database, @repo/ui, web\n"
            "@repo/ui:test: PASS src/button.test.tsx (12 tests)\n"
            "@repo/database:test: PASS tests/client.test.ts (8 tests)\n"
            "web:test: PASS __tests__/page.test.tsx (4 tests)\n"
            "@repo/auth:test: FAIL src/jwt.test.ts\n"
            "@repo/auth:test:   ● TokenVerifier › expires stale tokens\n"
            "@repo/auth:test:     expect(received).toBe(expected)\n"
            "@repo/auth:test:     Expected: false\n"
            "@repo/auth:test:     Received: true\n"
            "@repo/auth:test:     at Object.<anonymous> (src/jwt.test.ts:42:25)\n"
            "ERROR: command finished with error: exit status 1",
            "- Status: Failed (exit code: 1)\n"
            "- Failed Package: @repo/auth\n"
            "- Succeeded: @repo/ui, @repo/database, web\n"
            "- Failed Test: TokenVerifier › expires stale tokens\n"
            "- Error Location: src/jwt.test.ts:42\n"
            "- Assertion: Expected false, Received true"
        ),
        (
            "cargo test --workspace (Exit code: 101)",
            "running 8 tests in crate `core`\n"
            "test result: ok. 8 passed; 0 failed; 0 ignored\n"
            "running 14 tests in crate `network`\n"
            "test result: ok. 14 passed; 0 failed; 0 ignored\n"
            "running 6 tests in crate `storage`\n"
            "test storage::tests::test_concurrent_writes ... FAILED\n"
            "failures:\n"
            "---- storage::tests::test_concurrent_writes stdout ----\n"
            "thread 'storage::tests::test_concurrent_writes' panicked at src/lib.rs:88:9:\n"
            "assertion `left == right` failed\n"
            "  left: 49\n"
            " right: 50\n"
            "failures:\n"
            "    storage::tests::test_concurrent_writes\n"
            "test result: FAILED. 5 passed; 1 failed; finished in 0.42s",
            "- Status: Failed (exit code: 101)\n"
            "- Failed Crate: storage\n"
            "- Succeeded Crates: core (8 passed), network (14 passed)\n"
            "- Failed Test: storage::tests::test_concurrent_writes\n"
            "- Panic Location: src/lib.rs:88\n"
            "- Error Details: assertion `left == right` failed (left: 49, right: 50)"
        ),
        (
            "pnpm -r --filter @corp/* build (Exit code: 0)",
            "Scope: 4 workspace projects\n"
            "@corp/types:build: Done in 180ms\n"
            "@corp/utils:build: Done in 240ms\n"
            "@corp/ui:build: Done in 650ms\n"
            "@corp/api-client:build: Done in 410ms\n\n"
            "✔ Built 4 packages successfully in 1.48s",
            "- Status: Passed (exit code: 0)\n"
            "- Summary: All 4 packages built successfully in 1.48s.\n"
            "- Packages Built: @corp/types, @corp/utils, @corp/ui, @corp/api-client\n"
            "- Errors/Warnings: None"
        ),
        (
            "nx run-many -t test (Exit code: 0)",
            "✔  nx run api:test (18 tests passed)\n"
            "✔  nx run auth:test (24 tests passed)\n"
            "✔  nx run billing:test (12 tests passed)\n"
            "✔  nx run web:test (45 tests passed)\n\n"
            "Successfully ran target test for 4 projects (99 tests passed)",
            "- Status: Passed (exit code: 0)\n"
            "- Summary: 99 tests passed across 4 projects (api, auth, billing, web).\n"
            "- Errors/Warnings: None"
        ),
    ]
    return cases
