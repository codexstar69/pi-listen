# Contributing to pi-listen

## Development Setup

```bash
git clone https://github.com/codexstar69/pi-listen.git
cd pi-listen
bun install
```

## Commands

```bash
bun run typecheck   # TypeScript type checking
bun run test        # Run all tests
bun run check       # typecheck + tests
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module structure and design.

## Pull Requests

1. Fork and branch from `main`
2. Write tests for new functionality
3. Run `bun run check` before submitting
4. Keep PRs focused (one concern per PR)

## Code Style

- TypeScript, strict mode
- No `any` unless interfacing with untyped APIs (WebSocket, Pi SDK)
- Extract pure logic into testable modules under `extensions/voice/`
- Keep `extensions/voice.ts` focused on state management and UI

## License

By contributing, you agree your work is licensed under [MIT](LICENSE).
