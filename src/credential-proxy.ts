/**
 * Credential proxy for container isolation.
 * Containers connect here instead of directly to the Anthropic API.
 * The proxy injects real credentials so containers never see them.
 *
 * Two auth modes:
 *   API key:  Proxy injects x-api-key on every request.
 *   OAuth:    Container CLI exchanges its placeholder token for a temp
 *             API key via /api/oauth/claude_cli/create_api_key.
 *             Proxy injects real OAuth token on that exchange request;
 *             subsequent requests carry the temp key which is valid as-is.
 *
 * OAuth token lifecycle:
 *   Tokens are read from CLAUDE_CREDENTIALS_FILE (credentials.json) on each
 *   request with a 2-minute in-memory cache. When the token is within 5 min
 *   of expiry, the proxy refreshes it automatically using the OAuth refresh
 *   endpoint and rotates the refresh token in the file. Falls back to
 *   CLAUDE_CODE_OAUTH_TOKEN env var if the credentials file is unavailable.
 */
import fs from 'fs';
import { createServer, Server } from 'http';
import { request as httpsRequest } from 'https';
import { request as httpRequest, RequestOptions } from 'http';

import { readEnvFile } from './env.js';
import { logger } from './logger.js';

export type AuthMode = 'api-key' | 'oauth';

export interface ProxyConfig {
  authMode: AuthMode;
}

// ── OAuth token auto-refresh constants ───────────────────────────────────────

const OAUTH_TOKEN_URL_HOST = 'platform.claude.com';
const OAUTH_TOKEN_URL_PATH = '/v1/oauth/token';
const OAUTH_CLIENT_ID = '9d1c250a-e61b-44d9-88ed-5944d1962f5e';
const OAUTH_DEFAULT_SCOPES = [
  'user:profile',
  'user:inference',
  'user:sessions:claude_code',
  'user:mcp_servers',
];
/** Re-read credentials file at most every 2 minutes */
const TOKEN_CACHE_TTL_MS = 2 * 60 * 1000;
/** Refresh token when it expires within 5 minutes */
const TOKEN_REFRESH_BUFFER_MS = 5 * 60 * 1000;
/** Refresh request timeout */
const TOKEN_REFRESH_TIMEOUT_MS = 15_000;

interface OAuthTokens {
  accessToken: string;
  refreshToken: string | null;
  expiresAt: number | null;
  scopes: string[];
}

interface TokenCache {
  tokens: OAuthTokens;
  cachedAt: number;
}

let tokenCache: TokenCache | null = null;
let refreshInProgress: Promise<OAuthTokens | null> | null = null;

function getCredentialsPath(): string | null {
  return process.env.CLAUDE_CREDENTIALS_FILE || null;
}

function readCredentialsFile(): OAuthTokens | null {
  const credPath = getCredentialsPath();
  if (!credPath) return null;

  try {
    const content = fs.readFileSync(credPath, 'utf8');
    const data = JSON.parse(content);
    const o = data?.claudeAiOauth;
    if (!o?.accessToken) return null;
    return {
      accessToken: o.accessToken,
      refreshToken: o.refreshToken ?? null,
      expiresAt: o.expiresAt ?? null,
      scopes: o.scopes ?? OAUTH_DEFAULT_SCOPES,
    };
  } catch {
    return null;
  }
}

function writeCredentialsFile(tokens: OAuthTokens): void {
  const credPath = getCredentialsPath();
  if (!credPath) return;

  try {
    let existing: Record<string, unknown> = {};
    try {
      existing = JSON.parse(fs.readFileSync(credPath, 'utf8'));
    } catch {}

    const updated = {
      ...existing,
      claudeAiOauth: {
        ...(existing.claudeAiOauth as Record<string, unknown>),
        accessToken: tokens.accessToken,
        refreshToken: tokens.refreshToken,
        expiresAt: tokens.expiresAt,
        scopes: tokens.scopes,
      },
    };

    // Atomic write via temp file + rename
    const tmpPath = credPath + '.proxy.tmp';
    fs.writeFileSync(tmpPath, JSON.stringify(updated, null, 2));
    fs.renameSync(tmpPath, credPath);
    logger.debug('Credentials file updated with refreshed tokens');
  } catch (err) {
    logger.error(
      { err },
      'Failed to write credentials file after token refresh',
    );
  }
}

function doTokenRefresh(
  refreshToken: string,
  scopes: string[],
): Promise<OAuthTokens | null> {
  return new Promise((resolve) => {
    const body = JSON.stringify({
      grant_type: 'refresh_token',
      refresh_token: refreshToken,
      client_id: OAUTH_CLIENT_ID,
      scope: scopes.join(' '),
    });

    const req = httpsRequest(
      {
        hostname: OAUTH_TOKEN_URL_HOST,
        port: 443,
        path: OAUTH_TOKEN_URL_PATH,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(body),
        },
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => {
          try {
            const data = JSON.parse(Buffer.concat(chunks).toString());
            if (res.statusCode !== 200) {
              logger.error(
                { status: res.statusCode, data },
                'OAuth token refresh failed',
              );
              resolve(null);
              return;
            }
            resolve({
              accessToken: data.access_token,
              refreshToken: data.refresh_token ?? refreshToken,
              expiresAt: Date.now() + data.expires_in * 1000,
              scopes,
            });
          } catch (err) {
            logger.error({ err }, 'Failed to parse token refresh response');
            resolve(null);
          }
        });
        res.on('error', (err) => {
          logger.error({ err }, 'Token refresh response error');
          resolve(null);
        });
      },
    );

    req.on('error', (err) => {
      logger.error({ err }, 'Token refresh connection error');
      resolve(null);
    });

    req.setTimeout(TOKEN_REFRESH_TIMEOUT_MS, () => {
      req.destroy();
      logger.error('Token refresh timed out');
      resolve(null);
    });

    req.write(body);
    req.end();
  });
}

/**
 * Return a valid OAuth access token, refreshing automatically if needed.
 * Falls back to the static token from .env if credentials file is unavailable.
 */
async function getOAuthToken(
  envFallback: string | undefined,
): Promise<string | undefined> {
  const now = Date.now();

  // Serve from cache if still fresh and not about to expire
  if (tokenCache && now - tokenCache.cachedAt < TOKEN_CACHE_TTL_MS) {
    const { tokens } = tokenCache;
    if (
      tokens.expiresAt === null ||
      tokens.expiresAt > now + TOKEN_REFRESH_BUFFER_MS
    ) {
      return tokens.accessToken;
    }
  }

  // Re-read from file
  const tokens = readCredentialsFile();

  if (!tokens) {
    // Credentials file unavailable — use .env fallback
    return envFallback;
  }

  const isExpiring =
    tokens.expiresAt !== null &&
    tokens.expiresAt <= now + TOKEN_REFRESH_BUFFER_MS;

  if (!isExpiring) {
    tokenCache = { tokens, cachedAt: now };
    return tokens.accessToken;
  }

  // Token expired or expiring soon — attempt refresh
  if (tokens.refreshToken) {
    // Deduplicate concurrent refresh requests
    if (!refreshInProgress) {
      refreshInProgress = (async () => {
        try {
          logger.info(
            {
              expiresIn: tokens.expiresAt
                ? Math.round((tokens.expiresAt - now) / 1000)
                : 'unknown',
            },
            'OAuth token expiring — refreshing',
          );
          const newTokens = await doTokenRefresh(
            tokens.refreshToken!,
            tokens.scopes,
          );
          if (newTokens) {
            writeCredentialsFile(newTokens);
            tokenCache = { tokens: newTokens, cachedAt: Date.now() };
            logger.info('OAuth token refreshed successfully');
            return newTokens;
          }
          return null;
        } finally {
          refreshInProgress = null;
        }
      })();
    }

    const refreshed = await refreshInProgress;
    if (refreshed) return refreshed.accessToken;
  }

  // Refresh failed or no refresh token — use whatever is in the file
  logger.warn(
    'OAuth token refresh failed, using current (possibly expired) token',
  );
  tokenCache = { tokens, cachedAt: now };
  return tokens.accessToken;
}

// ── /api/chat helpers ─────────────────────────────────────────────────────────

function httpCall(
  makeReq: typeof httpsRequest,
  opts: RequestOptions,
  body: string,
): Promise<{ status: number; body: string }> {
  return new Promise((resolve, reject) => {
    const req = makeReq(opts, (res) => {
      const chunks: Buffer[] = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () =>
        resolve({
          status: res.statusCode!,
          body: Buffer.concat(chunks).toString(),
        }),
      );
    });
    req.on('error', reject);
    req.setTimeout(30_000, () => {
      req.destroy();
      reject(new Error('timeout'));
    });
    req.write(body);
    req.end();
  });
}

// ── Proxy server ─────────────────────────────────────────────────────────────

export function startCredentialProxy(
  port: number,
  host = '127.0.0.1',
): Promise<Server> {
  const secrets = readEnvFile([
    'ANTHROPIC_API_KEY',
    'CLAUDE_CODE_OAUTH_TOKEN',
    'ANTHROPIC_AUTH_TOKEN',
    'ANTHROPIC_BASE_URL',
  ]);

  const authMode: AuthMode = secrets.ANTHROPIC_API_KEY ? 'api-key' : 'oauth';
  // Static fallback for OAuth mode (used when credentials file is unavailable)
  const oauthFallback =
    secrets.CLAUDE_CODE_OAUTH_TOKEN || secrets.ANTHROPIC_AUTH_TOKEN;

  if (authMode === 'oauth') {
    const credPath = getCredentialsPath();
    if (credPath) {
      logger.info(
        { credentialsFile: credPath },
        'OAuth mode: using credentials file with auto-refresh',
      );
    } else {
      logger.warn(
        'CLAUDE_CREDENTIALS_FILE not set — OAuth token will not auto-refresh',
      );
    }
  }

  const upstreamUrl = new URL(
    secrets.ANTHROPIC_BASE_URL || 'https://api.anthropic.com',
  );
  const isHttps = upstreamUrl.protocol === 'https:';
  const makeRequest = isHttps ? httpsRequest : httpRequest;

  /** Get auth headers for calling the Anthropic API. */
  async function getAuthHeaders(): Promise<Record<string, string>> {
    if (authMode === 'api-key') {
      return { 'x-api-key': secrets.ANTHROPIC_API_KEY! };
    }
    // OAuth mode: use Bearer token directly
    const token = await getOAuthToken(oauthFallback);
    if (!token) throw new Error('No OAuth token available');
    return { authorization: `Bearer ${token}` };
  }

  return new Promise((resolve, reject) => {
    const server = createServer((req, res) => {
      // ── /api/chat — simple chat endpoint for n8n and external integrations ──
      if (req.url === '/api/chat' && req.method === 'POST') {
        const chatChunks: Buffer[] = [];
        req.on('data', (c) => chatChunks.push(c));
        req.on('end', async () => {
          try {
            const input = JSON.parse(Buffer.concat(chatChunks).toString());
            const message = input.message || input.content;
            if (!message && !input.messages) {
              res.writeHead(400, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ error: 'message field required' }));
              return;
            }

            const authHeaders = await getAuthHeaders();
            const model = input.model || 'claude-haiku-4-5-20251001';
            const maxTokens = input.max_tokens || 4096;
            const messages =
              input.messages || [{ role: 'user', content: message }];

            const apiBody = JSON.stringify({
              model,
              max_tokens: maxTokens,
              ...(input.system ? { system: input.system } : {}),
              messages,
            });

            const resp = await httpCall(
              makeRequest,
              {
                hostname: upstreamUrl.hostname,
                port: upstreamUrl.port || (isHttps ? 443 : 80),
                path: '/v1/messages?beta=true',
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                  'content-length': Buffer.byteLength(apiBody),
                  ...authHeaders,
                  host: upstreamUrl.host,
                  'anthropic-version': '2023-06-01',
                  'anthropic-beta': 'oauth-2025-04-20',
                },
              } as RequestOptions,
              apiBody,
            );

            const data = JSON.parse(resp.body);
            const text = data.content
              ?.filter((b: { type: string }) => b.type === 'text')
              .map((b: { text: string }) => b.text)
              .join('');

            if (resp.status !== 200) {
              logger.warn({ status: resp.status, body: resp.body.substring(0, 300) }, '/api/chat upstream error');
            }

            res.writeHead(resp.status, {
              'Content-Type': 'application/json',
            });
            res.end(JSON.stringify({ text: text || '', model: data.model, usage: data.usage }));
          } catch (err) {
            logger.error({ err }, '/api/chat error');
            if (!res.headersSent) {
              res.writeHead(500, { 'Content-Type': 'application/json' });
              res.end(
                JSON.stringify({
                  error: err instanceof Error ? err.message : 'internal error',
                }),
              );
            }
          }
        });
        return;
      }

      logger.debug({ method: req.method, url: req.url }, 'Proxy request');
      const chunks: Buffer[] = [];
      req.on('data', (c) => chunks.push(c));
      req.on('end', async () => {
        const body = Buffer.concat(chunks);
        const headers: Record<string, string | number | string[] | undefined> =
          {
            ...(req.headers as Record<string, string>),
            host: upstreamUrl.host,
            'content-length': body.length,
          };

        // Strip hop-by-hop headers that must not be forwarded by proxies
        delete headers['connection'];
        delete headers['keep-alive'];
        delete headers['transfer-encoding'];

        if (authMode === 'api-key') {
          delete headers['x-api-key'];
          headers['x-api-key'] = secrets.ANTHROPIC_API_KEY;
        } else {
          // OAuth mode: replace placeholder Bearer token with the real one
          // (only on exchange/auth requests that carry Authorization header)
          if (headers['authorization']) {
            delete headers['authorization'];
            const token = await getOAuthToken(oauthFallback);
            if (token) {
              headers['authorization'] = `Bearer ${token}`;
            }
          }
        }

        const upstream = makeRequest(
          {
            hostname: upstreamUrl.hostname,
            port: upstreamUrl.port || (isHttps ? 443 : 80),
            path: req.url,
            method: req.method,
            headers,
          } as RequestOptions,
          (upRes) => {
            res.writeHead(upRes.statusCode!, upRes.headers);
            upRes.pipe(res);
          },
        );

        upstream.on('error', (err) => {
          logger.error(
            { err, url: req.url },
            'Credential proxy upstream error',
          );
          if (!res.headersSent) {
            res.writeHead(502);
            res.end('Bad Gateway');
          }
        });

        upstream.write(body);
        upstream.end();
      });
    });

    server.listen(port, host, () => {
      logger.info({ port, host, authMode }, 'Credential proxy started');
      resolve(server);
    });

    server.on('error', reject);
  });
}

/** Detect which auth mode the host is configured for. */
export function detectAuthMode(): AuthMode {
  const secrets = readEnvFile(['ANTHROPIC_API_KEY']);
  return secrets.ANTHROPIC_API_KEY ? 'api-key' : 'oauth';
}
