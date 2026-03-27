/**
 * HTTP Channel for NanoClaw.
 *
 * Exposes a POST /api/chat endpoint that feeds messages into the full
 * NanoClaw pipeline (container agent with Claude Code CLI → Sonnet).
 *
 * Flow:
 *   HTTP POST → onMessage() → message loop → container agent → sendMessage() → HTTP response
 *
 * The key trick: sendMessage() buffers output, and setTyping(jid, false)
 * flushes the buffer and resolves the pending HTTP response. This works
 * because processGroupMessages() calls setTyping(false) when the agent finishes.
 */
import { createServer, Server, IncomingMessage, ServerResponse } from 'http';
import { randomUUID } from 'crypto';

import { ASSISTANT_NAME } from '../config.js';
import { readEnvFile } from '../env.js';
import { logger } from '../logger.js';
import { registerChannel, ChannelOpts } from './registry.js';
import { Channel, RegisteredGroup } from '../types.js';

const DEFAULT_TIMEOUT_MS = 120_000; // 2 minutes
const DEFAULT_SESSION = 'default';

interface PendingRequest {
  resolve: (text: string) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class HttpChannel implements Channel {
  name = 'http';

  private server: Server | null = null;
  private port: number;
  private host: string;
  private opts: ChannelOpts;
  private pending = new Map<string, PendingRequest>();

  constructor(port: number, host: string, opts: ChannelOpts) {
    this.port = port;
    this.host = host;
    this.opts = opts;
  }

  async connect(): Promise<void> {
    // Auto-register the default HTTP session
    this.ensureGroupRegistered(`http:${DEFAULT_SESSION}`, 'HTTP Default');

    this.server = createServer((req, res) => {
      if (req.method === 'POST' && req.url === '/api/chat') {
        this.handleChat(req, res);
      } else if (req.method === 'GET' && req.url === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'ok', channel: 'http' }));
      } else {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Not found. Use POST /api/chat' }));
      }
    });

    return new Promise<void>((resolve, reject) => {
      this.server!.on('error', reject);
      this.server!.listen(this.port, this.host, () => {
        logger.info(
          { port: this.port, host: this.host },
          'HTTP channel listening',
        );
        console.log(`\n  HTTP channel: http://${this.host}:${this.port}/api/chat\n`);
        resolve();
      });
    });
  }

  async sendMessage(jid: string, text: string): Promise<void> {
    const pending = this.pending.get(jid);
    if (!pending) {
      logger.debug({ jid }, 'HTTP sendMessage: no pending request (already resolved)');
      return;
    }

    // Resolve immediately on first sendMessage — the agent may send
    // multiple chunks but the first one contains the answer.
    // Container stays alive for session reuse (idle timeout), so we
    // can't wait for setTyping(false) which only fires after idle.
    clearTimeout(pending.timer);
    this.pending.delete(jid);
    pending.resolve(text);
    logger.info({ jid, length: text.length }, 'HTTP response sent');
  }

  async setTyping(_jid: string, _isTyping: boolean): Promise<void> {
    // No-op for HTTP channel. We resolve on first sendMessage.
  }

  isConnected(): boolean {
    return this.server !== null;
  }

  ownsJid(jid: string): boolean {
    return jid.startsWith('http:');
  }

  async disconnect(): Promise<void> {
    if (this.server) {
      this.server.close();
      this.server = null;
      for (const [, pending] of this.pending) {
        clearTimeout(pending.timer);
        pending.reject(new Error('Channel shutting down'));
      }
      this.pending.clear();
      logger.info('HTTP channel stopped');
    }
  }

  private ensureGroupRegistered(jid: string, name: string): void {
    const groups = this.opts.registeredGroups();
    if (groups[jid]) return;

    if (!this.opts.registerGroup) {
      logger.warn(
        { jid },
        'HTTP channel: cannot auto-register group (registerGroup not available)',
      );
      return;
    }

    const slug = jid.replace('http:', '').replace(/[^a-z0-9_-]/gi, '_');
    this.opts.registerGroup(jid, {
      name,
      folder: `http_${slug}`,
      trigger: `@${ASSISTANT_NAME}`,
      added_at: new Date().toISOString(),
      requiresTrigger: false,
      isMain: false,
    });
    logger.info({ jid, name }, 'HTTP group auto-registered');
  }

  private handleChat(req: IncomingMessage, res: ServerResponse): void {
    const chunks: Buffer[] = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      try {
        const body = JSON.parse(Buffer.concat(chunks).toString());
        const message: string = body.message;
        if (!message) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'message field required' }));
          return;
        }

        const sessionId = body.session_id || DEFAULT_SESSION;
        const jid = `http:${sessionId}`;
        const senderName = body.sender_name || 'HTTP User';
        const timeoutMs = body.timeout || DEFAULT_TIMEOUT_MS;
        const msgId = randomUUID();
        const timestamp = new Date().toISOString();

        // Ensure session group is registered
        this.ensureGroupRegistered(jid, `HTTP: ${sessionId}`);

        // Always prepend trigger so the message is processed
        const content = `@${ASSISTANT_NAME} ${message}`;

        // If there's already a pending request for this session, reject it
        const existing = this.pending.get(jid);
        if (existing) {
          clearTimeout(existing.timer);
          existing.reject(new Error('Superseded by new request'));
          this.pending.delete(jid);
        }

        // Create pending response
        const responsePromise = new Promise<string>((resolve, reject) => {
          const timer = setTimeout(() => {
            this.pending.delete(jid);
            reject(new Error('Response timeout'));
          }, timeoutMs);

          this.pending.set(jid, { resolve, reject, timer });
        });

        // Emit metadata and store message — triggers the pipeline
        this.opts.onChatMetadata(
          jid,
          timestamp,
          `HTTP: ${sessionId}`,
          'http',
          false,
        );
        this.opts.onMessage(jid, {
          id: msgId,
          chat_jid: jid,
          sender: 'http-user',
          sender_name: senderName,
          content,
          timestamp,
          is_from_me: false,
        });

        logger.info(
          { jid, senderName, messageLength: message.length },
          'HTTP chat request received',
        );

        // Wait for pipeline response
        responsePromise
          .then((text) => {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ text, session_id: sessionId }));
          })
          .catch((err) => {
            if (!res.headersSent) {
              res.writeHead(504, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ error: err.message }));
            }
          });
      } catch {
        if (!res.headersSent) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Invalid JSON' }));
        }
      }
    });
  }
}

registerChannel('http', (opts: ChannelOpts) => {
  const envVars = readEnvFile([
    'HTTP_CHANNEL_ENABLED',
    'HTTP_CHANNEL_PORT',
    'HTTP_CHANNEL_HOST',
  ]);
  const enabled =
    process.env.HTTP_CHANNEL_ENABLED || envVars.HTTP_CHANNEL_ENABLED;
  if (enabled !== 'true') {
    return null;
  }
  const port = parseInt(
    process.env.HTTP_CHANNEL_PORT || envVars.HTTP_CHANNEL_PORT || '3003',
    10,
  );
  const host =
    process.env.HTTP_CHANNEL_HOST || envVars.HTTP_CHANNEL_HOST || '0.0.0.0';
  return new HttpChannel(port, host, opts);
});
