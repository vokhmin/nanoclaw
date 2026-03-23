import fs from 'fs';
import https from 'https';
import path from 'path';
import { Api, Bot, InputFile } from 'grammy';
import Groq from 'groq-sdk';
import OpenAI from 'openai';

import { ASSISTANT_NAME, TRIGGER_PATTERN } from '../config.js';
import { DATA_DIR } from '../config.js';
import { readEnvFile } from '../env.js';
import { logger } from '../logger.js';
import { registerChannel, ChannelOpts } from './registry.js';
import {
  Channel,
  OnChatMetadata,
  OnInboundMessage,
  RegisteredGroup,
} from '../types.js';

export interface TelegramChannelOpts {
  onMessage: OnInboundMessage;
  onChatMetadata: OnChatMetadata;
  registeredGroups: () => Record<string, RegisteredGroup>;
}

/** Max characters to synthesize as voice — keeps messages under ~60-90 sec */
const TTS_MAX_CHARS = 800;

/** Voice mode per JID — persisted to data/voice-settings.json */
type VoiceMode = 'auto' | 'on' | 'off';

const VOICE_SETTINGS_FILE = path.join(DATA_DIR, 'voice-settings.json');

function loadVoiceSettings(): Record<string, VoiceMode> {
  try {
    return JSON.parse(fs.readFileSync(VOICE_SETTINGS_FILE, 'utf8'));
  } catch {
    return {};
  }
}

function saveVoiceSettings(settings: Record<string, VoiceMode>): void {
  try {
    fs.mkdirSync(path.dirname(VOICE_SETTINGS_FILE), { recursive: true });
    fs.writeFileSync(VOICE_SETTINGS_FILE, JSON.stringify(settings, null, 2));
  } catch (err) {
    logger.error({ err }, 'Failed to save voice settings');
  }
}

/**
 * Strip Markdown formatting before TTS synthesis.
 * Prevents the model from reading out asterisks, backticks, headers, etc.
 */
function stripMarkdown(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, '[код]')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/__(.*?)__/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/_(.*?)_/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/^\s*[-*+]\s+/gm, '')
    .trim();
}

/**
 * Send a message with Telegram Markdown parse mode, falling back to plain text.
 */
async function sendTelegramMessage(
  api: { sendMessage: Api['sendMessage'] },
  chatId: string | number,
  text: string,
  options: { message_thread_id?: number } = {},
): Promise<void> {
  try {
    await api.sendMessage(chatId, text, {
      ...options,
      parse_mode: 'Markdown',
    });
  } catch (err) {
    logger.debug({ err }, 'Markdown send failed, falling back to plain text');
    await api.sendMessage(chatId, text, options);
  }
}

/**
 * Download a file from a URL into a Buffer.
 */
function downloadBuffer(url: string): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    https
      .get(url, (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (chunk) => chunks.push(chunk));
        res.on('end', () => resolve(Buffer.concat(chunks)));
        res.on('error', reject);
      })
      .on('error', reject);
  });
}

export class TelegramChannel implements Channel {
  name = 'telegram';

  private bot: Bot | null = null;
  private opts: TelegramChannelOpts;
  private botToken: string;
  private groq: Groq | null = null;
  private openai: OpenAI | null = null;

  /** JIDs where the last incoming message was a voice — used in 'auto' mode */
  private voiceJids = new Set<string>();

  /** Per-JID voice mode, persisted to disk */
  private voiceSettings: Record<string, VoiceMode>;

  constructor(botToken: string, opts: TelegramChannelOpts) {
    this.botToken = botToken;
    this.opts = opts;
    this.voiceSettings = loadVoiceSettings();

    const groqApiKey = process.env.GROQ_API_KEY;
    if (groqApiKey) {
      this.groq = new Groq({ apiKey: groqApiKey });
      logger.info('Groq STT (voice transcription) enabled');
    } else {
      logger.warn(
        'GROQ_API_KEY not set — voice messages will not be transcribed',
      );
    }

    const openaiApiKey = process.env.OPENAI_API_KEY;
    if (openaiApiKey) {
      this.openai = new OpenAI({ apiKey: openaiApiKey });
      logger.info('OpenAI TTS (voice responses) enabled');
    } else {
      logger.warn('OPENAI_API_KEY not set — voice responses disabled');
    }
  }

  private getVoiceMode(jid: string): VoiceMode {
    return this.voiceSettings[jid] ?? 'auto';
  }

  private setVoiceMode(jid: string, mode: VoiceMode): void {
    this.voiceSettings[jid] = mode;
    saveVoiceSettings(this.voiceSettings);
  }

  /** Returns true if a voice response should be sent for this JID right now */
  private shouldSendVoice(jid: string): boolean {
    if (!this.openai) return false;
    const mode = this.getVoiceMode(jid);
    if (mode === 'on') return true;
    if (mode === 'off') return false;
    // auto: only if last message was voice
    return this.voiceJids.has(jid);
  }

  /**
   * Transcribe a Telegram voice message using Groq Whisper.
   */
  private async transcribeVoice(fileId: string): Promise<string | null> {
    if (!this.groq || !this.bot) return null;

    try {
      const file = await this.bot.api.getFile(fileId);
      if (!file.file_path) return null;

      const fileUrl = `https://api.telegram.org/file/bot${this.botToken}/${file.file_path}`;
      const audioBuffer = await downloadBuffer(fileUrl);

      const audioFile = new File([audioBuffer], 'voice.ogg', {
        type: 'audio/ogg',
      });

      const result = await this.groq.audio.transcriptions.create({
        file: audioFile,
        model: 'whisper-large-v3',
      });

      return result.text || null;
    } catch (err) {
      logger.error({ err }, 'Voice transcription failed');
      return null;
    }
  }

  /**
   * Synthesize text to speech via OpenAI TTS and send as Telegram voice message.
   * Only the first TTS_MAX_CHARS characters are synthesized — full text is always
   * sent as a separate text message by the caller.
   */
  private async synthesizeAndSendVoice(
    chatId: string | number,
    text: string,
    threadId?: number,
  ): Promise<void> {
    if (!this.openai || !this.bot) return;

    try {
      const stripped = stripMarkdown(text);
      const input =
        stripped.length > TTS_MAX_CHARS
          ? stripped.slice(0, TTS_MAX_CHARS) + '...'
          : stripped;

      const response = await this.openai.audio.speech.create({
        model: 'tts-1',
        voice: 'nova',
        input,
        response_format: 'opus',
      });

      const buffer = Buffer.from(await response.arrayBuffer());
      await this.bot.api.sendVoice(
        chatId,
        new InputFile(buffer, 'response.ogg'),
        threadId ? { message_thread_id: threadId } : {},
      );
      logger.info({ chatId, chars: input.length }, 'Voice response sent');
    } catch (err) {
      logger.error({ err }, 'TTS synthesis or voice send failed');
    }
  }

  async connect(): Promise<void> {
    this.bot = new Bot(this.botToken, {
      client: {
        baseFetchConfig: { agent: https.globalAgent, compress: true },
      },
    });

    // Command to get chat ID (useful for registration)
    this.bot.command('chatid', (ctx) => {
      const chatId = ctx.chat.id;
      const threadId = (ctx.message as any)?.message_thread_id;
      const chatType = ctx.chat.type;
      const baseChatJid = `tg:${chatId}`;
      const fullChatJid = threadId ? `${baseChatJid}:${threadId}` : baseChatJid;
      const chatName =
        chatType === 'private'
          ? ctx.from?.first_name || 'Private'
          : (ctx.chat as any).title || 'Unknown';

      ctx.reply(
        threadId
          ? `Chat ID: \`${fullChatJid}\`\nTopic ID: ${threadId}\nName: ${chatName}\nType: ${chatType}`
          : `Chat ID: \`${fullChatJid}\`\nName: ${chatName}\nType: ${chatType}`,
        { parse_mode: 'Markdown' },
      );
    });

    // Command to check bot status
    this.bot.command('ping', (ctx) => {
      ctx.reply(`${ASSISTANT_NAME} is online.`);
    });

    // /voice [on|off|auto] — get or set voice response mode
    this.bot.command('voice', (ctx) => {
      const chatJid = `tg:${ctx.chat.id}`;
      const arg = ctx.match?.trim().toLowerCase();

      const modeLabels: Record<VoiceMode, string> = {
        auto: 'auto — голос в ответ на голос, текст в ответ на текст',
        on: 'on — всегда отвечать голосом',
        off: 'off — никогда не отвечать голосом',
      };

      if (!arg) {
        const current = this.getVoiceMode(chatJid);
        ctx.reply(
          `🔊 Текущий режим: *${current}*\n_${modeLabels[current]}_\n\nИзменить: /voice on | /voice off | /voice auto`,
          { parse_mode: 'Markdown' },
        );
        return;
      }

      if (arg === 'on' || arg === 'off' || arg === 'auto') {
        this.setVoiceMode(chatJid, arg);
        ctx.reply(
          `✅ Режим голосовых ответов: *${arg}*\n_${modeLabels[arg]}_`,
          { parse_mode: 'Markdown' },
        );
        logger.info({ chatJid, mode: arg }, 'Voice mode changed');
      } else {
        ctx.reply(
          'Неизвестный режим. Используй: /voice on | /voice off | /voice auto',
        );
      }
    });

    const TELEGRAM_BOT_COMMANDS = new Set(['chatid', 'ping', 'voice']);

    this.bot.on('message:text', async (ctx) => {
      if (ctx.message.text.startsWith('/')) {
        const cmd = ctx.message.text.slice(1).split(/[\s@]/)[0].toLowerCase();
        if (TELEGRAM_BOT_COMMANDS.has(cmd)) return;
      }

      const threadId = ctx.message.message_thread_id;
      const baseChatJid = `tg:${ctx.chat.id}`;
      const chatJid = threadId ? `${baseChatJid}:${threadId}` : baseChatJid;
      let content = ctx.message.text;
      const timestamp = new Date(ctx.message.date * 1000).toISOString();
      const senderName =
        ctx.from?.first_name ||
        ctx.from?.username ||
        ctx.from?.id.toString() ||
        'Unknown';
      const sender = ctx.from?.id.toString() || '';
      const msgId = ctx.message.message_id.toString();

      const chatName =
        ctx.chat.type === 'private'
          ? senderName
          : (ctx.chat as any).title || chatJid;

      const botUsername = ctx.me?.username?.toLowerCase();
      if (botUsername) {
        const entities = ctx.message.entities || [];
        const isBotMentioned = entities.some((entity) => {
          if (entity.type === 'mention') {
            const mentionText = content
              .substring(entity.offset, entity.offset + entity.length)
              .toLowerCase();
            return mentionText === `@${botUsername}`;
          }
          return false;
        });
        if (isBotMentioned && !TRIGGER_PATTERN.test(content)) {
          content = `@${ASSISTANT_NAME} ${content}`;
        }
      }

      const isGroup =
        ctx.chat.type === 'group' || ctx.chat.type === 'supergroup';
      this.opts.onChatMetadata(
        chatJid,
        timestamp,
        chatName,
        'telegram',
        isGroup,
      );

      let group = this.opts.registeredGroups()[chatJid];
      if (!group && threadId) {
        // Topic not registered — check if base chat is registered.
        // If so, skip: topics are opt-in and should not fall through to the base chat.
        if (this.opts.registeredGroups()[baseChatJid]) {
          logger.debug(
            { chatJid, baseChatJid, threadId },
            'Topic message in registered chat — topic not separately registered, skipping',
          );
          return;
        }
      }
      if (!group) {
        logger.debug(
          { chatJid, chatName },
          'Message from unregistered Telegram chat',
        );
        return;
      }

      this.opts.onMessage(chatJid, {
        id: msgId,
        chat_jid: chatJid,
        sender,
        sender_name: senderName,
        content,
        timestamp,
        is_from_me: false,
      });

      logger.info(
        { chatJid, chatName, sender: senderName },
        'Telegram message stored',
      );
    });

    // Handle non-text messages with placeholders
    const storeNonText = (ctx: any, placeholder: string) => {
      const threadId = ctx.message?.message_thread_id;
      const baseChatJid = `tg:${ctx.chat.id}`;
      const chatJid = threadId ? `${baseChatJid}:${threadId}` : baseChatJid;
      const group = this.opts.registeredGroups()[chatJid];
      if (!group) return;

      const timestamp = new Date(ctx.message.date * 1000).toISOString();
      const senderName =
        ctx.from?.first_name ||
        ctx.from?.username ||
        ctx.from?.id?.toString() ||
        'Unknown';
      const caption = ctx.message.caption ? ` ${ctx.message.caption}` : '';

      const isGroup =
        ctx.chat.type === 'group' || ctx.chat.type === 'supergroup';
      this.opts.onChatMetadata(
        chatJid,
        timestamp,
        undefined,
        'telegram',
        isGroup,
      );
      this.opts.onMessage(chatJid, {
        id: ctx.message.message_id.toString(),
        chat_jid: chatJid,
        sender: ctx.from?.id?.toString() || '',
        sender_name: senderName,
        content: `${placeholder}${caption}`,
        timestamp,
        is_from_me: false,
      });
    };

    // Voice messages: transcribe via Groq Whisper, mark JID for auto-mode TTS
    this.bot.on('message:voice', async (ctx) => {
      const threadId = ctx.message.message_thread_id;
      const baseChatJid = `tg:${ctx.chat.id}`;
      const chatJid = threadId ? `${baseChatJid}:${threadId}` : baseChatJid;
      const group = this.opts.registeredGroups()[chatJid];
      if (!group) return;

      const timestamp = new Date(ctx.message.date * 1000).toISOString();
      const senderName =
        ctx.from?.first_name ||
        ctx.from?.username ||
        ctx.from?.id?.toString() ||
        'Unknown';
      const caption = ctx.message.caption ? ` ${ctx.message.caption}` : '';
      const isGroup =
        ctx.chat.type === 'group' || ctx.chat.type === 'supergroup';

      this.opts.onChatMetadata(
        chatJid,
        timestamp,
        undefined,
        'telegram',
        isGroup,
      );

      const transcription = await this.transcribeVoice(
        ctx.message.voice.file_id,
      );
      const content = transcription
        ? `@${ASSISTANT_NAME} [Voice: ${transcription}]${caption}`
        : `@${ASSISTANT_NAME} [Voice message — transcription unavailable]${caption}`;

      if (transcription) {
        // Mark for auto-mode: next response should be voice too
        this.voiceJids.add(chatJid);
        logger.info(
          { chatJid, chars: transcription.length },
          'Voice transcribed',
        );
      }

      this.opts.onMessage(chatJid, {
        id: ctx.message.message_id.toString(),
        chat_jid: chatJid,
        sender: ctx.from?.id?.toString() || '',
        sender_name: senderName,
        content,
        timestamp,
        is_from_me: false,
      });
    });

    this.bot.on('message:photo', (ctx) => storeNonText(ctx, '[Photo]'));
    this.bot.on('message:video', (ctx) => storeNonText(ctx, '[Video]'));
    this.bot.on('message:audio', (ctx) => storeNonText(ctx, '[Audio]'));
    this.bot.on('message:document', (ctx) => {
      const name = ctx.message.document?.file_name || 'file';
      storeNonText(ctx, `[Document: ${name}]`);
    });
    this.bot.on('message:sticker', (ctx) => {
      const emoji = ctx.message.sticker?.emoji || '';
      storeNonText(ctx, `[Sticker ${emoji}]`);
    });
    this.bot.on('message:location', (ctx) => storeNonText(ctx, '[Location]'));
    this.bot.on('message:contact', (ctx) => storeNonText(ctx, '[Contact]'));

    this.bot.catch((err) => {
      logger.error({ err: err.message }, 'Telegram bot error');
    });

    return new Promise<void>((resolve) => {
      this.bot!.start({
        onStart: (botInfo) => {
          logger.info(
            { username: botInfo.username, id: botInfo.id },
            'Telegram bot connected',
          );
          console.log(`\n  Telegram bot: @${botInfo.username}`);
          console.log(
            `  Send /chatid to the bot to get a chat's registration ID\n`,
          );
          resolve();
        },
      });
    });
  }

  async sendMessage(jid: string, text: string): Promise<void> {
    if (!this.bot) {
      logger.warn('Telegram bot not initialized');
      return;
    }

    try {
      const parts = jid.replace(/^tg:/, '').split(':');
      const numericId = parts[0];
      const threadId = parts.length > 1 ? parseInt(parts[1], 10) : undefined;
      const msgOpts = threadId ? { message_thread_id: threadId } : {};

      if (this.shouldSendVoice(jid)) {
        // Clear auto-mode flag
        this.voiceJids.delete(jid);
        // Fire-and-forget TTS — don't block text delivery
        this.synthesizeAndSendVoice(numericId, text, threadId).catch((err) =>
          logger.error({ jid, err }, 'Voice response failed'),
        );
      }

      // Always send full text response
      const MAX_LENGTH = 4096;
      if (text.length <= MAX_LENGTH) {
        await sendTelegramMessage(this.bot.api, numericId, text, msgOpts);
      } else {
        for (let i = 0; i < text.length; i += MAX_LENGTH) {
          await sendTelegramMessage(
            this.bot.api,
            numericId,
            text.slice(i, i + MAX_LENGTH),
            msgOpts,
          );
        }
      }
      logger.info({ jid, length: text.length }, 'Telegram message sent');
    } catch (err) {
      logger.error({ jid, err }, 'Failed to send Telegram message');
    }
  }

  isConnected(): boolean {
    return this.bot !== null;
  }

  ownsJid(jid: string): boolean {
    return jid.startsWith('tg:');
  }

  async disconnect(): Promise<void> {
    if (this.bot) {
      this.bot.stop();
      this.bot = null;
      logger.info('Telegram bot stopped');
    }
  }

  async setTyping(jid: string, isTyping: boolean): Promise<void> {
    if (!this.bot || !isTyping) return;
    try {
      const parts = jid.replace(/^tg:/, '').split(':');
      const numericId = parts[0];
      const threadId = parts.length > 1 ? parseInt(parts[1], 10) : undefined;
      await this.bot.api.sendChatAction(
        numericId,
        'typing',
        threadId ? { message_thread_id: threadId } : {},
      );
    } catch (err) {
      logger.debug({ jid, err }, 'Failed to send Telegram typing indicator');
    }
  }
}

registerChannel('telegram', (opts: ChannelOpts) => {
  const envVars = readEnvFile(['TELEGRAM_BOT_TOKEN']);
  const token =
    process.env.TELEGRAM_BOT_TOKEN || envVars.TELEGRAM_BOT_TOKEN || '';
  if (!token) {
    logger.warn('Telegram: TELEGRAM_BOT_TOKEN not set');
    return null;
  }
  return new TelegramChannel(token, opts);
});
