/**
 * `@sentinel/protocol` - the wire.
 *
 * Versioned envelopes, the channel vocabulary, and an RFC 6455 codec shared by
 * both ends of every realtime connection. Zero dependencies: a security
 * appliance's dependency list is part of its attack surface, and everything here
 * parses bytes that arrived from the network.
 */

export * from './envelope.ts';
export * from './channels.ts';
export * from './websocket/frame.ts';
export * from './websocket/handshake.ts';
