import { createHash, randomBytes, timingSafeEqual } from 'node:crypto';

/**
 * RFC 6455 opening handshake.
 *
 * The GUID below is not a secret and not a security measure. It exists so that a
 * server which blindly echoes a request header cannot accidentally complete a
 * WebSocket handshake - the client checks that the server actually performed the
 * transformation, which proves it understood the request rather than reflecting
 * it. That is the whole purpose, and it is worth stating because the constant
 * otherwise looks like a magic key.
 */
export const WEBSOCKET_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';

export const WEBSOCKET_VERSION = 13;

export const acceptKey = (clientKey: string): string =>
  createHash('sha1').update(`${clientKey}${WEBSOCKET_GUID}`).digest('base64');

/** A client key is 16 random bytes, base64-encoded. */
export const generateClientKey = (): string => randomBytes(16).toString('base64');

export type HandshakeRequest = {
  readonly method: string;
  readonly path: string;
  readonly headers: Readonly<Record<string, string>>;
};

export type HandshakeRejection = {
  readonly status: number;
  readonly reason: string;
};

export type HandshakeResult =
  | { readonly ok: true; readonly acceptKey: string; readonly protocol: string | null }
  | { readonly ok: false; readonly rejection: HandshakeRejection };

/**
 * Validate an upgrade request.
 *
 * Rejections carry a status and a reason because an upgrade that fails silently
 * is one of the more miserable things to debug: the client sees a closed socket
 * and the server logs nothing.
 */
export const validateHandshake = (
  request: HandshakeRequest,
  options: { readonly supportedProtocols?: readonly string[] } = {},
): HandshakeResult => {
  const header = (name: string): string => request.headers[name.toLowerCase()] ?? '';

  if (request.method.toUpperCase() !== 'GET') {
    return reject(405, 'A WebSocket upgrade must use GET.');
  }

  if (header('upgrade').toLowerCase() !== 'websocket') {
    return reject(400, 'Missing or incorrect Upgrade header.');
  }

  // Connection is a comma-separated list and may carry other tokens.
  const connection = header('connection').toLowerCase();
  if (!connection.split(',').some((token) => token.trim() === 'upgrade')) {
    return reject(400, 'Connection header does not request an upgrade.');
  }

  const version = Number.parseInt(header('sec-websocket-version'), 10);
  if (version !== WEBSOCKET_VERSION) {
    return reject(
      426,
      `Unsupported WebSocket version ${header('sec-websocket-version') || '(absent)'}; ` +
        `this server speaks ${WEBSOCKET_VERSION}.`,
    );
  }

  const key = header('sec-websocket-key');
  // 16 bytes base64 is exactly 24 characters ending in '=='. A key of any other
  // shape means the client did not follow the specification, and accepting it
  // would mean completing a handshake with something that is not a WebSocket.
  if (!/^[A-Za-z0-9+/]{22}==$/.test(key)) {
    return reject(400, 'Missing or malformed Sec-WebSocket-Key.');
  }

  let selected: string | null = null;
  const supported = options.supportedProtocols ?? [];
  if (supported.length > 0) {
    const offered = header('sec-websocket-protocol')
      .split(',')
      .map((entry) => entry.trim())
      .filter((entry) => entry !== '');

    selected = offered.find((entry) => supported.includes(entry)) ?? null;
    if (offered.length > 0 && selected === null) {
      return reject(
        400,
        `None of the offered subprotocols (${offered.join(', ')}) is supported; ` +
          `this server speaks ${supported.join(', ')}.`,
      );
    }
  }

  return { ok: true, acceptKey: acceptKey(key), protocol: selected };
};

const reject = (status: number, reason: string): HandshakeResult => ({
  ok: false,
  rejection: { status, reason },
});

/** The 101 response completing an accepted upgrade. */
export const upgradeResponse = (accept: string, protocol: string | null): string => {
  const lines = [
    'HTTP/1.1 101 Switching Protocols',
    'Upgrade: websocket',
    'Connection: Upgrade',
    `Sec-WebSocket-Accept: ${accept}`,
  ];
  if (protocol !== null) lines.push(`Sec-WebSocket-Protocol: ${protocol}`);

  return `${lines.join('\r\n')}\r\n\r\n`;
};

/** The plain HTTP response for a refused upgrade. */
export const rejectionResponse = (rejection: HandshakeRejection): string => {
  const body = rejection.reason;
  return (
    `HTTP/1.1 ${rejection.status} ${statusText(rejection.status)}\r\n` +
    'Content-Type: text/plain; charset=utf-8\r\n' +
    `Content-Length: ${Buffer.byteLength(body, 'utf8')}\r\n` +
    (rejection.status === 426 ? `Sec-WebSocket-Version: ${WEBSOCKET_VERSION}\r\n` : '') +
    'Connection: close\r\n\r\n' +
    body
  );
};

const statusText = (status: number): string => {
  switch (status) {
    case 400:
      return 'Bad Request';
    case 401:
      return 'Unauthorized';
    case 403:
      return 'Forbidden';
    case 405:
      return 'Method Not Allowed';
    case 426:
      return 'Upgrade Required';
    case 429:
      return 'Too Many Requests';
    default:
      return 'Error';
  }
};

/**
 * Client-side check that the server really performed the transformation.
 *
 * Compared in constant time. The value is not secret, so this is not about
 * leaking it - it is about not making comparison timing a habit that gets copied
 * to somewhere it matters.
 */
export const verifyAcceptKey = (clientKey: string, serverAccept: string): boolean => {
  const expected = Buffer.from(acceptKey(clientKey), 'utf8');
  const actual = Buffer.from(serverAccept, 'utf8');

  if (expected.length !== actual.length) return false;
  return timingSafeEqual(expected, actual);
};
