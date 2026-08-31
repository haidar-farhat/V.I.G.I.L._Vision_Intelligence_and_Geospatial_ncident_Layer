import { createServer } from 'node:net';
import type { Server, Socket } from 'node:net';
import { createHash, randomBytes } from 'node:crypto';

/**
 * A mock RTSP camera.
 *
 * Real IP cameras are the least standards-compliant devices most engineers ever
 * integrate with, and every one is wrong differently. Without something to test
 * against, an RTSP client is written from a specification and discovers the truth
 * on a customer's site at two in the morning.
 *
 * So this server can be configured to behave the way real cameras actually do:
 * demanding Digest before answering anything, offering only Basic, sending
 * responses split across TCP segments, declaring a Content-Length it never
 * satisfies, offering no video track, or simply lying about the protocol.
 *
 * It speaks enough RTSP to complete a full handshake, and it never carries real
 * credentials - the fixture password exists to prove it is never echoed.
 */

export type MockRtspBehaviour = {
  readonly username?: string;
  readonly password?: string;
  /** Which schemes the 401 offers. Defaults to Digest only. */
  readonly authSchemes?: readonly ('Digest' | 'Basic')[];
  /** Skip authentication entirely, as an unsecured camera does. */
  readonly noAuth?: boolean;
  /** SDP to return from DESCRIBE. A sensible H.264 default is supplied. */
  readonly sdp?: string;
  /** Advertise GET_PARAMETER in OPTIONS. */
  readonly supportsGetParameter?: boolean;
  /** Write responses one byte at a time, to exercise incremental parsing. */
  readonly dribble?: boolean;
  /** Declare a Content-Length larger than the body actually sent. */
  readonly lieAboutContentLength?: boolean;
  /** Answer with something that is not RTSP at all. */
  readonly speakGarbage?: boolean;
  /** Accept the connection and never answer, to exercise the timeout. */
  readonly silent?: boolean;
  /** Fail SETUP with this status, e.g. 461 Unsupported Transport. */
  readonly setupStatus?: number;
  /** Session timeout advertised in the SETUP response. */
  readonly sessionTimeoutSeconds?: number;
};

export const DEFAULT_SDP = [
  'v=0',
  'o=- 2890844526 2890842807 IN IP4 192.168.1.50',
  's=Sentinel Test Camera',
  't=0 0',
  'a=control:*',
  'm=video 0 RTP/AVP 96',
  'a=rtpmap:96 H264/90000',
  'a=fmtp:96 packetization-mode=1;profile-level-id=42001f;sprop-parameter-sets=Z0IAH5WoFAFuQA==,aM48gA==',
  'a=control:trackID=1',
  'm=audio 0 RTP/AVP 97',
  'a=rtpmap:97 MPEG4-GENERIC/16000',
  'a=control:trackID=2',
].join('\r\n');

export type MockRtspServer = {
  readonly port: number;
  /** Every request line the server received, for asserting handshake order. */
  readonly requests: readonly string[];
  /** Authorization headers received, so a test can prove no credential leaked. */
  readonly authorizations: readonly string[];
  close(): Promise<void>;
};

export const startMockRtspServer = async (
  behaviour: MockRtspBehaviour = {},
): Promise<MockRtspServer> => {
  const requests: string[] = [];
  const authorizations: string[] = [];
  const nonce = randomBytes(8).toString('hex');
  const realm = 'Sentinel Test Camera';
  const schemes = behaviour.authSchemes ?? ['Digest'];

  /*
   * Every accepted socket is tracked so close() can destroy it.
   *
   * net.Server.close() stops accepting but waits for existing connections to end
   * on their own, and an RTSP client deliberately holds its session open. Without
   * this the server never closes, the event loop never drains, and the test
   * process hangs until the runner kills it - which is exactly what happened the
   * first time this suite ran.
   */
  const sockets = new Set<Socket>();

  const server: Server = createServer((socket: Socket) => {
    sockets.add(socket);
    socket.on('close', () => sockets.delete(socket));

    socket.setEncoding('utf8');
    let buffer = '';

    socket.on('data', (chunk: string) => {
      buffer += chunk;

      // Requests without bodies end at the blank line.
      let end = buffer.indexOf('\r\n\r\n');
      while (end !== -1) {
        const raw = buffer.slice(0, end);
        buffer = buffer.slice(end + 4);
        handleRequest(socket, raw);
        end = buffer.indexOf('\r\n\r\n');
      }
    });

    socket.on('error', () => {
      // A client destroying the socket mid-handshake is a case under test.
    });
  });

  const send = (socket: Socket, text: string): void => {
    if (behaviour.dribble !== true) {
      socket.write(text);
      return;
    }

    /*
     * Split the response at the boundaries that actually break parsers, rather
     * than one byte at a time.
     *
     * Byte-at-a-time is the obvious approach and it is both slower and weaker: it
     * spends thousands of ticks re-testing uninteresting offsets. What breaks a
     * real client is a segment ending mid-status-line, mid-header-name, and
     * mid-body - so those are the three cuts made here.
     */
    const headerEnd = text.indexOf('\r\n\r\n');
    const cuts = [
      8, // mid "RTSP/1.0 200 OK"
      Math.max(9, Math.floor((headerEnd === -1 ? text.length : headerEnd) / 2)), // mid-header
      headerEnd === -1 ? text.length : headerEnd + 6, // just past the blank line, mid-body
    ]
      .filter((cut) => cut > 0 && cut < text.length)
      .sort((a, b) => a - b);

    const chunks: string[] = [];
    let previous = 0;
    for (const cut of cuts) {
      chunks.push(text.slice(previous, cut));
      previous = cut;
    }
    chunks.push(text.slice(previous));

    let index = 0;
    const writeNext = (): void => {
      if (index >= chunks.length || socket.destroyed) return;
      socket.write(chunks[index] ?? '');
      index += 1;
      setImmediate(writeNext);
    };
    setImmediate(writeNext);
  };

  const respond = (
    socket: Socket,
    status: number,
    statusText: string,
    headers: Record<string, string>,
    body = '',
  ): void => {
    const declaredLength = behaviour.lieAboutContentLength === true
      ? Buffer.byteLength(body, 'utf8') + 500
      : Buffer.byteLength(body, 'utf8');

    const all: Record<string, string> = { ...headers };
    if (body !== '' || declaredLength > 0) all['Content-Length'] = String(declaredLength);

    const text =
      `RTSP/1.0 ${status} ${statusText}\r\n` +
      Object.entries(all)
        .map(([key, value]) => `${key}: ${value}`)
        .join('\r\n') +
      '\r\n\r\n' +
      body;

    send(socket, text);
  };

  const authorised = (headerBlock: string): boolean => {
    if (behaviour.noAuth === true) return true;

    const match = /^Authorization:\s*(.+)$/im.exec(headerBlock);
    if (match?.[1] === undefined) return false;

    const header = match[1].trim();
    authorizations.push(header);

    const username = behaviour.username ?? 'admin';
    const password = behaviour.password ?? 'correct-horse-battery';

    if (header.startsWith('Basic ')) {
      if (!schemes.includes('Basic')) return false;
      const decoded = Buffer.from(header.slice(6), 'base64').toString('utf8');
      return decoded === `${username}:${password}`;
    }

    if (header.startsWith('Digest ')) {
      const params: Record<string, string> = {};
      for (const m of header.slice(7).matchAll(/(\w+)="?([^",]*)"?/g)) {
        const key = m[1];
        if (key !== undefined) params[key] = m[2] ?? '';
      }

      const method = /^(\w+)\s/.exec(headerBlock)?.[1] ?? '';
      const uri = params['uri'] ?? '';
      const md5 = (input: string): string => createHash('md5').update(input).digest('hex');

      const ha1 = md5(`${username}:${realm}:${password}`);
      const ha2 = md5(`${method}:${uri}`);

      const expected =
        params['qop'] === 'auth'
          ? md5(`${ha1}:${params['nonce']}:${params['nc']}:${params['cnonce']}:auth:${ha2}`)
          : md5(`${ha1}:${params['nonce']}:${ha2}`);

      return params['response'] === expected;
    }

    return false;
  };

  const handleRequest = (socket: Socket, raw: string): void => {
    if (behaviour.silent === true) return;

    const requestLine = raw.split('\r\n')[0] ?? '';
    requests.push(requestLine);

    if (behaviour.speakGarbage === true) {
      send(socket, 'HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n');
      return;
    }

    const cseq = /^CSeq:\s*(\d+)/im.exec(raw)?.[1] ?? '1';
    const method = requestLine.split(' ')[0] ?? '';

    if (!authorised(raw)) {
      const challenges: string[] = [];
      if (schemes.includes('Digest')) {
        challenges.push(`Digest realm="${realm}", nonce="${nonce}", qop="auth", algorithm=MD5`);
      }
      if (schemes.includes('Basic')) {
        challenges.push(`Basic realm="${realm}"`);
      }
      respond(socket, 401, 'Unauthorized', {
        CSeq: cseq,
        'WWW-Authenticate': challenges.join(', '),
      });
      return;
    }

    switch (method) {
      case 'OPTIONS':
        respond(socket, 200, 'OK', {
          CSeq: cseq,
          Public: `OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN${
            behaviour.supportsGetParameter === true ? ', GET_PARAMETER' : ''
          }`,
        });
        break;

      case 'DESCRIBE':
        respond(
          socket,
          200,
          'OK',
          { CSeq: cseq, 'Content-Type': 'application/sdp' },
          behaviour.sdp ?? DEFAULT_SDP,
        );
        break;

      case 'SETUP': {
        const status = behaviour.setupStatus ?? 200;
        if (status !== 200) {
          respond(socket, status, 'Unsupported Transport', { CSeq: cseq });
          break;
        }
        const timeout = behaviour.sessionTimeoutSeconds ?? 60;
        respond(socket, 200, 'OK', {
          CSeq: cseq,
          Session: `12345678;timeout=${timeout}`,
          Transport: 'RTP/AVP/TCP;unicast;interleaved=0-1',
        });
        break;
      }

      case 'PLAY':
        respond(socket, 200, 'OK', { CSeq: cseq, Session: '12345678', RTPInfo: 'url=trackID=1' });
        break;

      case 'GET_PARAMETER':
      case 'TEARDOWN':
        respond(socket, 200, 'OK', { CSeq: cseq });
        break;

      default:
        respond(socket, 501, 'Not Implemented', { CSeq: cseq });
    }
  };

  const port = await new Promise<number>((resolve, reject) => {
    // Loopback only, and an ephemeral port so parallel tests never collide.
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') {
        reject(new Error('mock RTSP server did not bind a port'));
        return;
      }
      resolve(address.port);
    });
    server.once('error', reject);
  });

  return {
    port,
    requests,
    authorizations,
    close: () =>
      new Promise<void>((resolve) => {
        for (const socket of sockets) socket.destroy();
        sockets.clear();
        server.close(() => resolve());
      }),
  };
};
