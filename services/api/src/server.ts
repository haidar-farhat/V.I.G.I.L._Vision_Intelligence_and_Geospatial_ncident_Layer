import { createServer } from 'node:http';
import type { IncomingMessage, Server } from 'node:http';
import type { Duplex } from 'node:stream';
import type { CameraId, MapPackageId, NodeId, UtcMillis } from '@sentinel/shared-types';
import { Permission, asId, utcMillis } from '@sentinel/shared-types';
import { EgressGuard, RateLimiter, networkIsolationReport, redact, secret } from '@sentinel/security';
import {
  rejectionResponse,
  upgradeResponse,
  validateHandshake,
} from '@sentinel/protocol';
import type { SqlDriver } from '@sentinel/database';
import { CameraRepository, MapPackageRepository } from '@sentinel/database';
import { Authenticator, SessionStore } from './auth.ts';
import type { ScryptCost, Session, StoredUser } from './auth.ts';
import { ApiError, Router, headerValue, readJsonBody, writeResult } from './router.ts';
import type { AuditEntry, LogEntry } from './router.ts';
import { Hub, HEARTBEAT_INTERVAL_MILLIS } from './hub.ts';

/**
 * The control plane, assembled.
 *
 * Binds loopback by default. This serves a desktop application on the same
 * machine, and a control plane listening on every interface is a decision
 * somebody should have to make deliberately rather than inherit from a default.
 *
 * The routes here are the configuration surface: cameras, zones, map packages,
 * events, incidents and diagnostics. Every one declares the permission it
 * requires, and the audit sink sees every privileged call whether it succeeded,
 * was refused, or failed.
 */

export type ServerOptions = {
  readonly nodeId: NodeId;
  readonly database: SqlDriver;
  /** Loopback unless an operator deliberately widens it. */
  readonly host?: string;
  readonly port?: number;
  /** Resolve a user for login. Supplied by the caller so storage stays out of here. */
  readonly findUser: (username: string) => StoredUser | undefined;
  readonly egressGuard?: EgressGuard;
  readonly scryptCost?: ScryptCost;
  readonly onAudit?: (entry: AuditEntry) => void;
  readonly onLog?: (entry: LogEntry) => void;
  readonly now?: () => UtcMillis;
};

export type RunningServer = {
  readonly port: number;
  readonly host: string;
  readonly hub: Hub;
  readonly sessions: SessionStore;
  close(): Promise<void>;
};

export const startApiServer = async (options: ServerOptions): Promise<RunningServer> => {
  const host = options.host ?? '127.0.0.1';
  const now = options.now ?? (() => utcMillis(Date.now()));

  const sessions = new SessionStore({ now });
  const authenticator = new Authenticator(sessions, {
    now,
    ...(options.scryptCost === undefined ? {} : { cost: options.scryptCost }),
  });

  const hub = new Hub({
    nodeId: options.nodeId,
    now,
    onLog: (level, message, detail) =>
      options.onLog?.({ requestId: 'hub', level, message, detail }),
  });

  const cameras = new CameraRepository(options.database);
  const maps = new MapPackageRepository(options.database);
  const guard = options.egressGuard ?? new EgressGuard();

  const router = buildRouter({
    sessions,
    authenticator,
    cameras,
    maps,
    guard,
    hub,
    findUser: options.findUser,
    ...(options.onAudit === undefined ? {} : { onAudit: options.onAudit }),
    ...(options.onLog === undefined ? {} : { onLog: options.onLog }),
  });

  const http: Server = createServer((request, response) => {
    void (async () => {
      try {
        const body =
          request.method === 'POST' || request.method === 'PUT' || request.method === 'PATCH'
            ? await readJsonBody(request)
            : undefined;

        const result = await router.handle(
          request.method ?? 'GET',
          request.url ?? '/',
          request.headers,
          body,
          sourceAddressOf(request),
        );

        writeResult(response, result);
      } catch (error) {
        const status = error instanceof ApiError ? error.status : 500;
        const message =
          error instanceof ApiError ? error.message : 'The request could not be completed.';

        response.writeHead(status, { 'Content-Type': 'application/json' });
        response.end(JSON.stringify({ error: { code: 'REQUEST_FAILED', message } }));
      }
    })();
  });

  /**
   * WebSocket upgrade.
   *
   * The session is checked *before* the handshake completes. Upgrading first and
   * authenticating afterwards leaves a window in which an unauthenticated socket
   * is attached to the hub, and windows like that are how a subscription reaches
   * somebody who never logged in.
   */
  http.on('upgrade', (request: IncomingMessage, socket: Duplex) => {
    const url = new URL(request.url ?? '/', 'http://localhost');

    const token =
      url.searchParams.get('token') ??
      (headerValue(request.headers, 'authorization').startsWith('Bearer ')
        ? headerValue(request.headers, 'authorization').slice(7).trim()
        : '');

    const session: Session | undefined = token === '' ? undefined : sessions.get(token);

    if (session === undefined) {
      socket.write(
        rejectionResponse({ status: 401, reason: 'A valid session is required to subscribe.' }),
      );
      socket.destroy();
      return;
    }

    const handshake = validateHandshake(
      {
        method: request.method ?? 'GET',
        path: url.pathname,
        headers: normaliseHeaders(request.headers),
      },
      { supportedProtocols: ['sentinel.v1'] },
    );

    if (!handshake.ok) {
      socket.write(rejectionResponse(handshake.rejection));
      socket.destroy();
      return;
    }

    socket.write(upgradeResponse(handshake.acceptKey, handshake.protocol));
    hub.accept(socket, session);
  });

  const heartbeat = setInterval(() => {
    hub.heartbeat();
    sessions.prune();
  }, HEARTBEAT_INTERVAL_MILLIS);
  heartbeat.unref();

  const port = await new Promise<number>((resolve, reject) => {
    http.listen(options.port ?? 0, host, () => {
      const address = http.address();
      resolve(typeof address === 'object' && address !== null ? address.port : 0);
    });
    http.once('error', reject);
  });

  return {
    port,
    host,
    hub,
    sessions,
    close: async () => {
      clearInterval(heartbeat);
      hub.closeAll();
      await new Promise<void>((resolve) => {
        http.closeAllConnections();
        http.close(() => resolve());
      });
    },
  };
};

const normaliseHeaders = (
  headers: NodeJS.Dict<string | string[]>,
): Record<string, string> => {
  const result: Record<string, string> = {};
  for (const [key, value] of Object.entries(headers)) {
    if (value === undefined) continue;
    result[key.toLowerCase()] = Array.isArray(value) ? (value[0] ?? '') : value;
  }
  return result;
};

/** The peer address, for rate limiting. Loopback-only, so no proxy headers are trusted. */
const sourceAddressOf = (request: IncomingMessage): string =>
  request.socket.remoteAddress ?? 'unknown';

type RouterDependencies = {
  readonly sessions: SessionStore;
  readonly authenticator: Authenticator;
  readonly cameras: CameraRepository;
  readonly maps: MapPackageRepository;
  readonly guard: EgressGuard;
  readonly hub: Hub;
  readonly findUser: (username: string) => StoredUser | undefined;
  readonly onAudit?: (entry: AuditEntry) => void;
  readonly onLog?: (entry: LogEntry) => void;
};

const buildRouter = (deps: RouterDependencies): Router => {
  const router = new Router({
    resolveSession: (token) => deps.sessions.get(token),
    ...(deps.onAudit === undefined ? {} : { onAudit: deps.onAudit }),
    ...(deps.onLog === undefined ? {} : { onLog: deps.onLog }),
  });

  // ------------------------------------------------------------------- health
  router.add({
    method: 'GET',
    pattern: '/api/health',
    // Deliberately unauthenticated: the desktop shell polls this to decide
    // whether the service came up at all, before anyone has logged in.
    permission: null,
    handler: () => ({ status: 200, body: { status: 'ok', version: 1 } }),
  });

  // -------------------------------------------------------------------- login
  router.add({
    method: 'POST',
    pattern: '/api/auth/login',
    permission: null,
    // A second limiter in front of the authenticator's own, keyed by address, so
    // a flood cannot even reach the password hash.
    limit: { attempts: 20, windowMillis: 15 * 60 * 1000 },
    handler: (context) => {
      const body = context.body as { username?: unknown; password?: unknown } | undefined;

      if (typeof body?.username !== 'string' || typeof body.password !== 'string') {
        throw new ApiError(400, 'BAD_REQUEST', 'A username and password are required.', false);
      }

      const outcome = deps.authenticator.login(
        body.username,
        secret(body.password),
        context.sourceAddress,
        deps.findUser,
      );

      if (!outcome.ok) {
        return {
          status: outcome.reason === 'RATE_LIMITED' ? 429 : 401,
          body: {
            error: {
              code: outcome.reason,
              // Never says which half was wrong.
              message:
                outcome.reason === 'RATE_LIMITED'
                  ? 'Too many attempts. Try again shortly.'
                  : outcome.reason === 'ACCOUNT_DISABLED'
                    ? 'This account is disabled.'
                    : 'The username or password is incorrect.',
              recoverable: true,
            },
          },
          ...(outcome.retryAfterMillis === undefined
            ? {}
            : {
                headers: {
                  'Retry-After': String(Math.ceil(outcome.retryAfterMillis / 1000)),
                },
              }),
        };
      }

      return {
        status: 200,
        body: {
          token: outcome.session.token,
          username: outcome.session.username,
          roles: outcome.session.roles,
          expiresAt: outcome.session.expiresAt,
        },
      };
    },
  });

  router.add({
    method: 'POST',
    pattern: '/api/auth/logout',
    permission: Permission.CameraView,
    handler: (context) => {
      if (context.session !== null) {
        deps.sessions.revoke(context.session.token);
        // A logged-out operator's live subscriptions must not survive the logout.
        deps.hub.disconnectUser(context.session.username, 'signed out');
      }
      return { status: 204 };
    },
  });

  router.add({
    method: 'GET',
    pattern: '/api/auth/session',
    permission: Permission.CameraView,
    handler: (context) => ({
      status: 200,
      body: {
        username: context.session?.username,
        roles: context.session?.roles,
        expiresAt: context.session?.expiresAt,
      },
    }),
  });

  // ------------------------------------------------------------------ cameras
  router.add({
    method: 'GET',
    pattern: '/api/cameras',
    permission: Permission.CameraView,
    handler: () => ({ status: 200, body: { cameras: deps.cameras.list() } }),
  });

  router.add({
    method: 'GET',
    pattern: '/api/cameras/:cameraId',
    permission: Permission.CameraView,
    handler: (context) => {
      const camera = deps.cameras.get(asId<CameraId>(context.params['cameraId'] ?? ''));
      if (camera === undefined) throw new ApiError(404, 'NOT_FOUND', 'No such camera.', false);

      return {
        status: 200,
        body: { camera, profiles: deps.cameras.profiles(camera.id) },
      };
    },
  });

  router.add({
    method: 'DELETE',
    pattern: '/api/cameras/:cameraId',
    // High-risk: the UI confirms, and this is audited whatever the outcome.
    permission: Permission.CameraDelete,
    handler: (context) => {
      const result = deps.cameras.remove(asId<CameraId>(context.params['cameraId'] ?? ''));
      if (!result.removed) throw new ApiError(404, 'NOT_FOUND', 'No such camera.', false);

      return {
        status: 200,
        // The caller purges the keychain entry; leaving it would orphan a secret
        // for a camera that no longer exists.
        body: { removed: true, credentialsRef: result.credentialsRef },
      };
    },
  });

  router.add({
    method: 'GET',
    pattern: '/api/cameras/:cameraId/topology',
    permission: Permission.CameraView,
    handler: () => ({ status: 200, body: { edges: deps.cameras.topology() } }),
  });

  // ------------------------------------------------------------------ mapping
  router.add({
    method: 'GET',
    pattern: '/api/maps',
    permission: Permission.CameraView,
    handler: () => ({
      status: 200,
      body: {
        packages: deps.maps.list(),
        default: deps.maps.default() ?? null,
        totalSizeBytes: deps.maps.totalSizeBytes(),
      },
    }),
  });

  router.add({
    method: 'POST',
    pattern: '/api/maps/:packageId/default',
    permission: Permission.MapImport,
    handler: (context) => {
      const ok = deps.maps.setDefault(asId<MapPackageId>(context.params['packageId'] ?? ''));
      if (!ok) throw new ApiError(404, 'NOT_FOUND', 'No such map package.', false);
      return { status: 204 };
    },
  });

  router.add({
    method: 'DELETE',
    pattern: '/api/maps/:packageId',
    permission: Permission.MapImport,
    handler: (context) => {
      const result = deps.maps.remove(asId<MapPackageId>(context.params['packageId'] ?? ''));
      if (!result.removed) throw new ApiError(404, 'NOT_FOUND', 'No such map package.', false);
      return { status: 200, body: { removed: true, relativePath: result.relativePath } };
    },
  });

  // -------------------------------------------------------------- diagnostics
  router.add({
    method: 'GET',
    pattern: '/api/diagnostics/network',
    permission: Permission.DiagnosticsRun,
    handler: () => ({
      status: 200,
      // The panel an auditor reads the offline guarantee off, rather than
      // inferring it from configuration.
      body: networkIsolationReport(deps.guard, true),
    }),
  });

  router.add({
    method: 'GET',
    pattern: '/api/diagnostics/realtime',
    permission: Permission.DiagnosticsRun,
    handler: () => ({
      status: 200,
      body: {
        connections: deps.hub.connectionCount,
        subscribers: {
          events: deps.hub.subscriberCount('events'),
          tracks: deps.hub.subscriberCount('tracks'),
          incidents: deps.hub.subscriberCount('incidents'),
          camera_status: deps.hub.subscriberCount('camera_status'),
          nodes: deps.hub.subscriberCount('nodes'),
          system: deps.hub.subscriberCount('system'),
        },
        activeSessions: deps.sessions.activeCount,
      },
    }),
  });

  return router;
};

/** A rate limiter for anything the caller wants to guard outside a route. */
export const sharedLimiter = (attempts: number, windowMillis: number): RateLimiter =>
  new RateLimiter(attempts, windowMillis);

export { redact };
